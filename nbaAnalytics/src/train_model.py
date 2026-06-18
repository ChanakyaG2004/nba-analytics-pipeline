import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:password123@localhost:5432/nba_db",
)
DEFAULT_MODEL_PATH = Path("models/nba_xgb_win_probability.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts")


def load_training_rows(engine):
    query = """
        SELECT
            game_id,
            sequence_number,
            period,
            seconds_remaining,
            home_score,
            away_score,
            score_margin,
            scoring_play
        FROM play_by_play
        WHERE home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND score_margin IS NOT NULL
          AND seconds_remaining IS NOT NULL
        ORDER BY game_id, sequence_number
    """
    return pd.read_sql(query, engine)


def build_training_frame(plays):
    if plays.empty:
        return plays

    plays = plays.sort_values(["game_id", "sequence_number"]).copy()
    final_scores = (
        plays.groupby("game_id", as_index=False)
        .tail(1)[["game_id", "home_score", "away_score"]]
        .rename(
            columns={
                "home_score": "final_home_score",
                "away_score": "final_away_score",
            }
        )
    )
    frame = plays.merge(final_scores, on="game_id", how="inner")
    frame["home_win"] = (frame["final_home_score"] > frame["final_away_score"]).astype(int)
    frame["abs_score_margin"] = frame["score_margin"].abs()
    frame["total_score"] = frame["home_score"] + frame["away_score"]
    frame["score_margin_per_minute"] = frame["score_margin"] / (
        (frame["seconds_remaining"].clip(lower=1) / 60)
    )
    frame["is_home_leading"] = (frame["score_margin"] > 0).astype(int)
    frame["is_tied"] = (frame["score_margin"] == 0).astype(int)
    frame["is_late_game"] = (frame["seconds_remaining"] <= 300).astype(int)
    frame["is_scoring_play"] = frame["scoring_play"].fillna(False).astype(int)
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["home_win"])
    return frame


def split_games(frame, test_size, random_state):
    games = (
        frame.groupby("game_id", as_index=False)["home_win"]
        .max()
        .rename(columns={"home_win": "label"})
    )

    if games["label"].nunique() < 2:
        raise ValueError("Training data must include both home wins and home losses.")

    rng = np.random.default_rng(random_state)
    test_games = set()

    for _, class_games in games.groupby("label"):
        class_ids = class_games["game_id"].to_numpy()
        class_test_count = max(1, int(round(len(class_ids) * test_size)))
        test_games.update(rng.permutation(class_ids)[:class_test_count])

    train_games = set(games["game_id"]) - test_games
    train_labels = set(games[games["game_id"].isin(train_games)]["label"])
    test_labels = set(games[games["game_id"].isin(test_games)]["label"])

    if len(train_labels) < 2 or len(test_labels) < 2:
        raise ValueError(
            "Could not create a stratified train/test split with both classes. "
            "Backfill more games before training."
        )

    return train_games, test_games


def calibration_summary(y_true, y_prob, bins):
    data = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    data["bin"] = pd.cut(data["y_prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    summary = (
        data.groupby("bin", observed=True)
        .agg(
            samples=("y_true", "size"),
            predicted_probability=("y_prob", "mean"),
            observed_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    summary["calibration_error"] = (
        summary["predicted_probability"] - summary["observed_rate"]
    ).abs()
    weighted_error = np.average(summary["calibration_error"], weights=summary["samples"])
    return summary, float(1 - weighted_error)


def train_model(frame, train_games, test_games, random_state):
    train = frame[frame["game_id"].isin(train_games)]
    test = frame[frame["game_id"].isin(test_games)]

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=2,
    )
    model.fit(train[FEATURE_COLUMNS], train["home_win"])

    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    calibration, calibration_accuracy = calibration_summary(
        test["home_win"].to_numpy(),
        probabilities,
        bins=10,
    )

    metrics = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_games": int(len(train_games)),
        "test_games": int(len(test_games)),
        "features": FEATURE_COLUMNS,
        "brier_score": float(brier_score_loss(test["home_win"], probabilities)),
        "log_loss": float(log_loss(test["home_win"], probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(test["home_win"], probabilities)),
        "calibration_accuracy": calibration_accuracy,
    }
    return model, metrics, calibration


def write_artifacts(model, metrics, calibration, model_path, artifact_dir):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(model_path)
    (artifact_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    calibration.to_csv(artifact_dir / "calibration_bins.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False)


def record_training_run(engine, metrics, model_path):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS model_training_runs (
                    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    model_path TEXT NOT NULL,
                    train_games INTEGER NOT NULL,
                    test_games INTEGER NOT NULL,
                    train_rows INTEGER NOT NULL,
                    test_rows INTEGER NOT NULL,
                    brier_score DOUBLE PRECISION NOT NULL,
                    log_loss DOUBLE PRECISION NOT NULL,
                    roc_auc DOUBLE PRECISION NOT NULL,
                    calibration_accuracy DOUBLE PRECISION NOT NULL,
                    metrics_json JSONB NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO model_training_runs (
                    model_path, train_games, test_games, train_rows, test_rows,
                    brier_score, log_loss, roc_auc, calibration_accuracy, metrics_json
                )
                VALUES (
                    :model_path, :train_games, :test_games, :train_rows, :test_rows,
                    :brier_score, :log_loss, :roc_auc, :calibration_accuracy,
                    CAST(:metrics_json AS JSONB)
                )
                """
            ),
            {
                **metrics,
                "model_path": str(model_path),
                "metrics_json": json.dumps(metrics),
            },
        )


def run_training(args):
    engine = create_engine(DATABASE_URL)
    plays = load_training_rows(engine)
    frame = build_training_frame(plays)
    game_count = frame["game_id"].nunique() if not frame.empty else 0

    if game_count < args.min_games:
        raise ValueError(
            f"Found {game_count} trainable games, but --min-games is {args.min_games}. "
            "Backfill more dates with src/etl_pipeline.py before training."
        )

    train_games, test_games = split_games(frame, args.test_size, args.random_state)
    model, metrics, calibration = train_model(frame, train_games, test_games, args.random_state)
    write_artifacts(model, metrics, calibration, args.model_path, args.artifact_dir)
    record_training_run(engine, metrics, args.model_path)
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Train NBA win-probability XGBoost model.")
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser


if __name__ == "__main__":
    metrics = run_training(build_parser().parse_args())
    print(json.dumps(metrics, indent=2))
