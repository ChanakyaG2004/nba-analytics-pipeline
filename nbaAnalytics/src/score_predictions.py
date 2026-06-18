import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS
from train_model import DEFAULT_MODEL_PATH


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:password123@localhost:5432/nba_db",
)


CREATE_PREDICTIONS_SQL = """
CREATE TABLE IF NOT EXISTS model_predictions (
    game_id TEXT NOT NULL,
    play_id TEXT NOT NULL,
    model_path TEXT NOT NULL,
    home_win_probability DOUBLE PRECISION NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_id, play_id, model_path),
    FOREIGN KEY (game_id, play_id)
        REFERENCES play_by_play(game_id, play_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_model_predictions_game
    ON model_predictions (game_id, scored_at DESC);
"""


def load_scoring_rows(engine, game_id=None):
    params = {}
    game_filter = ""

    if game_id:
        game_filter = "AND game_id = :game_id"
        params["game_id"] = game_id

    query = text(
        f"""
        SELECT
            game_id,
            play_id,
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
          {game_filter}
        ORDER BY game_id, sequence_number
        """
    )
    return pd.read_sql(query, engine, params=params)


def build_feature_frame(rows):
    features = rows.copy()
    features["abs_score_margin"] = features["score_margin"].abs()
    features["total_score"] = features["home_score"] + features["away_score"]
    features["score_margin_per_minute"] = features["score_margin"] / (
        features["seconds_remaining"].clip(lower=1) / 60
    )
    features["is_home_leading"] = (features["score_margin"] > 0).astype(int)
    features["is_tied"] = (features["score_margin"] == 0).astype(int)
    features["is_late_game"] = (features["seconds_remaining"] <= 300).astype(int)
    features["is_scoring_play"] = features["scoring_play"].fillna(False).astype(int)
    return features[FEATURE_COLUMNS]


def ensure_prediction_schema(conn):
    for statement in CREATE_PREDICTIONS_SQL.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(text(statement))


def upsert_predictions(engine, predictions):
    if not predictions:
        return

    with engine.begin() as conn:
        ensure_prediction_schema(conn)
        conn.execute(
            text(
                """
                INSERT INTO model_predictions (
                    game_id, play_id, model_path, home_win_probability
                )
                VALUES (
                    :game_id, :play_id, :model_path, :home_win_probability
                )
                ON CONFLICT (game_id, play_id, model_path) DO UPDATE SET
                    home_win_probability = EXCLUDED.home_win_probability,
                    scored_at = NOW()
                """
            ),
            predictions,
        )


def score_predictions(args):
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model_path}")

    engine = create_engine(DATABASE_URL)
    rows = load_scoring_rows(engine, args.game_id)

    if rows.empty:
        return {
            "model_path": str(args.model_path),
            "scored_rows": 0,
            "game_id": args.game_id,
        }

    model = XGBClassifier()
    model.load_model(args.model_path)
    probabilities = model.predict_proba(build_feature_frame(rows))[:, 1]

    output = pd.DataFrame(
        {
            "game_id": rows["game_id"],
            "play_id": rows["play_id"],
            "model_path": str(args.model_path),
            "home_win_probability": probabilities,
        }
    )

    records = output.to_dict(orient="records")
    for start in range(0, len(records), args.batch_size):
        upsert_predictions(engine, records[start : start + args.batch_size])

    return {
        "model_path": str(args.model_path),
        "scored_rows": len(records),
        "game_id": args.game_id,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Score warehouse plays with the trained model.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--game-id", help="Optional single game to score.")
    parser.add_argument("--batch-size", type=int, default=5000)
    return parser


if __name__ == "__main__":
    print(json.dumps(score_predictions(build_parser().parse_args()), indent=2))
