import argparse
import json
from pathlib import Path

from xgboost import XGBClassifier

from features import build_feature_frame
from train_model import DEFAULT_MODEL_PATH


def build_feature_row(args):
    return build_feature_frame(
        period=args.period,
        seconds_remaining=args.seconds_remaining,
        home_score=args.home_score,
        away_score=args.away_score,
        scoring_play=args.scoring_play,
    )


def predict(args):
    model = XGBClassifier()
    model.load_model(args.model_path)
    features = build_feature_row(args)
    probability = float(model.predict_proba(features)[0][1])
    return {
        "home_win_probability": probability,
        "model_path": str(args.model_path),
        "features": features.iloc[0].to_dict(),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run local win-probability inference.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--seconds-remaining", type=int, required=True)
    parser.add_argument("--home-score", type=int, required=True)
    parser.add_argument("--away-score", type=int, required=True)
    parser.add_argument("--scoring-play", action="store_true")
    return parser


if __name__ == "__main__":
    print(json.dumps(predict(build_parser().parse_args()), indent=2))
