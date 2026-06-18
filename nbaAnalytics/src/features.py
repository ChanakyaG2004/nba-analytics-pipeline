import pandas as pd


FEATURE_COLUMNS = [
    "period",
    "seconds_remaining",
    "home_score",
    "away_score",
    "score_margin",
    "abs_score_margin",
    "total_score",
    "score_margin_per_minute",
    "is_home_leading",
    "is_tied",
    "is_late_game",
    "is_scoring_play",
]


def build_feature_frame(
    period,
    seconds_remaining,
    home_score,
    away_score,
    scoring_play=False,
):
    score_margin = home_score - away_score
    score_margin_per_minute = score_margin / max(seconds_remaining / 60, 1 / 60)
    return pd.DataFrame(
        [
            {
                "period": period,
                "seconds_remaining": seconds_remaining,
                "home_score": home_score,
                "away_score": away_score,
                "score_margin": score_margin,
                "abs_score_margin": abs(score_margin),
                "total_score": home_score + away_score,
                "score_margin_per_minute": score_margin_per_minute,
                "is_home_leading": int(score_margin > 0),
                "is_tied": int(score_margin == 0),
                "is_late_game": int(seconds_remaining <= 300),
                "is_scoring_play": int(scoring_play),
            }
        ],
        columns=FEATURE_COLUMNS,
    )
