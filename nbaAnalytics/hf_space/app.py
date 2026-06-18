import json
from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "nba_xgb_win_probability.json"
METRICS_PATH = BASE_DIR / "artifacts" / "training_metrics.json"

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


@st.cache_data
def load_demo_data():
    games = pd.read_csv(DATA_DIR / "games.csv")
    plays = pd.read_csv(DATA_DIR / "plays_with_predictions.csv")
    return games, plays


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def build_features(period, seconds_remaining, home_score, away_score, scoring_play):
    score_margin = home_score - away_score
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
                "score_margin_per_minute": score_margin / max(seconds_remaining / 60, 1 / 60),
                "is_home_leading": int(score_margin > 0),
                "is_tied": int(score_margin == 0),
                "is_late_game": int(seconds_remaining <= 300),
                "is_scoring_play": int(scoring_play),
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def format_game(games, game_id):
    row = games.loc[games["game_id"] == game_id].iloc[0]
    label = row["short_name"] if pd.notna(row["short_name"]) else game_id
    return f"{label} · {row['game_date']}"


st.set_page_config(page_title="NBA AI Analytics", layout="wide")
st.title("NBA AI Analytics Dashboard")

games, plays = load_demo_data()
model = load_model()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

st.sidebar.header("Game")
selected_game_id = st.sidebar.selectbox(
    "Select game",
    games["game_id"].tolist(),
    format_func=lambda game_id: format_game(games, game_id),
)

game = games.loc[games["game_id"] == selected_game_id].iloc[0]
game_plays = plays[plays["game_id"] == selected_game_id].sort_values("sequence_number")

st.subheader(game["short_name"])
st.caption(f"{game['game_date']} · {game['status']}")

latest_prediction = game_plays.dropna(subset=["home_win_probability"]).tail(1)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Games In Demo", len(games))
col2.metric("Scored Plays", int(plays["home_win_probability"].notna().sum()))
col3.metric("Model Calibration", f"{metrics['calibration_accuracy'] * 100:.1f}%")
if latest_prediction.empty:
    col4.metric("Latest Home Win Prob", "n/a")
else:
    col4.metric("Latest Home Win Prob", f"{latest_prediction['home_win_probability'].iloc[0] * 100:.1f}%")

chart_col, table_col = st.columns([3, 2])

with chart_col:
    st.subheader("Win Probability")
    probability = game_plays[["sequence_number", "home_win_probability"]].dropna()
    if probability.empty:
        st.info("No scored plays are available for this game.")
    else:
        st.line_chart(probability.set_index("sequence_number"))

with table_col:
    st.subheader("Score Margin")
    margin = game_plays[["sequence_number", "score_margin"]].dropna()
    if margin.empty:
        st.info("No score margin values are available for this game.")
    else:
        st.line_chart(margin.set_index("sequence_number"))

st.subheader("Ad Hoc Prediction")
pred_col1, pred_col2, pred_col3, pred_col4, pred_col5 = st.columns(5)
period = pred_col1.number_input("Period", min_value=1, max_value=8, value=4)
seconds_remaining = pred_col2.number_input("Seconds Left", min_value=0, max_value=2880, value=300)
home_score = pred_col3.number_input("Home Score", min_value=0, value=98)
away_score = pred_col4.number_input("Away Score", min_value=0, value=95)
scoring_play = pred_col5.checkbox("Scoring Play")

features = build_features(period, seconds_remaining, home_score, away_score, scoring_play)
probability = float(model.predict_proba(features)[0][1])
st.metric("Predicted Home Win Probability", f"{probability * 100:.1f}%")

st.subheader("Play Log")
display_columns = [
    "sequence_number",
    "period",
    "clock_display",
    "home_score",
    "away_score",
    "score_margin",
    "home_win_probability",
    "play_type_text",
    "text",
]
st.dataframe(game_plays[display_columns], width="stretch", hide_index=True)

with st.expander("Model Metrics"):
    st.json(metrics)
