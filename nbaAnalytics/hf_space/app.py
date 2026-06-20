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


st.set_page_config(
    page_title="NBA AI Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --ink: #111827;
            --muted: #6b7280;
            --line: #d8dee9;
            --surface: #ffffff;
            --soft: #f5f7fb;
            --accent: #d64027;
            --accent-2: #146c94;
            --good: #15803d;
            --warn: #b45309;
        }

        .stApp {
            background: linear-gradient(180deg, rgba(245, 247, 251, 0.98), rgba(255, 255, 255, 1) 340px);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: #111827;
            color: #f9fafb;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #f9fafb;
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        .hero {
            border-bottom: 1px solid var(--line);
            padding: 0.2rem 0 1.35rem 0;
            margin-bottom: 1.1rem;
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .hero-title {
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 4.5rem);
            font-weight: 900;
            line-height: 0.95;
            margin: 0;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.55;
            margin-top: 0.95rem;
            max-width: 920px;
        }

        .metric-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            min-height: 118px;
            box-shadow: 0 16px 34px rgba(17, 24, 39, 0.06);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: 1.75rem;
            font-weight: 900;
            line-height: 1.1;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 0.45rem;
        }

        .section-title {
            font-size: 1rem;
            font-weight: 900;
            color: var(--ink);
            margin: 1.3rem 0 0.45rem 0;
        }

        .game-strip {
            background: #111827;
            border: 1px solid #253044;
            border-radius: 8px;
            color: #f9fafb;
            padding: 1rem 1.05rem;
            margin: 0.7rem 0 1rem 0;
        }

        .game-title {
            color: #ffffff;
            font-size: 1.55rem;
            font-weight: 900;
            margin: 0;
        }

        .game-meta {
            color: #cbd5e1;
            margin-top: 0.28rem;
            font-size: 0.92rem;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.85rem;
        }

        .pill {
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            color: #f8fafc;
            padding: 0.32rem 0.65rem;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .callout {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            color: #7c2d12;
            padding: 0.8rem 0.95rem;
            font-size: 0.92rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.65rem 0.95rem;
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_data():
    games = pd.read_csv(DATA_DIR / "games.csv")
    plays = pd.read_csv(DATA_DIR / "plays_with_predictions.csv")

    games["game_id"] = games["game_id"].astype(str)
    plays["game_id"] = plays["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    games["game_day"] = games["game_date"].dt.date
    games["month"] = games["game_date"].dt.strftime("%B %Y")

    numeric_columns = [
        "sequence_number",
        "period",
        "seconds_remaining",
        "home_score",
        "away_score",
        "score_margin",
        "home_win_probability",
    ]
    for column in numeric_columns:
        plays[column] = pd.to_numeric(plays[column], errors="coerce")

    return games, plays


@st.cache_data(show_spinner=False)
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data(show_spinner=False)
def build_game_stats(games, plays):
    rows = []
    for game_id, game_plays in plays.groupby("game_id", sort=False):
        scored = game_plays.dropna(subset=["home_score", "away_score"]).sort_values("sequence_number")
        predicted = game_plays.dropna(subset=["home_win_probability"]).sort_values("sequence_number")

        if scored.empty:
            continue

        final = scored.iloc[-1]
        opening_prob = float(predicted["home_win_probability"].iloc[0]) if not predicted.empty else None
        final_prob = float(predicted["home_win_probability"].iloc[-1]) if not predicted.empty else None
        min_prob = float(predicted["home_win_probability"].min()) if not predicted.empty else None
        max_prob = float(predicted["home_win_probability"].max()) if not predicted.empty else None
        score_margin = scored["score_margin"].dropna()
        signs = score_margin.apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
        lead_changes = int(((signs.shift(1) * signs) < 0).sum())
        home_score = int(final["home_score"])
        away_score = int(final["away_score"])

        rows.append(
            {
                "game_id": game_id,
                "home_score_final": home_score,
                "away_score_final": away_score,
                "winner": "Home" if home_score > away_score else "Away",
                "total_score": home_score + away_score,
                "final_margin": home_score - away_score,
                "abs_final_margin": abs(home_score - away_score),
                "opening_home_win_probability": opening_prob,
                "final_home_win_probability": final_prob,
                "win_probability_swing": (max_prob - min_prob) if min_prob is not None else None,
                "lead_changes": lead_changes,
                "largest_home_lead": int(score_margin.max()) if not score_margin.empty else None,
                "largest_away_lead": int(abs(score_margin.min())) if not score_margin.empty else None,
                "scored_plays": int(predicted["home_win_probability"].notna().sum()),
                "total_plays": int(len(game_plays)),
            }
        )

    stats = pd.DataFrame(rows)
    return games.merge(stats, on="game_id", how="left")


def build_team_stats(game_stats):
    rows = []
    teams = sorted(
        set(game_stats["home_team_abbrev"].dropna().astype(str))
        | set(game_stats["away_team_abbrev"].dropna().astype(str))
    )

    for team in teams:
        home_games = game_stats[game_stats["home_team_abbrev"] == team]
        away_games = game_stats[game_stats["away_team_abbrev"] == team]
        played = pd.concat([home_games, away_games], ignore_index=True)
        if played.empty:
            continue

        wins = int(
            (home_games["home_score_final"] > home_games["away_score_final"]).sum()
            + (away_games["away_score_final"] > away_games["home_score_final"]).sum()
        )
        points_for = pd.concat(
            [home_games["home_score_final"], away_games["away_score_final"]],
            ignore_index=True,
        )
        points_allowed = pd.concat(
            [home_games["away_score_final"], away_games["home_score_final"]],
            ignore_index=True,
        )

        rows.append(
            {
                "team": team,
                "games": int(len(played)),
                "wins": wins,
                "losses": int(len(played) - wins),
                "win_pct": wins / len(played) if len(played) else 0,
                "avg_points_for": points_for.mean(),
                "avg_points_allowed": points_allowed.mean(),
                "avg_margin": (points_for - points_allowed).mean(),
                "close_games": int((played["abs_final_margin"] <= 5).sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["win_pct", "avg_margin"], ascending=False)


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


def pct(value):
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def number(value):
    if pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def render_metric(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_game_label(row):
    date = row["game_date"].strftime("%b %-d, %Y") if pd.notna(row["game_date"]) else "Unknown date"
    score = ""
    if pd.notna(row.get("home_score_final")) and pd.notna(row.get("away_score_final")):
        score = f" - {int(row['away_score_final'])}-{int(row['home_score_final'])}"
    return f"{row['short_name']} - {date}{score}"


games, plays = load_demo_data()
metrics = load_metrics()
model = load_model()
game_stats = build_game_stats(games, plays)
team_stats = build_team_stats(game_stats)

teams = sorted(
    set(games["home_team_abbrev"].dropna().astype(str))
    | set(games["away_team_abbrev"].dropna().astype(str))
)
months = [month for month in games["month"].dropna().unique()]

with st.sidebar:
    st.markdown("## NBA AI Analytics")
    st.markdown("2025-26 season explorer")
    selected_team = st.selectbox("Team", ["All teams"] + teams)
    selected_month = st.selectbox("Month", ["All months"] + list(months))
    min_swing = st.slider("Minimum win-probability swing", 0, 100, 0, step=5)

filtered_games = game_stats.copy()
if selected_team != "All teams":
    filtered_games = filtered_games[
        (filtered_games["home_team_abbrev"] == selected_team)
        | (filtered_games["away_team_abbrev"] == selected_team)
    ]
if selected_month != "All months":
    filtered_games = filtered_games[filtered_games["month"] == selected_month]
filtered_games = filtered_games[
    filtered_games["win_probability_swing"].fillna(0) >= (min_swing / 100)
].sort_values("game_date", ascending=False)

if filtered_games.empty:
    st.warning("No games match the current filters.")
    st.stop()

selected_game_id = st.sidebar.selectbox(
    "Game",
    filtered_games["game_id"].tolist(),
    format_func=lambda game_id: format_game_label(
        filtered_games.loc[filtered_games["game_id"] == game_id].iloc[0]
    ),
)

game = game_stats.loc[game_stats["game_id"] == selected_game_id].iloc[0]
game_plays = plays[plays["game_id"] == selected_game_id].sort_values("sequence_number")
predicted_plays = game_plays.dropna(subset=["home_win_probability"]).copy()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">2025-26 NBA Season</div>
        <h1 class="hero-title">NBA AI Analytics Dashboard</h1>
        <div class="hero-copy">
            Full-game win probability, team performance, and play-level model output from the
            2025-26 season loaded into this project.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(5)
with metric_cols[0]:
    render_metric("Games", number(len(games)), "2025-26 season games in the public dataset")
with metric_cols[1]:
    render_metric("Scored Plays", number(plays["home_win_probability"].notna().sum()), "model predictions on play state")
with metric_cols[2]:
    render_metric("Calibration", pct(metrics["calibration_accuracy"]), "latest XGBoost training run")
with metric_cols[3]:
    render_metric("ROC AUC", f"{metrics['roc_auc']:.3f}", "held-out game split")
with metric_cols[4]:
    render_metric(
        "Avg Swing",
        pct(game_stats["win_probability_swing"].mean()),
        "average in-game probability range",
    )

tabs = st.tabs(["Game Center", "Season Board", "Team Stats", "Prediction Lab", "Model"])

with tabs[0]:
    game_date = game["game_date"].strftime("%B %-d, %Y") if pd.notna(game["game_date"]) else "Unknown date"
    final_score = "n/a"
    if pd.notna(game["home_score_final"]) and pd.notna(game["away_score_final"]):
        final_score = (
            f"{game['away_team_abbrev']} {int(game['away_score_final'])} - "
            f"{game['home_team_abbrev']} {int(game['home_score_final'])}"
        )

    st.markdown(
        f"""
        <div class="game-strip">
            <div class="game-title">{game['short_name']}</div>
            <div class="game-meta">{game_date} - {game['status']} - {final_score}</div>
            <div class="pill-row">
                <span class="pill">Final home win probability {pct(game['final_home_win_probability'])}</span>
                <span class="pill">Win-probability swing {pct(game['win_probability_swing'])}</span>
                <span class="pill">Lead changes {number(game['lead_changes'])}</span>
                <span class="pill">Total plays {number(game['total_plays'])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col, stat_col = st.columns([2.1, 1])
    with chart_col:
        st.markdown('<div class="section-title">Win Probability Timeline</div>', unsafe_allow_html=True)
        if predicted_plays.empty:
            st.info("No model predictions are available for this game.")
        else:
            probability_chart = predicted_plays[["sequence_number", "home_win_probability"]].rename(
                columns={"sequence_number": "Play", "home_win_probability": "Home win probability"}
            )
            st.line_chart(probability_chart, x="Play", y="Home win probability", height=360)

    with stat_col:
        st.markdown('<div class="section-title">Game Snapshot</div>', unsafe_allow_html=True)
        snap_cols = st.columns(2)
        with snap_cols[0]:
            render_metric("Home", number(game["home_score_final"]), game["home_team_abbrev"])
        with snap_cols[1]:
            render_metric("Away", number(game["away_score_final"]), game["away_team_abbrev"])
        render_metric("Largest Home Lead", number(game["largest_home_lead"]), "points")
        render_metric("Largest Away Lead", number(game["largest_away_lead"]), "points")

    st.markdown('<div class="section-title">Score Margin Timeline</div>', unsafe_allow_html=True)
    margin_chart = game_plays[["sequence_number", "score_margin"]].dropna().rename(
        columns={"sequence_number": "Play", "score_margin": "Home score margin"}
    )
    st.line_chart(margin_chart, x="Play", y="Home score margin", height=260)

    st.markdown('<div class="section-title">Play Log</div>', unsafe_allow_html=True)
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
    st.dataframe(
        game_plays[display_columns],
        width="stretch",
        height=520,
        hide_index=True,
        column_config={
            "sequence_number": st.column_config.NumberColumn("Play", width="small"),
            "period": st.column_config.NumberColumn("Q", width="small"),
            "clock_display": st.column_config.TextColumn("Clock", width="small"),
            "home_win_probability": st.column_config.ProgressColumn(
                "Home WP",
                min_value=0,
                max_value=1,
                format="%.3f",
            ),
            "text": st.column_config.TextColumn("Play", width="large"),
        },
    )

with tabs[1]:
    st.markdown('<div class="section-title">All 2025-26 Games</div>', unsafe_allow_html=True)
    board = game_stats[
        [
            "game_date",
            "short_name",
            "away_team_abbrev",
            "away_score_final",
            "home_team_abbrev",
            "home_score_final",
            "abs_final_margin",
            "total_score",
            "win_probability_swing",
            "lead_changes",
            "final_home_win_probability",
        ]
    ].sort_values("game_date", ascending=False)
    st.dataframe(
        board,
        width="stretch",
        height=620,
        hide_index=True,
        column_config={
            "game_date": st.column_config.DatetimeColumn("Date", format="MMM D, YYYY"),
            "short_name": "Matchup",
            "away_team_abbrev": "Away",
            "home_team_abbrev": "Home",
            "away_score_final": st.column_config.NumberColumn("Away Pts"),
            "home_score_final": st.column_config.NumberColumn("Home Pts"),
            "abs_final_margin": st.column_config.NumberColumn("Margin"),
            "total_score": st.column_config.NumberColumn("Total"),
            "win_probability_swing": st.column_config.ProgressColumn("WP Swing", min_value=0, max_value=1),
            "lead_changes": st.column_config.NumberColumn("Lead Changes"),
            "final_home_win_probability": st.column_config.ProgressColumn("Final Home WP", min_value=0, max_value=1),
        },
    )

with tabs[2]:
    st.markdown('<div class="section-title">Team Performance</div>', unsafe_allow_html=True)
    st.dataframe(
        team_stats,
        width="stretch",
        height=620,
        hide_index=True,
        column_config={
            "team": "Team",
            "games": st.column_config.NumberColumn("Games"),
            "wins": st.column_config.NumberColumn("Wins"),
            "losses": st.column_config.NumberColumn("Losses"),
            "win_pct": st.column_config.ProgressColumn("Win %", min_value=0, max_value=1, format="%.3f"),
            "avg_points_for": st.column_config.NumberColumn("PF/G", format="%.1f"),
            "avg_points_allowed": st.column_config.NumberColumn("PA/G", format="%.1f"),
            "avg_margin": st.column_config.NumberColumn("Margin/G", format="%.1f"),
            "close_games": st.column_config.NumberColumn("Close Games"),
        },
    )

with tabs[3]:
    st.markdown('<div class="section-title">Live Prediction Lab</div>', unsafe_allow_html=True)
    pred_col1, pred_col2, pred_col3, pred_col4, pred_col5 = st.columns(5)
    period = pred_col1.number_input("Period", min_value=1, max_value=8, value=4)
    seconds_remaining = pred_col2.number_input("Seconds Left", min_value=0, max_value=2880, value=300)
    home_score = pred_col3.number_input("Home Score", min_value=0, value=98)
    away_score = pred_col4.number_input("Away Score", min_value=0, value=95)
    scoring_play = pred_col5.checkbox("Scoring Play")

    features = build_features(period, seconds_remaining, home_score, away_score, scoring_play)
    probability = float(model.predict_proba(features)[0][1])
    lab_cols = st.columns([1, 2])
    with lab_cols[0]:
        render_metric("Predicted Home Win Probability", pct(probability), "XGBoost inference")
    with lab_cols[1]:
        st.dataframe(features, width="stretch", hide_index=True)

with tabs[4]:
    st.markdown('<div class="section-title">Model Metrics</div>', unsafe_allow_html=True)
    model_cols = st.columns(4)
    with model_cols[0]:
        render_metric("Train Rows", number(metrics["train_rows"]), f"{number(metrics['train_games'])} training games")
    with model_cols[1]:
        render_metric("Test Rows", number(metrics["test_rows"]), f"{number(metrics['test_games'])} test games")
    with model_cols[2]:
        render_metric("Log Loss", f"{metrics['log_loss']:.3f}", "lower is better")
    with model_cols[3]:
        render_metric("Brier Score", f"{metrics['brier_score']:.3f}", "probability calibration")

    st.markdown('<div class="section-title">Raw Training Snapshot</div>', unsafe_allow_html=True)
    st.json(metrics)
