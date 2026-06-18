import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:password123@localhost:5432/nba_db",
)

st.set_page_config(page_title="NBA Warehouse Overview", layout="wide")
engine = create_engine(DATABASE_URL)

st.title("NBA Warehouse Overview")

try:
    metrics = pd.read_sql(
        """
        SELECT
            COUNT(DISTINCT g.game_id) AS games,
            COUNT(p.play_id) AS plays,
            COUNT(mp.play_id) AS predictions,
            MAX(p.updated_at) AS latest_play_update
        FROM games g
        LEFT JOIN play_by_play p ON p.game_id = g.game_id
        LEFT JOIN model_predictions mp
            ON mp.game_id = p.game_id
           AND mp.play_id = p.play_id
        """,
        engine,
    ).iloc[0]

    run_history = pd.read_sql(
        """
        SELECT
            started_at,
            finished_at,
            source,
            requested_events,
            successful_events,
            total_plays,
            status,
            error
        FROM ingest_runs
        ORDER BY started_at DESC
        LIMIT 20
        """,
        engine,
    )

    games_by_date = pd.read_sql(
        """
        SELECT
            DATE(game_date) AS game_day,
            COUNT(*) AS games
        FROM games
        GROUP BY DATE(game_date)
        ORDER BY game_day
        """,
        engine,
    )

    training_runs = pd.read_sql(
        """
        SELECT
            created_at,
            train_games,
            test_games,
            train_rows,
            test_rows,
            brier_score,
            log_loss,
            roc_auc,
            calibration_accuracy,
            model_path
        FROM model_training_runs
        ORDER BY created_at DESC
        LIMIT 10
        """,
        engine,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Games", int(metrics["games"] or 0))
    col2.metric("Plays", int(metrics["plays"] or 0))
    col3.metric("Predictions", int(metrics["predictions"] or 0))
    col4.metric("Latest Update", str(metrics["latest_play_update"] or "none"))

    st.subheader("Games by Date")
    if games_by_date.empty:
        st.info("No ingested games yet.")
    else:
        st.bar_chart(games_by_date.set_index("game_day"))

    st.subheader("Ingest Runs")
    st.dataframe(run_history, width="stretch", hide_index=True)

    st.subheader("Model Training Runs")
    st.dataframe(training_runs, width="stretch", hide_index=True)

except Exception as exc:
    st.error(f"Could not load warehouse overview. Error: {exc}")
    st.info("Start Postgres, then run the ETL script or Prefect flow.")
