import os

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:password123@localhost:5432/nba_db",
)

st.set_page_config(page_title="NBA Live", layout="wide")
engine = create_engine(DATABASE_URL)

st.title("NBA Data Warehouse")

try:
    games = pd.read_sql(
        """
        SELECT
            game_id,
            game_date,
            short_name,
            status,
            home_team_abbrev,
            away_team_abbrev
        FROM games
        ORDER BY game_date DESC NULLS LAST
        """,
        engine,
    )

    if games.empty:
        st.warning("No games found. Run the ETL flow first.")
        st.stop()

    selected_game_id = st.sidebar.selectbox(
        "Game",
        options=games["game_id"].tolist(),
        format_func=lambda game_id: (
            games.loc[games["game_id"] == game_id, "short_name"].iloc[0] or game_id
        ),
    )

    selected_game = games[games["game_id"] == selected_game_id].iloc[0]
    st.subheader(selected_game["short_name"] or selected_game_id)
    st.caption(f"Status: {selected_game['status'] or 'unknown'}")

    plays = pd.read_sql(
        """
        WITH latest_model AS (
            SELECT game_id, model_path
            FROM (
                SELECT
                    game_id,
                    model_path,
                    ROW_NUMBER() OVER (
                        PARTITION BY game_id
                        ORDER BY MAX(scored_at) DESC
                    ) AS row_num
                FROM model_predictions
                GROUP BY game_id, model_path
            ) ranked
            WHERE row_num = 1
        )
        SELECT
            p.sequence_number,
            p.period,
            p.clock_display,
            p.home_score,
            p.away_score,
            p.score_margin,
            mp.home_win_probability,
            p.play_type_text,
            p.text
        FROM play_by_play p
        LEFT JOIN latest_model lm
            ON lm.game_id = p.game_id
        LEFT JOIN model_predictions mp
            ON mp.game_id = p.game_id
           AND mp.play_id = p.play_id
           AND mp.model_path = lm.model_path
        WHERE p.game_id = %(game_id)s
        ORDER BY p.sequence_number
        """,
        engine,
        params={"game_id": selected_game_id},
    )

    st.success(f"Loaded {len(plays)} plays for this game.")

    latest = plays.dropna(subset=["home_win_probability"]).tail(1)
    if latest.empty:
        st.warning("No model predictions found for this game. Run `python src/score_predictions.py`.")
    else:
        st.metric(
            "Latest Home Win Probability",
            f"{latest['home_win_probability'].iloc[0] * 100:.1f}%",
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Win Probability")
        probability = plays[["sequence_number", "home_win_probability"]].dropna()
        if probability.empty:
            st.info("Score this game to show model probabilities.")
        else:
            st.line_chart(probability.set_index("sequence_number"))

    with col2:
        st.subheader("Score Margin")
        margin = plays[["sequence_number", "score_margin"]].dropna()
        if margin.empty:
            st.info("No score margin values found for this game.")
        else:
            st.line_chart(margin.set_index("sequence_number"))

    st.subheader("Play Log")
    st.dataframe(plays, width="stretch", hide_index=True)

except Exception as exc:
    st.error(f"Could not load warehouse data. Error: {exc}")
    st.info("Start Postgres, then run the ETL script or Prefect flow.")
