import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

st.set_page_config(page_title="NBA Live", layout="wide")
engine = create_engine("postgresql://admin:password123@localhost:5432/nba_db")

st.title("🏀 NBA Data Warehouse")

try:
    # Query using the exact lowercase names
    df = pd.read_sql("SELECT * FROM play_by_play", engine)
    
    if df.empty:
        st.warning("Database table exists but is empty. Run ETL again!")
    else:
        st.success(f"Connected! Showing {len(df)} plays.")
        st.dataframe(df, use_container_width=True)
        
        # Simple Chart
        st.subheader("Plays per Period")
        period_counts = df['period_num'].value_counts().sort_index()
        st.bar_chart(period_counts)

except Exception as e:
    st.error(f"Could not find table. Error: {e}")
    st.info("Tip: Make sure you ran the ETL script and it printed 'VERIFIED'.")