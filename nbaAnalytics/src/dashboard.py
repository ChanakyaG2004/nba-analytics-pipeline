import streamlit as st
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="NBA AI Analytics", layout="wide")

st.title("🏀 NBA Real-Time Analytics Dashboard")

# Sidebar - Controls
st.sidebar.header("Game Selection")
game_id = st.sidebar.text_input("Enter Game ID", "0022300001")

# Main Content - Two Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Win Probability")
    # Simulate fetching data from your API/DB
    chart_data = pd.DataFrame({
        'Time': range(100),
        'Prob': [0.5 + (i * 0.004) for i in range(100)]
    })
    fig = px.line(chart_data, x='Time', y='Prob', range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AI Performance Insights")
    st.metric(label="Current Win Prob", value="74%", delta="+2.3%")
    st.write("**XGBoost Forecast:** Home team expected to win by 4.5 pts.")
    
    # Table of recent events
    st.dataframe(pd.DataFrame({
        'Event': ['3pt Shot', 'Foul', 'Substitution'],
        'Impact': ['+4.2%', '-1.1%', '+0.5%']
    }))