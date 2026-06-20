# NBA Analytics for 2025 - 2026 season

## Live website
  https://chanakyag-nba-ai-analytics-dashboard.hf.space

## Inspiration
As someone who enjoys sports betting, I wanted to make an application that could use probability and AI to see which teams have the higher win probability. (I have stopped sports betting but this was still fun to make)

## Very simple high level overview

```text
ESPN Data
    ↓
Python ETL Pipeline
    ↓
PostgreSQL ETL Pipeline
    ↓
Feature Engineering
    ↓
XGBoost Win Probability Model
    ↓
FastAPI Prediction Service
    ↓
Streamlit Dashboard
    ↓
Docker + Hugging Face Deployment
```

## Current Dataset
I decided to use the 2025-2026 NBA season export
The window is 10/01/2025 to 06/19/2026

Disclaimer: Not every play is scored because the model requires score/time fields. The plays missing include home_score, away_score, score_margin, or seconds_remaining are in the play log but not used in model scoring.

## Features
- Interactive analytics dashboard
- Team level game and performance analysis
- Win probability predictions using ML
- ETL pipeline for extracting, cleaning, and loading data
- 

## Core Stack
- Python for ETL (Extract, Transform, Lead), modeling, API, and dashboard Logic
- PostgreSQL for warehouse
- SQLAlchemy for database access
- Prefect for ETL orchestration
- pandas / NumPy for feature engineering
- XGBoost for probability modeling
- scikit learn for evaluation metrics
- FastAPI for prediction serving 
- Streamlit for dashboard
- Docker for local services
- Hugging Face for deployment


