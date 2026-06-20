# NBA Analytics for 2025-2026 season

## Live website
  https://chanakyag-nba-ai-analytics-dashboard.hf.space

## Inspiration
As someone who enjoys sports betting, I wanted to build an application that uses probability and AI to estimate which teams have a higher win probability. (I have stopped sports betting, but this was still a fun project to build)

## Very simple high level overview

```text
ESPN Data
    ↓
Python ETL Pipeline
    ↓
PostgreSQL Database
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
- PostgreSQL data warehouse for structured storage
- Feature engineering using pandas and NumPy
- XGBoost model for probability prediction
- Streamlit frontend to visualize dashboard
- Docker for local development setup
- Public deployment through Hugging Face

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

## Machine Learning Model

I used an XGBoost model to estimate win probability based on engineered basketball features. The model is trained on historical game and play level data, then evaluated using scikit learn metrics. This model outputs a probability estimate rather than just a win/loss prediction, making it more useful for understanding confidence and game context.

## API

I used FastAPI to serve model predictions. The API acts as the connection between the trained model and the dashboard

## Dashboard

The Streamlit dashboard provides a simple interface for exploring NBA data and viewing model predictions. It is designed to make the machine learning results understandable through charts, tables, and probability outputs.

## Running Locally

git clone https://github.com/ChanakyaG2004/nba-analytics-pipeline.git
cd https://github.com/ChanakyaG2004/nba-analytics-pipeline.git

Create and activate virtual environment:

python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Start local services with Docker Compose:

docker compose up

Run Streamlit and then the FastAPI server

streamlit run app.py
uvicorn main:app --reload

## Project Structure
```text
nba-analytics-pipeline/
  ├── README.md
  └── nbaAnalytics/
      ├── Dockerfile
      ├── docker-compose.yml
      ├── requirements.txt
      ├── src/
      │   ├── api.py
      │   ├── app.py
      │   ├── benchmark_api.py
      │   ├── dashboard.py
      │   ├── database.py
      │   ├── etl_pipeline.py
      │   ├── features.py
      │   ├── predict_model.py
      │   ├── score_predictions.py
      │   └── train_model.py
      ├── models/
      │   └── nba_xgb_win_probability.json
      ├── artifacts/
      │   ├── calibration_bins.csv
      │   ├── feature_importance.csv
      │   └── training_metrics.json
      ├── hf_space/
      │   ├── Dockerfile
      │   ├── README.md
      │   ├── app.py
      │   ├── requirements.txt
      │   ├── data/
      │   │   ├── games.csv
      │   │   └── plays_with_predictions.csv
      │   ├── models/
      │   │   └── nba_xgb_win_probability.json
      │   └── artifacts/
      │       ├── calibration_bins.csv
      │       ├── feature_importance.csv
      │       └── training_metrics.json
      ├── deploy/
      │   ├── deploy_ec2.sh
      │   ├── deploy_hf_space.py
      │   └── ec2_user_data.sh
      └── docs/
          ├── ec2_deployment.md
          └── huggingface_spaces.md

```

## Limitations

- Model depends on public data only from the 2025-2026 season
- Some factors including the player's personal life, mental state, trades, rest days, etc are not included in the probability factors of this dataset. 

## Future improvements

As said in the limitations, if I wanted to drastically strengthen the results of the data, I would include an X scraping API to include what the players are going through in their life outside of the game. I would also improve standard feature engineering and add more model comparison metrics. I feel like I could also add more visualizations for team and player trends.

## What I learned

There were many things I learned from this project. This project is a complete overview of building a data application from end to end. I worked on data extraction, cleaning, database design, machine learning, API development, dashboarding, containerization, and deployment. It also helped me understand how different parts of a production sytle analytics system connect together.

## Quick fun little disclaimer

Although I said in my inspiration section that I built this for potential use in sports betting, this is for analytical purposes only. It is not intended for betting advice or anything financial related. (If you use this as such, enjoy losing your money and let me know when you do so I can laugh at you)

Thanks for reading!!!