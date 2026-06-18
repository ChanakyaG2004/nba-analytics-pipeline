# NBA AI/ML Analytics Dashboard

Phase 1 implements the local data foundation for an NBA analytics system:

- ESPN scoreboard discovery by date or date range
- ESPN play-by-play summary ingestion by event id
- Prefect-orchestrated ETL tasks and flow
- Idempotent PostgreSQL upserts for historical backfills
- Normalized warehouse tables for games, plays, and ingest run history
- Streamlit warehouse views for validating loaded data
- XGBoost win-probability training artifacts and evaluation metrics
- Batch model scoring back into PostgreSQL
- FastAPI model serving and local latency benchmarking
- Docker Compose services for database, API, and Streamlit dashboard
- Hugging Face Spaces-ready public demo package

## Tech Stack

- Python
- Prefect
- PostgreSQL
- SQLAlchemy
- ESPN API
- XGBoost
- scikit-learn
- Streamlit

## Run Locally

Start PostgreSQL:

```bash
cd nbaAnalytics
docker compose up -d db
```

Install dependencies:

```bash
pip install -r requirements.txt
```

On macOS, XGBoost may also require OpenMP:

```bash
brew install libomp
```

Ingest a single ESPN event:

```bash
python src/etl_pipeline.py --event-id 401705141
```

Backfill games from the ESPN scoreboard for a date range:

```bash
python src/etl_pipeline.py --start-date 2024-10-22 --end-date 2024-10-29
```

Open the game-level warehouse viewer:

```bash
streamlit run src/app.py
```

Open the ingest overview:

```bash
streamlit run src/dashboard.py
```

Backfill enough games for model training:

```bash
python src/etl_pipeline.py --start-date 2024-10-22 --end-date 2024-11-15
```

Train the first XGBoost win-probability model:

```bash
python src/train_model.py --min-games 20
```

Run a local prediction from the saved model:

```bash
python src/predict_model.py --period 4 --seconds-remaining 300 --home-score 98 --away-score 95
```

Score warehouse plays into PostgreSQL:

```bash
python src/score_predictions.py
```

Serve the trained model behind FastAPI:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Check API health:

```bash
curl -s http://127.0.0.1:8000/health
```

Call the prediction endpoint:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"period":4,"seconds_remaining":300,"home_score":98,"away_score":95}'
```

Benchmark local inference latency:

```bash
python src/benchmark_api.py --requests 100
```

Run the API and dashboard with Docker Compose:

```bash
docker compose up -d --build db api dashboard
```

Open:

- API: `http://127.0.0.1:8000/health`
- Dashboard: `http://127.0.0.1:8501`

Deploy the public demo to Hugging Face Spaces:

```bash
huggingface-cli login
cd nbaAnalytics
python3 deploy/deploy_hf_space.py
```

The Space source lives in `nbaAnalytics/hf_space`.

## Warehouse Tables

`games`

- One row per ESPN game/event
- Stores game date, matchup, status, and home/away team metadata

`play_by_play`

- One row per ESPN play
- Primary key: `(game_id, play_id)`
- Stores sequence, period, clock, score, score margin, play type, text, and raw ESPN JSON

`ingest_runs`

- One row per ETL execution
- Tracks source, requested events, successful events, total plays, status, and errors

`model_training_runs`

- One row per model training execution
- Tracks train/test size, Brier score, log loss, ROC AUC, calibration accuracy, and model path

`model_predictions`

- One row per scored play/model artifact
- Stores model-generated home win probability for dashboard playback

## Model Artifacts

Training writes local artifacts under `nbaAnalytics/`:

- `models/nba_xgb_win_probability.json`
- `artifacts/training_metrics.json`
- `artifacts/calibration_bins.csv`
- `artifacts/feature_importance.csv`

## Current Scope

The project now supports local historical ingestion, training, batch scoring, dashboard visualization, API serving, local latency benchmarking, Docker Compose deployment, and a Hugging Face Spaces public demo package. EC2 deployment steps are documented in `nbaAnalytics/docs/ec2_deployment.md`; an actual AWS deployment requires AWS credentials and a target account.
