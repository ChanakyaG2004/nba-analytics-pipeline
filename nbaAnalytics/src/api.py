import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

try:
    from .features import FEATURE_COLUMNS, build_feature_frame
except ImportError:
    from features import FEATURE_COLUMNS, build_feature_frame


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/nba_xgb_win_probability.json"))


class PredictionRequest(BaseModel):
    period: int = Field(..., ge=1, le=8)
    seconds_remaining: int = Field(..., ge=0, le=2880)
    home_score: int = Field(..., ge=0)
    away_score: int = Field(..., ge=0)
    scoring_play: bool = False


class PredictionResponse(BaseModel):
    home_win_probability: float
    inference_ms: float
    model_path: str
    features: dict[str, float]


app = FastAPI(title="NBA Win Probability API", version="0.1.0")
MODEL = XGBClassifier()
MODEL_LOADED = False


@app.on_event("startup")
def load_model():
    global MODEL_LOADED

    if not MODEL_PATH.exists():
        MODEL_LOADED = False
        return

    MODEL.load_model(MODEL_PATH)
    MODEL_LOADED = True


def predict_probability(request):
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found or not loaded: {MODEL_PATH}",
        )

    features = build_feature_frame(
        period=request.period,
        seconds_remaining=request.seconds_remaining,
        home_score=request.home_score,
        away_score=request.away_score,
        scoring_play=request.scoring_play,
    )
    start = time.perf_counter()
    probability = float(MODEL.predict_proba(features)[0][1])
    inference_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        home_win_probability=probability,
        inference_ms=inference_ms,
        model_path=str(MODEL_PATH),
        features=features.iloc[0].to_dict(),
    )


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_LOADED else "model_not_loaded",
        "model_loaded": MODEL_LOADED,
        "model_path": str(MODEL_PATH),
        "features": FEATURE_COLUMNS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return predict_probability(request)


@app.get("/predict", response_model=PredictionResponse)
def predict_from_query(
    period: int = Query(..., ge=1, le=8),
    seconds_remaining: int = Query(..., ge=0, le=2880),
    home_score: int = Query(..., ge=0),
    away_score: int = Query(..., ge=0),
    scoring_play: bool = False,
):
    request = PredictionRequest(
        period=period,
        seconds_remaining=seconds_remaining,
        home_score=home_score,
        away_score=away_score,
        scoring_play=scoring_play,
    )
    return predict_probability(request)
