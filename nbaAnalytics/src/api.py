from fastapi import FastAPI
import xgboost as xgb
import numpy as np

app = FastAPI()

# Load model ONCE into memory for speed
MODEL = xgb.XGBClassifier()
# Assume model is saved as 'nba_model.json'
# MODEL.load_model("nba_model.json") 

@app.get("/predict")
async def get_win_probability(margin: int, sec_left: int):
    features = np.array([[sec_left, margin]])
    # Inference is extremely fast (<10ms)
    prob = float(MODEL.predict_proba(features)[0][1])
    return {"home_win_probability": prob}