import os

import joblib
import pandas as pd
from fastapi import FastAPI, Security

from src.config.auth import verify_api_key
from src.dto import TriagePredictionRequest, TriagePredictionResponse
from src.utils import calculate_features

app = FastAPI(title="Triage Prediction API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "src", "model", "triage_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "src", "model", "scaler.pkl"))
final_features = joblib.load(os.path.join(BASE_DIR, "src", "model", "final_features.pkl"))  # snake_case

@app.get("/")
def read_root(api_key: str = Security(verify_api_key)):
    return {"message": "Welcome to the Triage Prediction API"}

@app.post("/predict", response_model=TriagePredictionResponse)
def predict(request: TriagePredictionRequest, api_key: str = Security(verify_api_key)):
    features = calculate_features(request)

    df_features = pd.DataFrame([features])
    x_ordered = df_features[final_features]
    x_scaled = scaler.transform(x_ordered)
    proba = model.predict_proba(x_scaled)[0]

    threshold = 0.7
    pred_class = 0 if proba[0] > threshold else 1

    # if features['sbp'] <= 90 or features['hr'] >= 120 or features['rr'] >= 22 or features['bt'] >= 38:
    #     pred_class = 0

    severity = "Critical" if pred_class == 0 else "Non-Critical"
    confidence = proba[pred_class]

    camel_features = {
        "shockIndex": features["shock_index"],
        "pulsePressure": features["pulse_pressure"],
        "ppRatio": features["pp_ratio"],
        "hrBtInteraction": features["hr_bt_interaction"],
        "rrHrRatio": features["rr_hr_ratio"],
        "isFever": bool(features["is_fever"]),
        "isTachy": bool(features["is_tachy"]),
        "isLowSbp": bool(features["is_low_sbp"]),
        "isLowDbp": bool(features["is_low_dbp"]),
        "isTachypnea": bool(features["is_tachypnea"]),
    }

    response = TriagePredictionResponse(
        triageLevel=pred_class,
        confidence=confidence,
        severity=severity,
        **request.model_dump(),
        **camel_features
    )

    print(response.model_dump())

    return response

