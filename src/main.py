from fastapi import FastAPI
import joblib
import pandas as pd
from src.dto import TriagePredictionRequest, TriagePredictionResponse
import os

app = FastAPI(title="Triage Prediction API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained objects
model = joblib.load(os.path.join(BASE_DIR, "src", "model", "triage_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "src", "model", "scaler.pkl"))
final_features = joblib.load(os.path.join(BASE_DIR, "src", "model", "final_features.pkl"))  # snake_case


def calculate_features(req: TriagePredictionRequest):
    """Compute derived features in snake_case to match training."""
    df = pd.DataFrame([{
        "sex": req.sex,
        "arrival_mode": req.arrivalMode,
        "injury": req.injury,
        "mental": req.mental,
        "pain": req.pain,
        "age": req.age,
        "sbp": req.sbp,
        "dbp": req.dbp,
        "hr": req.hr,
        "rr": req.rr,
        "bt": req.bt
    }])
    
    df['shock_index'] = df['hr'] / df['sbp']
    df['pulse_pressure'] = df['sbp'] - df['dbp']
    df['pp_ratio'] = df['pulse_pressure'] / df['sbp']
    df['hr_bt_interaction'] = df['hr'] * df['bt']
    df['rr_hr_ratio'] = df['rr'] / (df['hr'] + 1)
    df['is_fever'] = (df['bt'] >= 38).astype(int)
    df['is_tachy'] = (df['hr'] >= 120).astype(int)
    df['is_low_sbp'] = (df['sbp'] <= 90).astype(int)
    df['is_low_dbp'] = (df['dbp'] <= 60).astype(int)
    df['is_tachypnea'] = (df['rr'] >= 22).astype(int)
    
    return df.iloc[0].to_dict()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Triage Prediction API"}

@app.post("/predict", response_model=TriagePredictionResponse)
def predict(request: TriagePredictionRequest):
    features = calculate_features(request)

    df_features = pd.DataFrame([features])
    X_ordered = df_features[final_features]
    X_scaled = scaler.transform(X_ordered)
    proba = model.predict_proba(X_scaled)[0]

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

    return TriagePredictionResponse(
        predictedTriageLevel=pred_class,
        confidence=confidence,
        severity=severity,
        **request.model_dump(),
        **camel_features
    )

