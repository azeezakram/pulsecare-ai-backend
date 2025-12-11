from fastapi import FastAPI
import joblib
import numpy as np
from dto import TriagePredictionRequest, TriagePredictionResponse

app = FastAPI(title="Triage Prediction API")

# Load model + pipeline
model = joblib.load("model/model.pkl")
pipeline = joblib.load("model/pipeline.pkl")


def compute_derived_features(data: TriagePredictionRequest):
    sbp, dbp, hr, rr, bt = data.sbp, data.dbp, data.hr, data.rr, data.bt

    shock_index = data.shockIndex or round(hr / sbp, 3)
    pulse_pressure = data.pulsePressure or round(sbp - dbp, 3)
    pp_ratio = data.ppRatio or round(pulse_pressure / sbp, 3)
    hr_bt_interaction = data.hrBtInteraction or round(hr * bt, 3)
    rr_hr_ratio = data.rrHrRatio or round(rr / hr, 3)

    is_fever = data.isFever if data.isFever is not None else bt >= 38.0
    is_tachy = data.isTachy if data.isTachy is not None else hr > 100
    is_low_sbp = data.isLowSbp if data.isLowSbp is not None else sbp < 90
    is_low_dbp = data.isLowDbp if data.isLowDbp is not None else dbp < 60
    is_tachypnea = data.isTachypnea if data.isTachypnea is not None else rr > 20

    return {
        "shockIndex": shock_index,
        "pulsePressure": pulse_pressure,
        "ppRatio": pp_ratio,
        "hrBtInteraction": hr_bt_interaction,
        "rrHrRatio": rr_hr_ratio,
        "isFever": is_fever,
        "isTachy": is_tachy,
        "isLowSbp": is_low_sbp,
        "isLowDbp": is_low_dbp,
        "isTachypnea": is_tachypnea
    }


@app.post("/triage/predict", response_model=TriagePredictionResponse)
def predict(request: TriagePredictionRequest):
    derived = compute_derived_features(request)

    # Combine features exactly in model training order
    feature_vector = np.array([[
        request.sex,
        request.arrivalMode,
        request.injury,
        request.mental,
        request.pain,
        request.age,
        request.sbp,
        request.dbp,
        request.hr,
        request.rr,
        request.bt,

        derived["shockIndex"],
        derived["pulsePressure"],
        derived["ppRatio"],
        derived["hrBtInteraction"],
        derived["rrHrRatio"],

        int(derived["isFever"]),
        int(derived["isTachy"]),
        int(derived["isLowSbp"]),
        int(derived["isLowDbp"]),
        int(derived["isTachypnea"])
    ]])

    processed = pipeline.transform(feature_vector)
    prediction = model.predict(processed)[0]
    confidence = np.max(model.predict_proba(processed))

    severity = "CRITICAL" if prediction == 0 else "NON_CRITICAL"

    return TriagePredictionResponse(
        predictedTriageLevel=int(prediction),
        confidence=float(round(confidence, 4)),
        severity=severity,
        **request.dict(),
        **derived
    )
