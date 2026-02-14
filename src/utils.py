from src.dto import TriagePredictionRequest
import pandas as pd

def calculate_features(req: TriagePredictionRequest):
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