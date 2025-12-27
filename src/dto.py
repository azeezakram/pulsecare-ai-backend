from pydantic import BaseModel, Field
from typing import Optional

class TriagePredictionRequest(BaseModel):
    sex: int = Field(..., ge=0, le=1)
    arrivalMode: int = Field(..., ge=1, le=7)
    injury: int = Field(..., ge=1, le=2)
    mental: int = Field(..., ge=1, le=4)
    pain: int = Field(..., ge=0, le=1)

    age: int = Field(..., ge=0, le=150)
    sbp: int = Field(..., ge=50, le=250)
    dbp: int = Field(..., ge=30, le=150)
    hr: int = Field(..., ge=30, le=250)
    rr: int = Field(..., ge=5, le=60)
    bt: float = Field(..., ge=35.0, le=42.0)

class TriagePredictionResponse(BaseModel):
    triageLevel: int
    confidence: float
    severity: str

    # Echo back features
    sex: int
    arrivalMode: int
    injury: int
    mental: int
    pain: int
    age: int
    sbp: int
    dbp: int
    hr: int
    rr: int
    bt: float

    shockIndex: float
    pulsePressure: float
    ppRatio: float
    hrBtInteraction: float
    rrHrRatio: float

    isFever: bool
    isTachy: bool
    isLowSbp: bool
    isLowDbp: bool
    isTachypnea: bool
