from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_PATH = PROJECT_ROOT / "src"

sys.path.append(str(SRC_PATH))


# ============================================================
# IMPORT PREDICTION FUNCTION
# ============================================================

from predict import predict_risk


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for Home Credit payment difficulty prediction",
    version="1.0.0"
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class ApplicantRequest(BaseModel):
    applicant_data: dict


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Credit Risk Prediction API is running",
        "status": "success"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: ApplicantRequest):

    try:

        # Convert input dictionary into DataFrame
        applicant_df = pd.DataFrame(
            [request.applicant_data]
        )

        # Run prediction pipeline
        result = predict_risk(
            applicant_df
        )

        return {
            "status": "success",
            "prediction": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )