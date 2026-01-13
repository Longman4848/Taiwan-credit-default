import os
import logging
import pandas as pd
import json
import numpy as np
import mlflow
import mlflow.xgboost  # Switched to xgboost flavor
from contextlib import asynccontextmanager
from mlflow.tracking import MlflowClient
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator, ValidationInfo

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model Lifecycle Management
ml_models: Dict[str, Any] = {}


ml_models = {}

MLFLOW_TRACKING_URI = "sqlite:///C:/Users/EmmyTech/Downloads/Credit-Card-Default-Taiwan-MLProject/data/mlruns/mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "Taiwan_XGBoost_Model"
MODEL_STAGE = "Production"



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client = MlflowClient()

        # ---- Load Production Model ----
        versions = client.get_latest_versions(
            MODEL_NAME, stages=[MODEL_STAGE]
        )
        if not versions:
            raise RuntimeError(
                f"No {MODEL_STAGE} version found for {MODEL_NAME}"
            )

        model_version = versions[0]
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

        logger.info(f"Loading model from {model_uri}")
        ml_models["model"] = mlflow.xgboost.load_model(model_uri)

        # ---- Load Threshold Artifact (SAFE) ----
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=model_version.run_id,
                artifact_path="threshold_config.json"
            )

            with open(local_path) as f:
                config = json.load(f)

            ml_models["threshold"] = float(config["threshold"])
            logger.info(
                f"Threshold loaded: {ml_models['threshold']}"
            )

        except Exception:
            logger.warning(
                "Threshold artifact missing. Defaulting to 0.5"
            )
            ml_models["threshold"] = 0.5

        logger.info(
            f"API READY | Model v{model_version.version} | "
            f"Threshold={ml_models['threshold']}"
        )

    except Exception as e:
        logger.exception("Startup failed")
        raise

    yield

 
    ml_models.clear()
    logger.info("Resources released")

app = FastAPI(
    title='Taiwan Credit Card Default Prediction API',
    version='1.1.0',
    lifespan=lifespan
)


# 3. Input Schema (Kept your validation logic)
class CreditRiskFeatures(BaseModel):
    X1: float = Field(..., gt=0)
    X2: int; X3: int; X4: int; X5: int
    X6: int; X7: int; X8: int; X9: int; X10: int; X11: int
    X12: float; X13: float; X14: float; X15: float; X16: float; X17: float
    X18: float; X19: float; X20: float; X21: float; X22: float; X23: float

# 4. Endpoints
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "model" in ml_models}

@app.post("/predict")
async def predict_default(data: CreditRiskFeatures):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        model = ml_models["model"]
        threshold = ml_models.get("threshold", 0.5) # Use the loaded 0.45!
        
        # Get probability from XGBoost
        probabilities = model.predict_proba(input_df)
        prob_default = float(probabilities[0][1])
        
        # Logic: If prob is >= 0.45, it's a Default (1)
        custom_prediction = 1 if prob_default >= threshold else 0

        return {
            "prediction": custom_prediction,
            "probability_default": round(prob_default, 4),
            "applied_threshold": threshold,
            "label": "Default" if custom_prediction == 1 else "No Default"
        }
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Inference failed")
# To run the app, use the command:
# uvicorn backend.main:app --reload
#python -m uvicorn backend.main:app --reload


#model = joblib.load("RF_TaiwanCredit_Default_model_v1.pkl")
