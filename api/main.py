from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "modeling"
    / "best_model_random_forest.joblib"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# API
app = FastAPI(
    title="Arabic Education Indicators API",
    description=(
        "API for estimating education indicator values "
        "using a saved Random Forest pipeline."
    ),
    version="1.0.0",
)


# Request Schema
class PredictionRequest(BaseModel):
    Indicator: str = Field(min_length=1)
    Country: str = Field(min_length=1)
    Nationality: str = Field(min_length=1)
    Sex: str = Field(min_length=1)
    Educational_level: str = Field(min_length=1)
    Sector: str = Field(min_length=1)

    Year: int = Field(ge=1900, le=2100)


# Endpoints
@app.get("/")
def home():
    return {
        "message": "Arabic Education Indicators API",
        "status": "running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert request to a DataFrame
        input_data = request.model_dump()

        year = input_data.pop("Year")

        # Recreate the same numerical features
        # used during model training
        years_since_2000 = year - 2000

        input_data["Years_Since_2000"] = years_since_2000
        input_data["Year_Squared"] = years_since_2000 ** 2

        # Preserve the feature names expected by the pipeline
        feature_columns = [
            "Indicator",
            "Country",
            "Nationality",
            "Sex",
            "Educational_level",
            "Sector",
            "Years_Since_2000",
            "Year_Squared",
        ]

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns,
        )

        # Generate prediction
        prediction = float(model.predict(input_df)[0])

        return {
            "success": True,
            "prediction": {
                "Country": request.Country,
                "Indicator": request.Indicator,
                "Year": year,
                "predicted_value": prediction,
            },
            "model": "Random Forest",
            "note": (
                "This is a model estimate, not a guarantee "
                "of the actual or future value."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc