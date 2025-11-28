import pandas as pd
import mlflow.sklearn  # <-- DEĞİŞİKLİK 1: pyfunc yerine sklearn import ettik
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.config import config

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    model_name = config.MODEL_NAME
    stage = "Production"

    print(f"📡 Loading model: {model_name} (Stage: {stage})...")

    try:
        model_uri = f"models:/{model_name}/{stage}"


        ml_models["model"] = mlflow.sklearn.load_model(model_uri)

        print("✅ The model has been successfully loaded and the API is ready!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")

    yield
    ml_models.clear()
    print("🧹 The memory has been cleared.")



app = FastAPI(title="Telco Churn Prediction API", version="1.0", lifespan=lifespan)


class CustomerData(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    tenure_months: int
    monthlycharges: float
    totalcharges: float


@app.get("/")
def home():
    return {"message": "Telco Churn Prediction API is Running! 🚀"}


@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": "model" in ml_models}


@app.post("/predict")
def predict(data: CustomerData):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="The model is out of service.")

    try:
        input_df = pd.DataFrame([data.model_dump()])

        prediction = ml_models["model"].predict(input_df)
        churn_probability = ml_models["model"].predict_proba(input_df)

        result = int(prediction[0])
        prob_churn = float(churn_probability[0][1])

        return {
            "prediction": result,
            "churn_status": "Yes" if result == 1 else "No",
            "churn_probability": round(prob_churn, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)