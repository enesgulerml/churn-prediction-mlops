import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.config import config
import redis
import json
import os
from prometheus_fastapi_instrumentator import Instrumentator # <--- NEW IMPORTS

ml_models = {}
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 1. REDIS CONNECTION ---
    global redis_client
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
        redis_client.ping()
        print(f"✅ Redis Connection Established on {redis_host}!")
    except Exception as e:
        print(f"⚠️ Redis Connection Failed: {e}. Caching will be disabled.")
        redis_client = None

    # --- 2. MODEL LOADING ---
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

    # Closing transactions
    ml_models.clear()
    print("🧹 The memory has been cleared.")


app = FastAPI(title="Telco Churn Prediction API", version="1.0", lifespan=lifespan)

# --- MONITORING INSTRUMENTATION (NEW) ---
# Expose metrics to Prometheus
Instrumentator().instrument(app).expose(app)
# ----------------------------------------


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
    redis_status = "active" if redis_client and redis_client.ping() else "inactive"
    return {
        "status": "active",
        "model_loaded": "model" in ml_models,
        "redis_cache": redis_status
    }


@app.post("/predict")
def predict(data: CustomerData):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="The model is out of service.")

    try:
        # --- CACHING LOGIC BEGINS ---

        # 1. Generate a unique key.
        cache_key = f"prediction:{data.model_dump_json()}"

        # 2. Check Redis (If Redis is running)
        if redis_client:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                print("⚡ CACHE HIT: The result is coming back from Redis!")
                return json.loads(cached_result)

        # 3. Cache Miss (Run Model)
        print("🐢 CACHE MISS: Running model...")
        input_df = pd.DataFrame([data.model_dump()])

        prediction = ml_models["model"].predict(input_df)
        churn_probability = ml_models["model"].predict_proba(input_df)

        result = int(prediction[0])
        prob_churn = float(churn_probability[0][1])

        response_data = {
            "prediction": result,
            "churn_status": "Yes" if result == 1 else "No",
            "churn_probability": round(prob_churn, 4),
            "source": "model"
        }

        # 4. Save Result to Redis (TTL: 1 Hour)
        if redis_client:
            cache_data = response_data.copy()
            cache_data["source"] = "cache"

            redis_client.setex(cache_key, 3600, json.dumps(cache_data))

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Env variable ile host settings
    uvicorn.run(app, host="0.0.0.0", port=8000)