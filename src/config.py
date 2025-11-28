import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    PROJ_ROOT = Path(__file__).resolve().parents[1]

    # Data Path
    DATA_RAW_PATH = PROJ_ROOT / "data" / "raw" / "WA_FN-UseC_-Telco-Customer-Churn.csv"
    DATA_PROCESSED_PATH = PROJ_ROOT / "data" / "processed" / "churn_data_processed.csv"

    # MLflow Settings
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Model Registry Name
    MODEL_NAME = "TelcoCustomerChurn"

    # Experiment Name
    EXPERIMENT_NAME = "churn-prediction-exp"

    # Database
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB = os.getenv("POSTGRES_DB")


config = Config()