from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from src.app import app, ml_models

client = TestClient(app)


def test_health_check():
    """
    Test the /health endpoint.
    This doesn't need a model, so it should pass easily.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "active"


def test_prediction_endpoint():
    """
    Test the /predict endpoint with MOCK data.
    Instead of loading the real model from MLflow (which fails locally due to path issues),
    we inject a 'Fake Model' into the application memory.
    """

    # 1. CREATE A FAKE MODEL (MOCK)
    fake_model = MagicMock()

    fake_model.predict.return_value = [1]

    fake_model.predict_proba.return_value = [[0.15, 0.85]]

    # 2. INJECT FAKE MODEL INTO APPLICATION
    ml_models["model"] = fake_model

    # 3. Test Data
    payload = {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "phoneservice": "No",
        "multiplelines": "No phone service",
        "internetservice": "DSL",
        "onlinesecurity": "No",
        "onlinebackup": "Yes",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": 29.85,
        "totalcharges": 29.85
    }

    # 4. Post Request
    try:
        response = client.post("/predict", json=payload)

        # 5. Kontroller
        assert response.status_code == 200, f"Error Detail: {response.text}"

        data = response.json()
        assert data["prediction"] == 1
        assert "churn_probability" in data
        assert data["churn_status"] == "Yes"

    finally:
        ml_models.clear()