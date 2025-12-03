# 🔮 End-to-End Telco Churn Prediction System (MLOps)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Docker](https://img.shields.io/badge/Docker-Microservices-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-yellow)

## 📌 Project Overview
This project is a production-grade **MLOps solution** designed to predict customer churn in the telecommunications industry. Unlike traditional data science workflows, this system implements a fully containerized **End-to-End Pipeline** that includes data ingestion, automated experiment tracking, model registry, and real-time serving via microservices.

The architecture is designed to be **reproducible, scalable, and deployable** with a single command.

---

## 🏗️ Architecture

The system runs as a set of orchestrated Docker containers:

* **Database:** PostgreSQL (Backend storage for MLflow).
* **Experiment Tracking:** MLflow Server (Tracks parameters, metrics, and artifacts).
* **Model Registry:** Centralized repository for managing model versions (Staging/Production).
* **Inference API:** FastAPI (Asynchronous REST API handling predictions).
* **Frontend:** Streamlit (User interface for business stakeholders).

### Key Features
* **Champion/Challenger Strategy:** Baseline Random Forest vs. Optimized XGBoost (GridSearchCV).
* **Zero-Dependency Deployment:** Works on any machine with Docker installed.
* **Dynamic Model Loading:** API automatically fetches the latest "Production" model from the registry.
* **Persistent Storage:** Database and model artifacts are preserved via Docker Volumes.

---
## 📂 Dataset Setup

The dataset used in this project is the **Telco Customer Churn** dataset.

1.  **Download:** Get the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file from [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn).
2.  **Place:** Move the downloaded CSV file into the `data/raw/` directory.

**Expected Directory Structure:**
```text
churn-prediction-mlops/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## 🛠️ Installation & Setup

You don't need to install Python or libraries manually. **Docker** handles everything.

### 1. Clone the Repository
```bash
git clone https://github.com/enesgulerml/churn-prediction-mlops.git
cd churn-prediction-mlops
```

### 2. Environment Configuration
Rename the example environment file to activate configuration:
```bash
# Linux/Mac
cp .env.example .env

# Windows (CMD)
copy .env.example .env
```

```bash
docker-compose up -d --build
```

## 🚀 Usage Pipeline
Once the containers are running, you need to execute the training pipeline inside the container to ensure path consistency.

### Step 1: Data Ingestion
Loads raw data and prepares it for training.

```bash
docker exec mlops_api python -m src.data_loader
```

### Step 2: Model Training (XGBoost)
Trains the model using GridSearchCV and logs metrics to MLflow.

```bash
docker exec mlops_api python -m src.train
```

### Step 3: Register Model
Promotes the best model to the "Production" stage in MLflow Registry.

```bash
docker exec mlops_api python -m src.register_model
```

### Step 4: Refresh API
Restart the API service to load the newly registered production model.

```bash
docker restart mlops_api
```

## 🧪 Testing

To ensure system reliability, the project includes a comprehensive test suite using **pytest**.

### Running Tests Locally
You can execute unit and integration tests directly from your local environment. The API tests use **mocking** to simulate model inference, removing the need for active Docker containers during testing.

```bash
# Install test dependencies (if not already installed)
pip install pytest httpx

# Run the test suite with verbose output
pytest -v
```

**Test Scope**

* Unit Tests (tests/test_preprocessing.py): Validates data cleaning logic, ensuring critical features like customerid are removed and target variables are correctly mapped.

* Integration Tests (tests/test_api.py): Verifies the stability of API endpoints (/health and /predict), checking response status codes and JSON schema validity.

## 📊 Accessing the Interfaces

| Service | URL | Description |
| :--- | :--- | :--- |
| **Streamlit Dashboard** | `http://localhost:8501` | Interactive UI for Churn Prediction |
| **FastAPI Swagger** | `http://localhost:8000/docs` | API Documentation & Testing |
| **MLflow UI** | `http://localhost:5000` | Experiment Tracking & Model Registry |

---

## 🛑 Stopping the System

To stop the services while preserving the database and model registry data (Standard Shutdown):
```bash
docker-compose down
```

To stop the services and remove all persistent data volumes (Reset everything to a clean state, useful for debugging or fresh start):

```bash
docker-compose down -v
```

## 📈 Performance Results

| Model | Accuracy   | F1 Score | Status |
| :--- |:-----------| :--- | :--- |
| Random Forest (Baseline) | 78.8%      | 0.54 | Archived |
| **XGBoost (Tuned)** | **80.92%** | **0.58** | **Production** |
