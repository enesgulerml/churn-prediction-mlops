import mlflow
from mlflow.tracking import MlflowClient
from src.config import config


def register_best_model():
    """
    The function of this function is to:

    1. Find the model with the HIGHEST accuracy value among the experiments.
    2. Register that model in the Model Registry system.
    3. Update its label to 'Production' (Live).
    """

    # 1. Let's connect with MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 2. Let's find the ID of the experiment (We search by name)
    experiment = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        print("❌ Error: No experiments found. Train.py must be run first.")
        return

    experiment_id = experiment.experiment_id

    # 3. Let's find the best model (Sort by Accuracy metric, take the top one)
    print("🔍 Looking for the best model...")
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        print("❌ No 'run' found.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_acc = best_run.data.metrics.get("accuracy", 0)

    print(f"🏆 En İyi Run ID: {best_run_id}")
    print(f"📊 Accuracy: {best_acc:.4f}")

    # 4. Save Model to Registry (Create Version)
    model_uri = f"runs:/{best_run_id}/model"
    model_name = config.MODEL_NAME  # "TelcoCustomerChurn"

    print(f"💾 Saving the model: {model_name}...")
    model_version = mlflow.register_model(model_uri, model_name)

    # 5. Move the model to the 'Production' stage
    print(f"🚀 Version {model_version.version} -> Moving to 'Production' stage...")

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print("✅ PROCESS SUCCESSFUL! The model is now ready to go live.")


if __name__ == "__main__":
    register_best_model()