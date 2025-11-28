import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from src.config import config
from src.preprocessing import load_train_data, prepare_data, create_preprocessor


def train():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    print(f"🚀 MLflow Tracking URI: {config.MLFLOW_TRACKING_URI}")

    # 1. Prepare Data
    df = load_train_data()
    X, y = prepare_data(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Create Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessor()),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    # 3. Parameter Grid to be Searched
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
    }

    # 4. Start Grid Search
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

    with mlflow.start_run(run_name="XGBoost_GridSearch"):
        print("🕵️‍♂️ Hyperparameter scan starting... (This may take a while)")
        grid_search.fit(X_train, y_train)

        # Buy the best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"🏆 Best Parameters: {best_params}")

        # 5. Test with Validation Set
        y_pred = best_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        print(f"📊 Final Test Results -> Acc: {acc:.4f}, F1: {f1:.4f}")

        # 6. Log Results
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_params(best_params)

        print("💾 Saving the best model...")
        signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=config.MODEL_NAME,
            signature=signature
        )

        print("✅ Grid Search completed!")


if __name__ == "__main__":
    train()