import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.config import config

# --- Columns Name ---
CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]

NUMERICAL_FEATURES = [
    "tenure_months",
    "monthlycharges",
    "totalcharges"
]


def load_train_data():
    """
    Reads the processed training data from disk.
    """
    path = config.PROJ_ROOT / "data" / "processed" / "churn_train.csv"
    df = pd.read_csv(path)
    return df


def create_preprocessor():
    """
    Creates the Pipeline object that will process numerical and categorical data.
    Return: ColumnTransformer
    """

    # Step 1: Converter for Categorical Variables (One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Step 2: Scaling for Numeric Variables
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Step 3: Combine It All
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def prepare_data(df):
    """
    It separates the data into X (Features) and y (Target).
    Target (Churn) 'Yes'/'No' -> 1/0 conversion is performed.
    """
    X = df.drop(["churn", "customerid"], axis=1)

    # Change target variable to 1 and 0
    y = df["churn"].map({"Yes": 1, "No": 0})

    return X, y


if __name__ == "__main__":
    print("🔄 Preprocessing test begins...")

    df = load_train_data()
    X, y = prepare_data(df)

    preprocessor = create_preprocessor()

    print("🛠️ Pipeline training (fitting)...")
    X_processed = preprocessor.fit_transform(X)

    print(f"✅ Success! Size of processed data:{X_processed.shape}")
    print(f"   Original number of lines: {X.shape[0]}")
    print(f"   Number of new features:{X_processed.shape[1]}")

    try:
        feature_names = preprocessor.get_feature_names_out()
        print(f"  Example features: {feature_names[:5]}")
    except:
        pass