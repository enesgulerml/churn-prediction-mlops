import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import config
import os


def load_dataset(path):
    """
    Reads CSV file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    print(f"✅ Data loaded. Size: {df.shape}")
    return df


def clean_column_names(df):
    """
    Standardizes column names (Lower Case & Snake Case).
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # Some special fixes
    df.rename(columns={'seniorcitizen': 'senior_citizen', 'tenure': 'tenure_months'}, inplace=True)
    return df


def handle_imbalanced_data(df):

    return df


def split_and_save(df):
    """
    It separates and saves the data into Training (80%) and Simulation (20%).
    """

    train_df, unseen_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['churn'])

    # Create folder if it doesn't exist
    output_dir = config.PROJ_ROOT / "data" / "processed"
    os.makedirs(output_dir, exist_ok=True)

    # Save files
    train_path = output_dir / "churn_train.csv"
    unseen_path = output_dir / "churn_unseen.csv"

    train_df.to_csv(train_path, index=False)
    unseen_df.to_csv(unseen_path, index=False)

    print(f"✅ Data was parsed and recorded:")
    print(f"   📂 Train Data: {train_df.shape} -> {train_path}")
    print(f"   📂 Unseen Data: {unseen_df.shape} -> {unseen_path} (Save that for live simulation!)")


if __name__ == "__main__":
    df = load_dataset(config.DATA_RAW_PATH)
    df = clean_column_names(df)

    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df['totalcharges'] = df['totalcharges'].fillna(0)

    split_and_save(df)