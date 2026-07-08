"""Data cleaning and feature engineering pipeline for the Credit Card Fraud dataset.

Usage:
    python -m src.data_pipeline
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_DIR = Path("data/processed")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates and impute any missing values with the column median."""
    df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols = df.columns
    missing_before = df[numeric_cols].isna().sum().sum()
    if missing_before:
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical hour-of-day features and a log-transformed Amount."""
    seconds_in_day = 24 * 3600
    hour_fraction = (df["Time"] % seconds_in_day) / seconds_in_day

    df["Hour_sin"] = np.sin(2 * np.pi * hour_fraction)
    df["Hour_cos"] = np.cos(2 * np.pi * hour_fraction)
    df["Amount_log"] = np.log1p(df["Amount"])

    return df


def scale_amount(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Fit a RobustScaler on Amount/Amount_log using train only, apply to both splits."""
    cols_to_scale = ["Amount", "Amount_log"]
    scaler = RobustScaler()

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    return train_df, test_df, scaler


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["Class"],
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run(raw_path: Path = RAW_PATH, output_dir: Path = PROCESSED_DIR) -> None:
    print(f"[1/5] Loading raw data from {raw_path} ...")
    df = load_raw(raw_path)
    print(f"      -> {df.shape[0]} rows, {df.shape[1]} columns")

    print("[2/5] Cleaning (duplicates, missing values) ...")
    n_before = len(df)
    df = clean(df)
    print(f"      -> removed {n_before - len(df)} duplicate rows, {df.shape[0]} rows remain")

    print("[3/5] Engineering features (Hour_sin/cos, Amount_log) ...")
    df = engineer_features(df)

    print("[4/5] Splitting train/test (stratified on Class, test_size=0.2) ...")
    train_df, test_df = split(df)
    fraud_rate_train = train_df["Class"].mean() * 100
    fraud_rate_test = test_df["Class"].mean() * 100
    print(f"      -> train: {len(train_df)} rows ({fraud_rate_train:.4f}% fraud)")
    print(f"      -> test:  {len(test_df)} rows ({fraud_rate_test:.4f}% fraud)")

    print("[5/5] Scaling Amount/Amount_log with RobustScaler (fit on train only) ...")
    train_df, test_df, _ = scale_amount(train_df, test_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Saved processed data to {output_dir}/train.csv and {output_dir}/test.csv")


if __name__ == "__main__":
    run()
