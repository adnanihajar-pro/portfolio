"""Nettoyage, feature engineering et split pour le dataset Give Me Some Credit."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RAW_PATH = "data/raw/cs-training.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TARGET = "SeriousDlqin2yrs"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    # MonthlyIncome et NumberOfDependents ont des valeurs manquantes dans ce dataset
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )

    # Valeurs sentinelles connues du dataset (96/98) pour les compteurs de retard
    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    for col in late_cols:
        df[col] = df[col].clip(upper=df[col].quantile(0.999))

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    df = df.copy()
    df["DebtRatio_log"] = np.log1p(df["DebtRatio"].clip(lower=0))
    df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"].clip(lower=0))
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    return df


def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_cols = ["MonthlyIncome_log", "DebtRatio_log"]
    scaler = RobustScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    return train_df, test_df


def main():
    df = load_raw()
    df = clean(df)
    df = add_features(df)
    train_df, test_df = split_and_scale(df)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")


if __name__ == "__main__":
    main()
