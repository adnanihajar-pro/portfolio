"""Nettoyage, feature engineering et split pour le dataset Give Me Some Credit."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RAW_PATH = "data/raw/cs-training.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TARGET = "SeriousDlqin2yrs"

LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def clean_data(df: pd.DataFrame, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Nettoie un DataFrame (train ou test).

    Si `stats` est None, ce df est traité comme le train : les médianes sont
    calculées dessus et retournées dans `stats`. Si `stats` est fourni (cas du
    test), les médianes du train sont réutilisées telles quelles, sans être
    recalculées, pour éviter toute fuite de données train -> test.
    """
    df = df.copy()
    is_train = stats is None
    if is_train:
        stats = {}

    # --- 2. Valeurs manquantes : indicateur AVANT imputation, médiane du train ---
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)
    if is_train:
        stats["monthly_income_median"] = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(stats["monthly_income_median"])

    df["NumberOfDependents_missing"] = df["NumberOfDependents"].isna().astype(int)
    if is_train:
        stats["dependents_median"] = df["NumberOfDependents"].median()
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(stats["dependents_median"])

    # --- 4. Flags d'incohérence, calculés AVANT le plafonnement (3a / 3d) ---
    df["is_inconsistent_utilization"] = (
        (df["NumberOfOpenCreditLinesAndLoans"] == 0)
        & (df["RevolvingUtilizationOfUnsecuredLines"] > 0)
    ).astype(int)
    df["is_inconsistent_debtratio"] = (
        (df["MonthlyIncome"] == 0) & (df["DebtRatio"] > 1)
    ).astype(int)

    # --- 3a. RevolvingUtilizationOfUnsecuredLines : borné à 1 par définition ---
    df["utilisation_aberrante"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 1).astype(int)
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=1)

    # --- 3b. age == 0 : valeur impossible, remplacée par la médiane du train ---
    if is_train:
        stats["age_median"] = df["age"].median()
    df["age"] = df["age"].replace(0, stats["age_median"])

    # --- 3c. Compteurs de retard : 96/98 sont des codes système, pas de vrais comptages ---
    df["retard_code_suspect"] = df[LATE_COLS].isin([96, 98]).any(axis=1).astype(int)
    if is_train:
        non_suspect = df.loc[df["retard_code_suspect"] == 0]
        stats["late_medians"] = {col: non_suspect[col].median() for col in LATE_COLS}
    for col in LATE_COLS:
        df.loc[df["retard_code_suspect"] == 1, col] = stats["late_medians"][col]

    # --- 3d. DebtRatio : valeurs extrêmes liées à un revenu quasi nul, pas un vrai signal ---
    df["debtratio_aberrant"] = (df["DebtRatio"] > 1).astype(int)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=1)

    return df, stats


def print_cleaning_summary(n_before_dedup, n_after_dedup, train_df, test_df):
    full = pd.concat([train_df, test_df])
    n = len(full)

    def pct(x):
        return f"{x} ({100 * x / n:.2f}%)"

    print("=" * 80)
    print("RESUME DU NETTOYAGE (clean_data)")
    print("=" * 80)

    print(f"\nDoublons : {n_before_dedup} lignes avant -> {n_after_dedup} lignes apres "
          f"(suppression de {n_before_dedup - n_after_dedup} doublons)")

    print("\nValeurs imputees :")
    print(f"  MonthlyIncome_missing      = 1 pour {pct(int(full['MonthlyIncome_missing'].sum()))}")
    print(f"  NumberOfDependents_missing = 1 pour {pct(int(full['NumberOfDependents_missing'].sum()))}")

    print("\nIndicateurs de valeurs aberrantes :")
    print(f"  utilisation_aberrante        = 1 pour {pct(int(full['utilisation_aberrante'].sum()))}")
    print(f"  retard_code_suspect          = 1 pour {pct(int(full['retard_code_suspect'].sum()))}")
    print(f"  debtratio_aberrant           = 1 pour {pct(int(full['debtratio_aberrant'].sum()))}")

    print("\nIndicateurs d'incoherence entre colonnes :")
    print(f"  is_inconsistent_utilization  = 1 pour {pct(int(full['is_inconsistent_utilization'].sum()))}")
    print(f"  is_inconsistent_debtratio    = 1 pour {pct(int(full['is_inconsistent_debtratio'].sum()))}")

    n_na = int(full.isna().sum().sum())
    print(f"\nValeurs manquantes restantes (toutes colonnes confondues) : {n_na}")
    print("-> Aucune valeur manquante restante." if n_na == 0 else "-> ATTENTION : il reste des NaN.")

    util_ok = full["RevolvingUtilizationOfUnsecuredLines"].between(0, 1).all()
    debt_ok = full["DebtRatio"].between(0, 1).all()
    print(f"\nRevolvingUtilizationOfUnsecuredLines dans [0, 1] : {util_ok} "
          f"(min={full['RevolvingUtilizationOfUnsecuredLines'].min()}, "
          f"max={full['RevolvingUtilizationOfUnsecuredLines'].max()})")
    print(f"DebtRatio dans [0, 1] : {debt_ok} "
          f"(min={full['DebtRatio'].min()}, max={full['DebtRatio'].max()})")
    print("=" * 80)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DebtRatio_log"] = np.log1p(df["DebtRatio"].clip(lower=0))
    df["MonthlyIncome_log"] = np.log1p(df["MonthlyIncome"].clip(lower=0))
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    return df


def scale(train_df: pd.DataFrame, test_df: pd.DataFrame, scale_cols: list):
    scaler = RobustScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])
    return train_df, test_df


SCALE_COLS = ["MonthlyIncome_log", "DebtRatio_log", "age"]


def print_feature_engineering_summary(before_df, after_scaled_df, preview_cols, scale_cols):
    new_cols = [c for c in after_scaled_df.columns if c not in before_df.columns]

    print("=" * 80)
    print("RESUME DU FEATURE ENGINEERING + SCALING")
    print("=" * 80)

    print(f"\nColonnes creees par add_features() : {new_cols}")

    print("\nApercu AVANT add_features (colonnes sources, 5 premieres lignes du train) :")
    print(before_df[["DebtRatio", "MonthlyIncome"] + LATE_COLS].head())

    print("\nApercu APRES add_features + scaling (nouvelles colonnes, 5 premieres lignes du train) :")
    print(after_scaled_df[preview_cols].head())

    print(f"\nColonnes mises a l'echelle (RobustScaler, fit sur train uniquement) : {scale_cols}")
    print("\nStatistiques AVANT scaling (train) :")
    print(before_df.assign(
        DebtRatio_log=lambda d: np.log1p(d["DebtRatio"].clip(lower=0)),
        MonthlyIncome_log=lambda d: np.log1p(d["MonthlyIncome"].clip(lower=0)),
    )[scale_cols].describe())
    print("\nStatistiques APRES scaling (train) :")
    print(after_scaled_df[scale_cols].describe())
    print("=" * 80)


def main():
    df = load_raw()

    # 1. Doublons : supprimes avant le split (une ligne dupliquee ne doit pas se
    # retrouver a la fois en train et en test).
    n_before_dedup = len(df)
    df = df.drop_duplicates()
    n_after_dedup = len(df)

    # Split AVANT toute statistique (medianes) pour eviter la fuite de donnees.
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=42
    )

    train_clean, stats = clean_data(train_df)
    test_clean, _ = clean_data(test_df, stats=stats)

    print_cleaning_summary(n_before_dedup, n_after_dedup, train_clean, test_clean)

    # Feature engineering (log-transform, TotalPastDue) puis scaling (fit sur train uniquement).
    train_feat = add_features(train_clean)
    test_feat = add_features(test_clean)

    train_scaled, test_scaled = scale(train_feat, test_feat, SCALE_COLS)

    preview_cols = ["DebtRatio", "DebtRatio_log", "MonthlyIncome", "MonthlyIncome_log", "age", "TotalPastDue"]
    print_feature_engineering_summary(train_clean, train_scaled, preview_cols, SCALE_COLS)

    print(f"\nTrain final : {train_scaled.shape}, Test final : {test_scaled.shape}")

    train_scaled.to_csv(TRAIN_PATH, index=False)
    test_scaled.to_csv(TEST_PATH, index=False)
    print(f"\nSauvegarde : {TRAIN_PATH} et {TEST_PATH}")


if __name__ == "__main__":
    main()
