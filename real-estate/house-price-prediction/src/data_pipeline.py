"""Nettoyage, imputation, retrait des outliers documentés et split pour House Prices."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RAW_PATH = "data/raw/train.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TARGET = "SalePrice"

# Groupe A : NaN documenté dans data_description.txt comme absence de la
# caractéristique (colonnes catégorielles -> "None"). GarageYrBlt (numérique,
# NaN sur les mêmes 81 lignes que les autres colonnes garage) est traité à part.
NA_MEANS_ABSENCE_CAT = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtExposure", "BsmtFinType2", "BsmtQual", "BsmtCond", "BsmtFinType1",
]

# Outliers documentés par l'auteur du dataset (De Cock, 2011) : GrLivArea > 4000
# avec un prix incohérent (surface/qualité/prix), très influents sur une
# régression linéaire. Retrait par Id explicite plutôt que par condition, pour
# rester fidèle à la décision validée (pas de dérive si d'autres lignes venaient
# un jour à dépasser 4000 pi²).
OUTLIER_IDS = [524, 1299]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def remove_documented_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les 2 maisons GrLivArea > 4000 à prix incohérent (Id 524, 1299),
    recommandation officielle de l'auteur du dataset. Fait avant le split : ce
    n'est pas une statistique calculée sur les données, donc aucune fuite."""
    return df[~df["Id"].isin(OUTLIER_IDS)].copy()


def clean_data(df: pd.DataFrame, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Nettoie un DataFrame (train ou test).

    Si `stats` est None, ce df est traité comme le train : la médiane de
    LotFrontage par quartier et le mode d'Electrical sont calculés dessus et
    retournés dans `stats`. Si `stats` est fourni (cas du test), ils sont
    réutilisés tels quels, sans être recalculés, pour éviter toute fuite de
    données train -> test.
    """
    df = df.copy()
    is_train = stats is None
    if is_train:
        stats = {}

    # --- Groupe A : NaN = absence de la caractéristique (mapping fixe, pas de
    # statistique calculée sur les données -> pas de risque de fuite) ---
    for col in NA_MEANS_ABSENCE_CAT:
        df[col] = df[col].fillna("None")
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    # --- Groupe B : NaN = vraiment manquant ---

    # LotFrontage : médiane par quartier calculée sur le train uniquement.
    if is_train:
        stats["lotfrontage_by_neighborhood"] = df.groupby("Neighborhood")["LotFrontage"].median()
        stats["lotfrontage_global_median"] = df["LotFrontage"].median()
    neighborhood_median = df["Neighborhood"].map(stats["lotfrontage_by_neighborhood"])
    df["LotFrontage"] = df["LotFrontage"].fillna(neighborhood_median)
    # Filet de sécurité : quartier absent des stats train, ou médiane de
    # quartier elle-même NaN (quartier sans aucune valeur LotFrontage connue
    # côté train). Repli sur la médiane globale du train.
    df["LotFrontage"] = df["LotFrontage"].fillna(stats["lotfrontage_global_median"])

    # MasVnrType / MasVnrArea : NaN = très probablement pas de revêtement
    # maçonné (mapping fixe, pas de statistique calculée).
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    # Electrical : imputation par le mode du train.
    if is_train:
        stats["electrical_mode"] = df["Electrical"].mode().iloc[0]
    df["Electrical"] = df["Electrical"].fillna(stats["electrical_mode"])

    return df, stats


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute SalePrice_log = log1p(SalePrice), cible d'entraînement (corrige
    l'asymétrie de SalePrice). expm1(SalePrice_log) permet de revenir à
    l'échelle dollar pour l'interprétation finale des métriques (RMSE, etc.)."""
    df = df.copy()
    df["SalePrice_log"] = np.log1p(df[TARGET])
    return df


def print_summary(n_before_outliers, n_after_outliers, train_df, test_df,
                   na_before_train, na_before_test, train_final, test_final, stats):
    print("=" * 80)
    print("RESUME DU NETTOYAGE (data_pipeline.py)")
    print("=" * 80)

    print(f"\nOutliers documentés (Id {OUTLIER_IDS}, GrLivArea > 4000) :")
    print(f"  {n_before_outliers} lignes avant -> {n_after_outliers} lignes après "
          f"(suppression de {n_before_outliers - n_after_outliers} lignes)")

    print(f"\nSplit train/test (avant tout calcul de statistique, random_state=42) :")
    print(f"  train : {len(train_df)} lignes ({len(train_df) / n_after_outliers:.1%})")
    print(f"  test  : {len(test_df)} lignes ({len(test_df) / n_after_outliers:.1%})")

    na_group_a = NA_MEANS_ABSENCE_CAT + ["GarageYrBlt"]
    na_group_b = ["LotFrontage", "MasVnrType", "MasVnrArea", "Electrical"]

    def group_total(na_counts, cols):
        return int(na_counts[cols].sum())

    print("\nValeurs imputées — Groupe A (NaN = absence de la caractéristique) :")
    print(f"  train : {group_total(na_before_train, na_group_a)} valeurs imputées au total")
    for col in na_group_a:
        n_tr, n_te = int(na_before_train[col]), int(na_before_test[col])
        if n_tr or n_te:
            print(f"    {col:15s} -> train: {n_tr:4d}   test: {n_te:4d}")
    print(f"  test  : {group_total(na_before_test, na_group_a)} valeurs imputées au total")

    print("\nValeurs imputées — Groupe B (NaN = vraiment manquant) :")
    for col in na_group_b:
        n_tr, n_te = int(na_before_train[col]), int(na_before_test[col])
        print(f"    {col:15s} -> train: {n_tr:4d}   test: {n_te:4d}")
    print(f"  Total train : {group_total(na_before_train, na_group_b)}   "
          f"Total test : {group_total(na_before_test, na_group_b)}")

    lf_neighborhoods_missing = stats["lotfrontage_by_neighborhood"].isna().sum()
    print(f"\n  Détail LotFrontage :")
    print(f"    Médiane calculée sur {stats['lotfrontage_by_neighborhood'].notna().sum()} "
          f"quartiers (train) ; médiane globale de repli (train) = "
          f"{stats['lotfrontage_global_median']:.1f}")
    if lf_neighborhoods_missing:
        print(f"    ATTENTION : {lf_neighborhoods_missing} quartier(s) du train sans "
              f"aucune valeur LotFrontage connue (repli sur la médiane globale)")
    print(f"    Mode Electrical utilisé pour l'imputation (train) : "
          f"'{stats['electrical_mode']}'")

    n_na_train_after = int(train_final.isna().sum().sum())
    n_na_test_after = int(test_final.isna().sum().sum())
    print(f"\nValeurs manquantes restantes après nettoyage : train={n_na_train_after}, "
          f"test={n_na_test_after}")
    print("-> Aucune valeur manquante restante." if n_na_train_after == 0 and n_na_test_after == 0
          else "-> ATTENTION : il reste des NaN.")

    print("\nCorrection de l'asymétrie de la cible (log1p) :")
    for name, df in [("train", train_final), ("test", test_final)]:
        skew_before = df[TARGET].skew()
        skew_after = df["SalePrice_log"].skew()
        print(f"  {name} : skewness SalePrice = {skew_before:.4f}  ->  "
              f"skewness SalePrice_log = {skew_after:.4f}")

    print("=" * 80)


def main():
    df = load_raw()

    n_before_outliers = len(df)
    df = remove_documented_outliers(df)
    n_after_outliers = len(df)

    # Split AVANT toute statistique (médiane par quartier, mode Electrical) pour
    # éviter la fuite de données. Régression -> pas de stratification possible,
    # split aléatoire simple.
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    na_before_train = train_df.isna().sum()
    na_before_test = test_df.isna().sum()

    train_clean, stats = clean_data(train_df)
    test_clean, _ = clean_data(test_df, stats=stats)

    train_final = add_log_target(train_clean)
    test_final = add_log_target(test_clean)

    print_summary(n_before_outliers, n_after_outliers, train_df, test_df,
                  na_before_train, na_before_test, train_final, test_final, stats)

    print(f"\nTrain final : {train_final.shape}, Test final : {test_final.shape}")

    train_final.to_csv(TRAIN_PATH, index=False)
    test_final.to_csv(TEST_PATH, index=False)
    print(f"\nSauvegarde : {TRAIN_PATH} et {TEST_PATH}")


if __name__ == "__main__":
    main()
