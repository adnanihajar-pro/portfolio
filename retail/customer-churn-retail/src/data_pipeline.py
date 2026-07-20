"""Nettoyage et split pour Telco Customer Churn."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TARGET = "Churn"

# Colonnes où "No internet service" / "No phone service" sont des doublons
# fonctionnels de "No" (le client n'a simplement pas le service parent) plutôt
# qu'une vraie 3e catégorie informative.
REDUNDANT_NO_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

INCOHERENCE_THRESHOLD_PCT = 10


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Nettoie un DataFrame (train ou test).

    Toutes les transformations ici sont des règles métier fixes (pas de
    moyenne/médiane/mode calculés sur les données) : elles ne créent donc pas
    de fuite train -> test. `stats` est conservé pour rester cohérent avec le
    pattern des autres projets et pour accueillir de futures statistiques
    (ex. scaling) sans changer la signature.
    """
    df = df.copy()
    if stats is None:
        stats = {}

    # --- customerID : identifiant sans valeur prédictive, exclu des features ---
    df = df.drop(columns=["customerID"])

    # --- TotalCharges : object -> numérique. Les 11 NaN résultants
    # correspondent exactement aux clients tenure=0 (nouveaux clients, pas
    # encore de facture cumulée) -> remplacés par 0 (règle métier, pas une
    # imputation statistique). ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --- Catégories redondantes : "No internet/phone service" -> "No" ---
    df[REDUNDANT_NO_COLS] = df[REDUNDANT_NO_COLS].replace(
        {"No internet service": "No", "No phone service": "No"}
    )

    # --- Indicateur d'incohérence TotalCharges vs tenure*MonthlyCharges,
    # sans modifier les valeurs sous-jacentes. tenure=0 -> expected_total=0 ->
    # division 0/0 traitée comme "pas d'incohérence" (TotalCharges vaut aussi 0
    # pour ces lignes après l'imputation ci-dessus). ---
    expected_total = df["tenure"] * df["MonthlyCharges"]
    rel_diff_pct = (df["TotalCharges"] - expected_total).abs() / expected_total.replace(0, np.nan) * 100
    df["charges_incoherentes"] = (rel_diff_pct > INCOHERENCE_THRESHOLD_PCT).fillna(False).astype(int)

    # --- Cible : Churn -> 1 (Yes) / 0 (No) ---
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df, stats


def print_summary(n_before_dedup, n_after_dedup, train_raw, test_raw, train_clean, test_clean):
    print("=" * 90)
    print("RESUME DU NETTOYAGE (clean_data)")
    print("=" * 90)

    print(f"\nShape brut (avant suppression des doublons) : ({n_before_dedup}, 21)")

    n_removed = n_before_dedup - n_after_dedup
    print(f"\nDoublons (hors customerID, 20 groupes : 18 paires + 2 triplets, "
          f"tenure/MonthlyCharges/TotalCharges identiques confirmés) :")
    print(f"  {n_before_dedup} lignes avant -> {n_after_dedup} lignes après "
          f"(suppression de {n_removed} lignes excédentaires, une seule occurrence gardée par groupe)")
    print("  Supprimés AVANT le split, pour qu'un même profil ne se retrouve pas à la fois "
          "en train et en test.")

    print(f"\nSplit (avant toute statistique, stratify=Churn, test_size=0.2, random_state=42) :")
    print(f"  train : {train_raw.shape}  ({len(train_raw) / n_after_dedup:.1%})")
    print(f"  test  : {test_raw.shape}  ({len(test_raw) / n_after_dedup:.1%})")

    print("\n--- customerID ---")
    print("  Colonne supprimée (identifiant unique par ligne, aucune valeur prédictive).")

    print("\n--- TotalCharges (object -> numérique) ---")
    for name, raw_part in [("train", train_raw), ("test", test_raw)]:
        n_bad = pd.to_numeric(raw_part["TotalCharges"], errors="coerce").isna().sum()
        print(f"  {name} : {n_bad} valeurs non convertibles -> remplacées par 0 (tenure=0)")

    print("\n--- Catégories redondantes fusionnées en 'No' ---")
    for col in REDUNDANT_NO_COLS:
        n_train_before = (train_raw[col].isin(["No internet service", "No phone service"])).sum()
        n_test_before = (test_raw[col].isin(["No internet service", "No phone service"])).sum()
        if n_train_before or n_test_before:
            print(f"  {col:20s} -> train: {n_train_before:4d} fusionnées   test: {n_test_before:4d} fusionnées")
    print("  Valeurs uniques restantes après fusion (train, exemple OnlineSecurity) :",
          sorted(train_clean["OnlineSecurity"].unique().tolist()))

    print("\n--- Indicateur charges_incoherentes (écart relatif > "
          f"{INCOHERENCE_THRESHOLD_PCT}%, valeurs sous-jacentes non modifiées) ---")
    for name, part in [("train", train_clean), ("test", test_clean)]:
        n = int(part["charges_incoherentes"].sum())
        print(f"  {name} : {n} lignes marquées ({n / len(part):.2%})")

    print("\n--- Cible Churn (encodage Yes/No -> 1/0) ---")
    for name, raw_part, clean_part in [("train", train_raw, train_clean), ("test", test_raw, test_clean)]:
        before = raw_part[TARGET].value_counts()
        after = clean_part[TARGET].value_counts()
        pct_yes = clean_part[TARGET].mean()
        print(f"  {name} : avant {dict(before)}  ->  après {dict(after)}  (taux de churn = {pct_yes:.2%})")

    n_na_train = int(train_clean.isna().sum().sum())
    n_na_test = int(test_clean.isna().sum().sum())
    print(f"\nValeurs manquantes restantes : train={n_na_train}, test={n_na_test}")
    print("-> Aucune valeur manquante restante." if n_na_train == 0 and n_na_test == 0
          else "-> ATTENTION : il reste des NaN.")

    print(f"\nShape final : train={train_clean.shape}, test={test_clean.shape}")
    print("=" * 90)


def main():
    df = load_raw()

    # Doublons (hors customerID) : correspondance exacte confirmée sur les
    # variables continues (tenure, MonthlyCharges, TotalCharges) en plus des
    # colonnes catégorielles -> supprimés avant le split, comme pour les
    # autres projets du portfolio, pour qu'un même profil ne se retrouve pas
    # à la fois en train et en test.
    cols_no_id = [c for c in df.columns if c != "customerID"]
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=cols_no_id, keep="first")
    n_after_dedup = len(df)

    # Split AVANT toute statistique. Stratification sur Churn (variable brute
    # Yes/No, encodée en 0/1 seulement après le split, dans clean_data).
    train_raw, test_raw = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=42
    )

    train_clean, stats = clean_data(train_raw)
    test_clean, _ = clean_data(test_raw, stats=stats)

    print_summary(n_before_dedup, n_after_dedup, train_raw, test_raw, train_clean, test_clean)

    train_clean.to_csv(TRAIN_PATH, index=False)
    test_clean.to_csv(TEST_PATH, index=False)
    print(f"\nSauvegarde : {TRAIN_PATH} et {TEST_PATH}")


if __name__ == "__main__":
    main()
