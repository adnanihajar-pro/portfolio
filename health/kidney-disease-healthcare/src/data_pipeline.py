"""Nettoyage, feature engineering et split pour le dataset Chronic Kidney Disease."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RAW_PATH = "data/raw/kidney_disease.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TARGET = "classification"

NUMERIC_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
CATEGORICAL_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

# Encodage 0/1 explicite des colonnes binaires : 1 = modalite associee a un
# risque/une anomalie clinique, 0 = modalite normale/absente.
# rbc, pc      : normal -> 1, abnormal -> 0
# pcc, ba      : present -> 1, notpresent -> 0
# htn, dm, cad, pe, ane : yes -> 1, no -> 0
# appet        : good -> 1, poor -> 0
BINARY_MAPS = {
    "rbc": {"normal": 1, "abnormal": 0},
    "pc": {"normal": 1, "abnormal": 0},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0},
    "cad": {"yes": 1, "no": 0},
    "appet": {"good": 1, "poor": 0},
    "pe": {"yes": 1, "no": 0},
    "ane": {"yes": 1, "no": 0},
}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _clean_target(series: pd.Series) -> pd.Series:
    """Normalisation deterministe (strip + mapping) de la cible.

    N'implique aucune statistique calculee sur les donnees (pas de moyenne, de
    mediane, etc.) : c'est une transformation ligne a ligne fixe, donc sans
    risque de fuite meme appliquee avant le split (necessaire ici pour pouvoir
    stratifier le split sur la cible).
    """
    cleaned = series.astype(str).str.strip()
    return cleaned.map({"ckd": 1, "notckd": 0})


def clean_data(df: pd.DataFrame, stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Nettoie un DataFrame (train ou test).

    Si `stats` est None, ce df est traite comme le train : les medianes/modes
    sont calcules dessus et retournes dans `stats`. Si `stats` est fourni (cas
    du test), les statistiques du train sont reutilisees telles quelles, sans
    etre recalculees, pour eviter toute fuite de donnees train -> test.
    """
    df = df.copy()
    is_train = stats is None
    if is_train:
        stats = {}

    # --- 0. id : encode uniquement la position dans le CSV, pas une feature ---
    df = df.drop(columns=["id"])

    # --- 1. pcv/wc/rc : tabs et '\t?' -> NaN, conversion en numerique ---
    for col in ["pcv", "wc", "rc"]:
        df[col] = df[col].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 2. dm/cad/classification : espaces/tabs residuels sur le texte ---
    for col in ["dm", "cad", "classification"]:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})

    # --- 3. Encodage cible : classification -> 1 (ckd) / 0 (notckd) ---
    df[TARGET] = df["classification"].map({"ckd": 1, "notckd": 0})

    # --- 4. Valeurs medicalement impossibles -> NaN, avec indicateur dedie ---
    # sc (creatinine) > 20 mg/dL, sod (sodium) < 100 mEq/L, pot (potassium) > 15
    # mEq/L : incompatibles avec la vie, donc traitees comme des erreurs de
    # saisie plutot que comme un signal clinique, et remplacees par NaN.
    medically_impossible = (df["sc"] > 20) | (df["sod"] < 100) | (df["pot"] > 15)
    df["valeur_medicale_aberrante"] = medically_impossible.astype(int)
    df.loc[df["sc"] > 20, "sc"] = np.nan
    df.loc[df["sod"] < 100, "sod"] = np.nan
    df.loc[df["pot"] > 15, "pot"] = np.nan

    # --- 5. Indicateurs {colonne}_missing, calcules AVANT imputation ---
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # --- 6. Imputation mediane (numerique) / mode (categorielle) ---
    # Stats calculees sur le train uniquement, reutilisees telles quelles sur
    # le test pour eviter toute fuite de donnees.
    if is_train:
        stats["medians"] = {col: df[col].median() for col in NUMERIC_COLS}
        stats["modes"] = {col: df[col].mode(dropna=True).iloc[0] for col in CATEGORICAL_COLS}
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(stats["medians"][col])
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(stats["modes"][col])

    # --- 7. Encodage 0/1 explicite des colonnes binaires (voir BINARY_MAPS) ---
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping)

    return df, stats


def print_cleaning_summary(n_raw, n_missing_raw, train_df, test_df):
    full = pd.concat([train_df, test_df])
    n = len(full)

    def pct(x):
        return f"{x} ({100 * x / n:.2f}%)"

    print("=" * 80)
    print("RESUME DU NETTOYAGE (clean_data)")
    print("=" * 80)

    print(f"\nLignes : {n_raw} avant nettoyage -> {n} apres (train+test), colonne 'id' retiree")

    n_missing_clean = int(full[NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]].isna().sum().sum())
    print(f"\nValeurs manquantes (features + cible) : {n_missing_raw} avant -> {n_missing_clean} apres imputation")
    print("-> Aucune valeur manquante restante." if n_missing_clean == 0 else "-> ATTENTION : il reste des NaN.")

    print("\nIndicateur de valeurs medicalement aberrantes (sc>20, sod<100, pot>15) :")
    print(f"  valeur_medicale_aberrante = 1 pour {pct(int(full['valeur_medicale_aberrante'].sum()))}")

    print("\nIndicateurs de valeurs manquantes cree (avant imputation), non nuls :")
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        n_missing_col = int(full[f"{col}_missing"].sum())
        if n_missing_col:
            print(f"  {col}_missing".ljust(22) + f"= 1 pour {pct(n_missing_col)}")

    print("\nEncodage des colonnes binaires (valeurs uniques apres encodage) :")
    for col in CATEGORICAL_COLS:
        print(f"  {col} : {sorted(full[col].unique().tolist())}")

    print(f"\nDistribution de la cible ({TARGET}) : {full[TARGET].value_counts().to_dict()}")
    print(f"\nTrain : {train_df.shape}, Test : {test_df.shape}")
    print("=" * 80)


def main():
    df = load_raw()
    n_raw = len(df)
    n_missing_raw = int(df.isna().sum().sum())

    # Cible normalisee (transformation deterministe, sans statistique) pour
    # pouvoir stratifier le split -- split AVANT tout calcul de mediane/mode.
    target_for_split = _clean_target(df["classification"])

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=target_for_split, random_state=42
    )

    train_clean, stats = clean_data(train_df)
    test_clean, _ = clean_data(test_df, stats=stats)

    print_cleaning_summary(n_raw, n_missing_raw, train_clean, test_clean)

    train_clean.to_csv(TRAIN_PATH, index=False)
    test_clean.to_csv(TEST_PATH, index=False)
    print(f"\nSauvegarde : {TRAIN_PATH} et {TEST_PATH}")


if __name__ == "__main__":
    main()
