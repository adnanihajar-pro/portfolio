"""Entraînement et évaluation des modèles de classification (Churn) :
Random Forest (One-Hot Encoding) vs CatBoost (catégorielles natives).

Usage:
    python -m src.train_model
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TARGET = "Churn"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/metrics.json")
RANDOM_STATE = 42

# 16 colonnes catégorielles (15 en object + SeniorCitizen, binaire 0/1 mais
# conceptuellement une catégorie, cf. diagnostic) ; 4 colonnes numériques
# (tenure, MonthlyCharges, TotalCharges, charges_incoherentes).
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "charges_incoherentes"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def load_splits():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    # CatBoost impose un type str pour les colonnes déclarées via cat_features
    # (pas de float). SeniorCitizen (int 0/1) est casté en string comme les
    # autres colonnes catégorielles.
    X_train[CATEGORICAL_COLS] = X_train[CATEGORICAL_COLS].astype(str)
    X_test[CATEGORICAL_COLS] = X_test[CATEGORICAL_COLS].astype(str)

    return X_train, X_test, y_train, y_test


def one_hot_encode(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-Hot Encoding fit sur train, colonnes de test alignées sur train
    (une catégorie présente uniquement en test n'apparaît pas au fit -> 0)."""
    X_train_ohe = pd.get_dummies(X_train, columns=CATEGORICAL_COLS)
    X_test_ohe = pd.get_dummies(X_test, columns=CATEGORICAL_COLS)
    X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)
    return X_train_ohe, X_test_ohe


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        class_weight="balanced", n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train) -> CatBoostClassifier:
    model = CatBoostClassifier(
        auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=0
    )
    model.fit(X_train, y_train, cat_features=CATEGORICAL_COLS)
    return model


def evaluate(model, X_test, y_test, name: str) -> dict:
    y_pred = np.ravel(model.predict(X_test))
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas de churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matrice de confusion — {name}")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    return metrics


def _to_class1_explanation(shap_values, X_sample: pd.DataFrame, is_catboost: bool) -> shap.Explanation:
    """Normalise la sortie de shap.TreeExplainer vers une Explanation propre
    pour la classe positive (Churn=1).

    CatBoost (classification binaire) renvoie déjà les valeurs de la classe
    positive en 2D (n, features), mais `.data` pointe vers l'objet Pool passé
    en entrée (inexploitable pour les plots) -> reconstruit avec les valeurs
    réelles de X_sample. Random Forest renvoie un array 3D (n, features, 2
    classes) -> on garde uniquement la tranche classe 1.
    """
    if is_catboost:
        values = shap_values.values
        base_values = shap_values.base_values
    else:
        values = shap_values.values[..., 1]
        base_values = shap_values.base_values[..., 1]

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=X_sample.values,
        feature_names=list(X_sample.columns),
    )


def shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str,
                   is_catboost: bool, sample_size: int = 1000):
    """SHAP sur un échantillon du test set : summary plot + 2 cas individuels
    (client à haut risque bien identifié, cas limite proche de la frontière
    de décision à 0.5)."""
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(X_test.index, size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test.loc[sample_idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    if is_catboost:
        pool_sample = Pool(X_sample, cat_features=CATEGORICAL_COLS)
        shap_values = explainer(pool_sample)
    else:
        shap_values = explainer(X_sample)

    shap_values_class1 = _to_class1_explanation(shap_values, X_sample, is_catboost)

    plt.figure()
    shap.summary_plot(shap_values_class1, show=False)
    plt.title(f"SHAP summary — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        np.abs(shap_values_class1.values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)
    top_features = mean_abs_shap.head(10).index.tolist()

    proba_sample = model.predict_proba(X_sample)[:, 1]

    # Cas 1 : client à haut risque bien identifié -> probabilité prédite
    # maximale dans l'échantillon (client que le modèle juge le plus sûr).
    high_risk_pos = int(np.argmax(proba_sample))

    # Cas 2 : cas limite -> probabilité prédite la plus proche de 0.5.
    borderline_pos = int(np.argmin(np.abs(proba_sample - 0.5)))

    cases = {"high_risk": high_risk_pos, "borderline": borderline_pos}
    case_summaries = {}
    for rank, (case_name, pos) in enumerate(cases.items(), start=1):
        plt.figure()
        shap.plots.waterfall(shap_values_class1[pos], show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_case_{rank}_{case_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        case_summaries[case_name] = {
            "proba_churn": float(proba_sample[pos]),
        }

    return top_features, case_summaries


def run() -> None:
    print(f"[1/6] Chargement des données ({len(FEATURE_COLS)} features : "
          f"{len(CATEGORICAL_COLS)} catégorielles, {len(NUMERIC_COLS)} numériques) ...")
    X_train, X_test, y_train, y_test = load_splits()
    print(f"      -> train: {X_train.shape}, test: {X_test.shape}")

    print("\n[2/6] Entraînement Random Forest (One-Hot Encoding, class_weight='balanced') ...")
    X_train_ohe, X_test_ohe = one_hot_encode(X_train, X_test)
    print(f"      -> {X_train_ohe.shape[1]} colonnes après One-Hot Encoding")
    rf_model = train_random_forest(X_train_ohe, y_train)

    print("[3/6] Entraînement CatBoost (catégorielles natives via cat_features, "
          "auto_class_weights='Balanced') ...")
    cb_model = train_catboost(X_train, y_train)

    print("\n[4/6] Évaluation des deux modèles ...")
    rf_metrics = evaluate(rf_model, X_test_ohe, y_test, "Random Forest")
    cb_metrics = evaluate(cb_model, X_test, y_test, "CatBoost")

    results = {"Random Forest": rf_metrics, "CatBoost": cb_metrics}

    print("\n=== Comparaison des modèles (Precision / Recall / F1 / AUC-PR) ===")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4).to_string())

    best_name = max(results, key=lambda k: results[k]["auc_pr"])
    is_best_catboost = best_name == "CatBoost"
    best_model = cb_model if is_best_catboost else rf_model
    best_X_train = X_train if is_best_catboost else X_train_ohe
    best_X_test = X_test if is_best_catboost else X_test_ohe
    print(f"\nMeilleur modèle (AUC-PR) : {best_name} (AUC-PR={results[best_name]['auc_pr']:.4f})")

    print(f"\n[5/6] Analyse SHAP sur le meilleur modèle ({best_name}) ...")
    top_features, case_summaries = shap_analysis(
        best_model, best_X_train, best_X_test, best_name, is_catboost=is_best_catboost
    )
    print("      Variables les plus prédictives (SHAP, |valeur moyenne| décroissante) :")
    for i, feat in enumerate(top_features, start=1):
        print(f"        {i}. {feat}")
    print(f"      Cas 'haut risque' (proba churn={case_summaries['high_risk']['proba_churn']:.4f}) "
          f"-> reports/figures/shap_case_1_high_risk.png")
    print(f"      Cas 'limite' (proba churn={case_summaries['borderline']['proba_churn']:.4f}) "
          f"-> reports/figures/shap_case_2_borderline.png")

    print("\n[6/6] Sauvegarde du meilleur modèle ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "results": results,
        "best_model": best_name,
        "top_shap_features": top_features,
        "shap_cases": case_summaries,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Résumé final ===")
    for name, m in results.items():
        marker = " <== meilleur modèle" if name == best_name else ""
        print(
            f"{name:15s} | precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
            f"f1={m['f1']:.4f}  AUC-PR={m['auc_pr']:.4f}{marker}"
        )
    print(f"\nModèle sauvegardé : {MODELS_DIR / 'best_model.pkl'}")
    print(f"Figures sauvegardées dans {FIGURES_DIR}/")
    print(f"Rapport de métriques sauvegardé dans {METRICS_PATH}")


if __name__ == "__main__":
    run()
