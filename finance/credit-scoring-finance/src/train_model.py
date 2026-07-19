"""Sélection de features, entraînement, évaluation et analyse SHAP des modèles
de credit scoring (Logistic Regression vs XGBoost).

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TARGET = "SeriousDlqin2yrs"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/metrics.json")
RANDOM_STATE = 42

# 16 variables métier alignées sur les critères bancaires classiques (Capacity,
# Character) + leurs indicateurs de valeur manquante / aberrante associés.
# Exclues : NumberRealEstateLoansOrLines, NumberOfDependents,
# NumberOfDependents_missing (pas de lien clair avec Capacity/Character) ;
# DebtRatio (colinéaire avec DebtRatio_log, seule cette dernière est conservée
# pour éviter de fausser l'interprétation SHAP).
FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "utilisation_aberrante",
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "retard_code_suspect",
    "TotalPastDue",
    "age",
    "MonthlyIncome",
    "MonthlyIncome_missing",
    "MonthlyIncome_log",
    "debtratio_aberrant",
    "DebtRatio_log",
    "NumberOfOpenCreditLinesAndLoans",
    "is_inconsistent_utilization",
    "is_inconsistent_debtratio",
]

# Colonnes numériques restantes, pas encore mises à l'échelle par le pipeline
# (age, DebtRatio_log, MonthlyIncome_log l'ont déjà été via RobustScaler ;
# RevolvingUtilizationOfUnsecuredLines est déjà bornée [0, 1] ; les indicateurs
# binaires n'ont pas besoin de scaling).
STANDARD_SCALE_COLS = [
    "MonthlyIncome",
    "TotalPastDue",
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfOpenCreditLinesAndLoans",
]


def load_splits():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df[TARGET]

    # StandardScaler fit sur train uniquement, appliqué à train et test (même
    # logique anti-fuite que le RobustScaler du pipeline de données).
    scaler = StandardScaler()
    X_train[STANDARD_SCALE_COLS] = scaler.fit_transform(X_train[STANDARD_SCALE_COLS])
    X_test[STANDARD_SCALE_COLS] = scaler.transform(X_test[STANDARD_SCALE_COLS])

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas de défaut", "Défaut"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matrice de confusion — {name}")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    return metrics


def shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str,
                   sample_size: int = 1000) -> list[str]:
    """SHAP sur un échantillon du test set : summary plot + 2 cas individuels
    de défaut. Utilise shap.Explainer, qui choisit automatiquement l'explainer
    adapté au type de modèle (linéaire pour Logistic Regression, arbre pour XGBoost)."""
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(X_test.index, size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test.loc[sample_idx]

    background = shap.sample(X_train, 100, random_state=RANDOM_STATE)
    explainer = shap.Explainer(model.predict_proba, background, feature_names=X_sample.columns)
    shap_values = explainer(X_sample)

    if shap_values.values.ndim == 3:
        shap_values_class1 = shap_values[..., 1]
    else:
        shap_values_class1 = shap_values

    plt.figure()
    shap.summary_plot(shap_values_class1, X_sample, show=False)
    plt.title(f"SHAP summary — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        np.abs(shap_values_class1.values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)
    top_features = mean_abs_shap.head(10).index.tolist()

    y_sample = model.predict(X_sample)
    default_positions = [i for i, pred in enumerate(y_sample) if pred == 1][:2]

    for rank, pos in enumerate(default_positions, start=1):
        plt.figure()
        shap.plots.waterfall(shap_values_class1[pos], show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_case_{rank}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return top_features


def run() -> None:
    print(f"[1/6] Chargement des données et sélection des {len(FEATURE_COLS)} variables métier ...")
    X_train, X_test, y_train, y_test = load_splits()
    print(f"      -> train: {X_train.shape}, test: {X_test.shape}")
    print(f"      -> features retenues: {list(X_train.columns)}")

    print("\n[2/6] Entraînement Logistic Regression (class_weight='balanced') ...")
    lr_model = train_logistic_regression(X_train, y_train)

    print("[3/6] Entraînement XGBoost (scale_pos_weight) ...")
    xgb_model = train_xgboost(X_train, y_train)

    print("\n[4/6] Évaluation des deux modèles ...")
    lr_metrics = evaluate(lr_model, X_test, y_test, "Logistic Regression")
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")

    results = {"Logistic Regression": lr_metrics, "XGBoost": xgb_metrics}

    print("\n=== Comparaison des modèles ===")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4).to_string())

    best_name = max(results, key=lambda k: results[k]["auc_pr"])
    best_model = lr_model if best_name == "Logistic Regression" else xgb_model
    print(f"\nMeilleur modèle (AUC-PR): {best_name} (AUC-PR={results[best_name]['auc_pr']:.4f})")

    print(f"\n[5/6] Analyse SHAP sur le meilleur modèle ({best_name}) ...")
    top_features = shap_analysis(best_model, X_train, X_test, best_name)
    print("      Variables les plus prédictives (SHAP, |valeur moyenne| décroissante):")
    for i, feat in enumerate(top_features, start=1):
        print(f"        {i}. {feat}")

    print("\n[6/6] Sauvegarde des modèles et du rapport ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "results": results,
        "best_model": best_name,
        "top_shap_features": top_features,
        "features_used": list(X_train.columns),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Résumé final ===")
    for name, m in results.items():
        marker = " <== meilleur modèle" if name == best_name else ""
        print(
            f"{name:20s} | accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
            f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  AUC-PR={m['auc_pr']:.4f}{marker}"
        )
    print(f"\nModèles sauvegardés dans {MODELS_DIR}/")
    print(f"Figures sauvegardées dans {FIGURES_DIR}/")
    print(f"Rapport de métriques sauvegardé dans {METRICS_PATH}")


if __name__ == "__main__":
    run()
