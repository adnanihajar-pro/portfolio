"""Entraînement, évaluation et analyse SHAP des modèles de détection de CKD
(Logistic Regression vs XGBoost).

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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_pipeline import CATEGORICAL_COLS, NUMERIC_COLS, TARGET

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/metrics.json")
RANDOM_STATE = 42

# Toutes les variables produites par clean_data() : 14 numeriques + 10
# categorielles (deja encodees 0/1) + l'indicateur d'aberration medicale +
# les 24 indicateurs {colonne}_missing. Pas de selection metier ici
# (contrairement a credit-scoring-finance) : avec seulement 400 patients et
# une cible tres separable (cf. notebooks/01_exploration.ipynb), retirer des
# variables n'a pas de justification documentee.
MISSING_INDICATOR_COLS = [f"{col}_missing" for col in NUMERIC_COLS + CATEGORICAL_COLS]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ["valeur_medicale_aberrante"] + MISSING_INDICATOR_COLS


def load_splits():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df[TARGET]

    # StandardScaler fit sur train uniquement, applique aux variables
    # numeriques continues (les indicateurs binaires n'en ont pas besoin).
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_train, X_test, y_train, y_test


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)


def build_xgboost(y_train: pd.Series) -> XGBClassifier:
    # Hyperparametres volontairement conservateurs (max_depth=3, n_estimators=50) :
    # avec seulement 320 lignes de train, un arbre plus profond ou davantage
    # d'arbres augmenteraient le risque de sur-apprentissage sans justification.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return XGBClassifier(
        max_depth=3,
        n_estimators=50,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        n_jobs=-1,
    )


def cross_validate(model, X_train: pd.DataFrame, y_train: pd.Series, name: str) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")
    print(f"      {name:20s} AUC-PR (5 folds) = {np.round(scores, 4)} "
          f"| moyenne={scores.mean():.4f} ecart-type={scores.std():.4f}")
    return scores


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["notckd", "ckd"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matrice de confusion — {name}")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str) -> list[str]:
    """SHAP sur le test set entier (80 lignes) : summary plot + 2 cas
    individuels de patients predits CKD. shap.Explainer choisit
    automatiquement l'explainer adapte (lineaire pour Logistic Regression,
    arbre pour XGBoost)."""
    background = shap.sample(X_train, min(100, len(X_train)), random_state=RANDOM_STATE)
    explainer = shap.Explainer(model.predict_proba, background, feature_names=X_test.columns)
    shap_values = explainer(X_test)

    if shap_values.values.ndim == 3:
        shap_values_class1 = shap_values[..., 1]
    else:
        shap_values_class1 = shap_values

    plt.figure()
    shap.summary_plot(shap_values_class1, X_test, show=False)
    plt.title(f"SHAP summary — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        np.abs(shap_values_class1.values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)
    top_features = mean_abs_shap.head(10).index.tolist()

    y_test_pred = model.predict(X_test)
    ckd_positions = [i for i, pred in enumerate(y_test_pred) if pred == 1][:2]

    for rank, pos in enumerate(ckd_positions, start=1):
        plt.figure()
        shap.plots.waterfall(shap_values_class1[pos], show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_case_{rank}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return top_features


def run() -> None:
    print(f"[1/6] Chargement des donnees ({len(FEATURE_COLS)} variables) ...")
    X_train, X_test, y_train, y_test = load_splits()
    print(f"      -> train: {X_train.shape}, test: {X_test.shape}")

    # Rappel : une AUC-PR quasi parfaite (~1.0) est attendue et documentee
    # (cf. notebooks/01_exploration.ipynb) -- ce dataset contient des
    # variables cliniquement diagnostiques de la CKD (sg, al, hemo, pcv...),
    # ce qui rend la separation quasi triviale. Ce n'est PAS un signe de
    # sur-optimisation ou de fuite de donnees : verifie explicitement lors du
    # diagnostic (pas de leakage via 'id' ou l'ordre du CSV, pas de feature
    # dominante isolee dans les coefficients de la regression logistique).
    print("\n[2/6] Validation croisee (StratifiedKFold, 5 folds) sur le train ...")
    lr_cv_scores = cross_validate(build_logistic_regression(), X_train, y_train, "Logistic Regression")
    xgb_cv_scores = cross_validate(build_xgboost(y_train), X_train, y_train, "XGBoost")

    cv_results = {
        "Logistic Regression": {"auc_pr_mean": lr_cv_scores.mean(), "auc_pr_std": lr_cv_scores.std()},
        "XGBoost": {"auc_pr_mean": xgb_cv_scores.mean(), "auc_pr_std": xgb_cv_scores.std()},
    }
    best_name = max(cv_results, key=lambda k: cv_results[k]["auc_pr_mean"])
    print(f"\n      Meilleur modele (AUC-PR cross-validation) : {best_name} "
          f"(moyenne={cv_results[best_name]['auc_pr_mean']:.4f})")

    print(f"\n[3/6] Entrainement final des deux modeles sur tout le train ...")
    lr_model = build_logistic_regression()
    lr_model.fit(X_train, y_train)
    xgb_model = build_xgboost(y_train)
    xgb_model.fit(X_train, y_train)
    best_model = lr_model if best_name == "Logistic Regression" else xgb_model

    print("\n[4/6] Evaluation finale sur le test set (80 lignes, jamais vu) ...")
    lr_metrics = evaluate(lr_model, X_test, y_test, "Logistic Regression")
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")
    test_results = {"Logistic Regression": lr_metrics, "XGBoost": xgb_metrics}

    print("\n=== Comparaison des modeles (test set) ===")
    comparison_df = pd.DataFrame(test_results).T.drop(columns=["confusion_matrix"])
    print(comparison_df.round(4).to_string())

    print("\nMatrices de confusion (test set, lignes=reel, colonnes=predit [notckd, ckd]) :")
    for name, m in test_results.items():
        print(f"  {name} :")
        for row in m["confusion_matrix"]:
            print(f"    {row}")

    print(f"\n[5/6] Analyse SHAP sur le meilleur modele ({best_name}) ...")
    top_features = shap_analysis(best_model, X_train, X_test, best_name)
    print("      Variables les plus predictives (SHAP, |valeur moyenne| decroissante):")
    for i, feat in enumerate(top_features, start=1):
        print(f"        {i}. {feat}")

    print("\n[6/6] Sauvegarde des modeles et du rapport ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "cross_validation": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in cv_results.items()},
        "test_results": test_results,
        "best_model": best_name,
        "top_shap_features": top_features,
        "features_used": list(X_train.columns),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Resume final (test set) ===")
    for name, m in test_results.items():
        marker = " <== meilleur modele (cross-validation)" if name == best_name else ""
        print(
            f"{name:20s} | precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
            f"f1={m['f1']:.4f}  AUC-PR={m['auc_pr']:.4f}{marker}"
        )
    print(f"\nModeles sauvegardes dans {MODELS_DIR}/")
    print(f"Figures sauvegardees dans {FIGURES_DIR}/")
    print(f"Rapport de metriques sauvegarde dans {METRICS_PATH}")


if __name__ == "__main__":
    run()
