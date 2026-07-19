"""Train, evaluate, compare and explain fraud detection models.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42

TARGET = "Class"
DROP_COLS = ["Time", "Class"]  # Time is superseded by Hour_sin/Hour_cos


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train_df, test_df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]
    return X, y


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Légitime", "Fraude"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matrice de confusion — {name}")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    return metrics


def shap_analysis(model, X_test: pd.DataFrame, model_name: str, sample_size: int = 1000) -> list[str]:
    """Run SHAP TreeExplainer on a sample of the test set, save a summary plot
    and 1-2 individual case explanations. Returns the top predictive features."""
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(X_test.index, size=min(sample_size, len(X_test)), replace=False)
    X_sample = X_test.loc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # For binary classifiers shap_values may have shape (n, n_features, 2)
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

    # Individual case explanations: pick up to 2 cases predicted as fraud in the sample
    y_sample = model.predict(X_sample)
    fraud_positions = [i for i, pred in enumerate(y_sample) if pred == 1][:2]

    for rank, pos in enumerate(fraud_positions, start=1):
        plt.figure()
        shap.plots.waterfall(shap_values_class1[pos], show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_case_{rank}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return top_features


def run() -> None:
    print("[1/6] Loading processed train/test data ...")
    train_df, test_df = load_processed()
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    print(f"      -> train: {X_train.shape}, test: {X_test.shape}")

    print("[2/6] Training Random Forest (class_weight='balanced') ...")
    rf_model = train_random_forest(X_train, y_train)

    print("[3/6] Training XGBoost (scale_pos_weight) ...")
    xgb_model = train_xgboost(X_train, y_train)

    print("[4/6] Evaluating both models ...")
    rf_metrics = evaluate(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost")

    results = {"Random Forest": rf_metrics, "XGBoost": xgb_metrics}

    print("\n=== Comparaison des modèles ===")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4).to_string())

    best_name = max(results, key=lambda k: results[k]["auc_pr"])
    best_model = rf_model if best_name == "Random Forest" else xgb_model
    print(f"\nMeilleur modèle (AUC-PR): {best_name} (AUC-PR={results[best_name]['auc_pr']:.4f})")

    print(f"\n[5/6] Analyse SHAP sur le meilleur modèle ({best_name}) ...")
    top_features = shap_analysis(best_model, X_test, best_name)
    print("      Variables les plus prédictives (SHAP, |valeur moyenne| décroissante):")
    for i, feat in enumerate(top_features, start=1):
        print(f"        {i}. {feat}")

    print("\n[6/6] Sauvegarde des modèles et du rapport ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "results": results,
        "best_model": best_name,
        "top_shap_features": top_features,
        "features_used": list(X_train.columns),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Résumé final ===")
    for name, m in results.items():
        marker = " <== meilleur modèle" if name == best_name else ""
        print(
            f"{name:15s} | accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
            f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  AUC-PR={m['auc_pr']:.4f}{marker}"
        )
    print(f"\nModèles sauvegardés dans {MODELS_DIR}/")
    print(f"Figures sauvegardées dans {FIGURES_DIR}/")
    print(f"Rapport de métriques sauvegardé dans {REPORTS_DIR}/metrics.json")


if __name__ == "__main__":
    run()
