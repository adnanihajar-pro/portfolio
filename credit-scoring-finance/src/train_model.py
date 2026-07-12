"""Entraînement, évaluation et analyse SHAP des modèles de credit scoring."""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

TARGET = "SeriousDlqin2yrs"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"
METRICS_PATH = "reports/metrics.json"


def load_splits():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_pr": average_precision_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {name}")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/confusion_matrix_{name.lower().replace(' ', '_')}.png")
    plt.close(fig)

    return metrics


def main():
    X_train, X_test, y_train, y_test = load_splits()

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="aucpr",
    )
    xgb.fit(X_train, y_train)
    xgb_metrics = evaluate("XGBoost", xgb, X_test, y_test)

    results = {"Random Forest": rf_metrics, "XGBoost": xgb_metrics}
    best_name = max(results, key=lambda k: results[k]["auc_pr"])
    best_model = rf if best_name == "Random Forest" else xgb

    joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")
    joblib.dump(xgb, f"{MODELS_DIR}/xgboost.pkl")

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    shap_values_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(shap_values_pos, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_summary.png")
    plt.close()

    mean_abs_shap = pd.Series(
        abs(shap_values_pos).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)
    top_features = mean_abs_shap.head(10).index.tolist()

    summary = {
        "best_model": best_name,
        "metrics": results,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "top_shap_features": top_features,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Meilleur modèle : {best_name}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
