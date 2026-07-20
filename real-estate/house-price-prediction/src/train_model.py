"""Encodage, entraînement, évaluation et analyse SHAP des modèles de prédiction
de prix immobilier (Ridge vs XGBoost).

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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TARGET = "SalePrice"
LOG_TARGET = "SalePrice_log"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/metrics.json")
RANDOM_STATE = 42
N_SPLITS = 5

DROP_COLS = ["Id", TARGET, LOG_TARGET]


def load_splits():
    """Charge train/test, encode les variables catégorielles (One-Hot, fit sur
    train uniquement, colonnes de test alignées sur celles du train) et
    construit deux jeux de features : un mis à l'échelle pour Ridge
    (StandardScaler sur les colonnes numériques uniquement, fit sur train),
    un brut pour XGBoost (inutile de standardiser pour un modèle à base
    d'arbres)."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    y_train_log = train_df[LOG_TARGET]
    y_test_log = test_df[LOG_TARGET]
    y_train_dollars = train_df[TARGET]
    y_test_dollars = test_df[TARGET]

    X_train_raw = train_df.drop(columns=DROP_COLS)
    X_test_raw = test_df.drop(columns=DROP_COLS)
    numeric_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()

    # One-Hot Encoding : fit sur train uniquement. reindex aligne les colonnes
    # de test sur celles du train (categories absentes du train -> colonnes
    # supprimees ; categories absentes du test -> colonnes a 0).
    # dtype=float64 explicite : pd.get_dummies produit des colonnes bool par
    # defaut (pandas >= 2.x), ce qui fait planter shap.summary_plot (mix de
    # types non supporte en interne).
    X_train_encoded = pd.get_dummies(X_train_raw, dtype=np.float64)
    X_test_encoded = pd.get_dummies(X_test_raw, dtype=np.float64).reindex(
        columns=X_train_encoded.columns, fill_value=0
    )

    # Version XGBoost : encodee, non standardisee.
    X_train_xgb = X_train_encoded.copy()
    X_test_xgb = X_test_encoded.copy()

    # Version Ridge : encodee + StandardScaler sur les colonnes numeriques
    # uniquement (fit sur train, applique a train et test).
    scaler = StandardScaler()
    X_train_ridge = X_train_encoded.copy()
    X_test_ridge = X_test_encoded.copy()
    X_train_ridge[numeric_cols] = scaler.fit_transform(X_train_ridge[numeric_cols])
    X_test_ridge[numeric_cols] = scaler.transform(X_test_ridge[numeric_cols])

    return {
        "X_train_ridge": X_train_ridge, "X_test_ridge": X_test_ridge,
        "X_train_xgb": X_train_xgb, "X_test_xgb": X_test_xgb,
        "y_train_log": y_train_log, "y_test_log": y_test_log,
        "y_train_dollars": y_train_dollars, "y_test_dollars": y_test_dollars,
        "numeric_cols": numeric_cols,
        "feature_cols": X_train_encoded.columns.tolist(),
    }


def train_ridge(X_train, y_train_log) -> Ridge:
    model = Ridge(alpha=10)
    model.fit(X_train, y_train_log)
    return model


def train_xgboost(X_train, y_train_log) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_log)
    return model


def evaluate(model, X_test, y_test_dollars, name: str):
    """Prédit sur l'échelle log puis reconvertit en dollars (expm1) avant de
    calculer RMSE/MAE/R², pour que les métriques soient interprétables dans
    l'unité du problème (dollars), pas en échelle logarithmique."""
    y_pred_log = model.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)

    metrics = {
        "rmse": root_mean_squared_error(y_test_dollars, y_pred_dollars),
        "mae": mean_absolute_error(y_test_dollars, y_pred_dollars),
        "r2": r2_score(y_test_dollars, y_pred_dollars),
    }
    return metrics, y_pred_dollars


def _rmse_dollars(y_true_log, y_pred_log):
    """Scorer RMSE en dollars à partir de prédictions/valeurs en échelle log."""
    return root_mean_squared_error(np.expm1(y_true_log), np.expm1(y_pred_log))


RMSE_DOLLARS_SCORER = make_scorer(_rmse_dollars, greater_is_better=False)


def cross_validate_model(model, X_train, y_train_log, numeric_cols, scale: bool):
    """KFold à 5 plis sur le train, RMSE en dollars (échelle interprétable).
    Pour Ridge (scale=True), le StandardScaler est intégré dans un Pipeline
    afin d'être re-fit à chaque pli (sur le sous-train du pli uniquement),
    et non sur tout le train, pour éviter toute fuite de données pendant la
    validation croisée elle-même."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    if scale:
        preprocessor = ColumnTransformer(
            transformers=[("scale", StandardScaler(), numeric_cols)],
            remainder="passthrough",
        )
        estimator = Pipeline([("preprocess", preprocessor), ("model", model)])
    else:
        estimator = model

    neg_rmse_scores = cross_val_score(
        estimator, X_train, y_train_log, cv=kf, scoring=RMSE_DOLLARS_SCORER, n_jobs=-1
    )
    rmse_scores = -neg_rmse_scores
    return {"cv_rmse_mean": rmse_scores.mean(), "cv_rmse_std": rmse_scores.std()}


def plot_predictions_vs_actual(y_test_dollars, y_pred_dollars, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test_dollars, y_pred_dollars, alpha=0.5, edgecolor="k", linewidth=0.3)
    lims = [
        min(y_test_dollars.min(), y_pred_dollars.min()),
        max(y_test_dollars.max(), y_pred_dollars.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="Prédiction parfaite")
    ax.set_xlabel("SalePrice réel ($)")
    ax.set_ylabel("SalePrice prédit ($)")
    ax.set_title(f"Prédictions vs valeurs réelles — {model_name}")
    ax.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"pred_vs_actual_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def shap_analysis(model, X_train, X_test, model_name: str, y_test_dollars, y_pred_dollars,
                   sample_size: int = 300):
    """SHAP sur le test set (ou un échantillon si > sample_size) : summary
    plot + 2 cas individuels (la maison la mieux prédite et celle avec la plus
    grosse erreur). LinearExplainer pour Ridge (exact, rapide), TreeExplainer
    pour XGBoost (exact, rapide) — plus adaptés ici que l'explainer générique
    utilisé pour les modèles de classification du portfolio, vu le nombre de
    features après One-Hot (permutation générique serait trop lente)."""
    background = shap.sample(X_train, min(100, len(X_train)), random_state=RANDOM_STATE)

    if isinstance(model, Ridge):
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model)

    if len(X_test) > sample_size:
        sample_idx = X_test.sample(sample_size, random_state=RANDOM_STATE).index
        X_sample = X_test.loc[sample_idx]
        y_test_sample = y_test_dollars.loc[sample_idx]
        y_pred_sample = pd.Series(y_pred_dollars, index=X_test.index).loc[sample_idx]
    else:
        X_sample = X_test
        y_test_sample = y_test_dollars
        y_pred_sample = pd.Series(y_pred_dollars, index=X_test.index)

    shap_values = explainer(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP summary — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)
    top_features = mean_abs_shap.head(10).index.tolist()

    # Cas individuels : residus en dollars, sur le meme echantillon que SHAP.
    residuals = (y_pred_sample - y_test_sample).abs()
    best_pos = residuals.values.argmin()
    worst_pos = residuals.values.argmax()

    for label, pos in [("best_case", best_pos), ("worst_case", worst_pos)]:
        plt.figure()
        shap.plots.waterfall(shap_values[pos], show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    best_id = X_sample.index[best_pos]
    worst_id = X_sample.index[worst_pos]
    case_summary = {
        "best_case": {
            "real_price": float(y_test_sample.iloc[best_pos]),
            "predicted_price": float(y_pred_sample.iloc[best_pos]),
            "abs_error": float(residuals.iloc[best_pos]),
        },
        "worst_case": {
            "real_price": float(y_test_sample.iloc[worst_pos]),
            "predicted_price": float(y_pred_sample.iloc[worst_pos]),
            "abs_error": float(residuals.iloc[worst_pos]),
        },
    }

    return top_features, case_summary


def run() -> None:
    print("[1/7] Chargement des données, encodage One-Hot et scaling ...")
    data = load_splits()
    print(f"      -> train: {data['X_train_ridge'].shape}, test: {data['X_test_ridge'].shape}")
    print(f"      -> {len(data['feature_cols'])} features après encodage One-Hot "
          f"({len(data['numeric_cols'])} numériques standardisées pour Ridge)")

    print("\n[2/7] Entraînement Ridge (alpha=10) sur SalePrice_log ...")
    ridge_model = train_ridge(data["X_train_ridge"], data["y_train_log"])

    print("[3/7] Entraînement XGBoost (n_estimators=300, max_depth=4, "
          "learning_rate=0.05) sur SalePrice_log ...")
    xgb_model = train_xgboost(data["X_train_xgb"], data["y_train_log"])

    print("\n[4/7] Évaluation sur le test set (métriques reconverties en dollars, expm1) ...")
    ridge_metrics, ridge_pred = evaluate(ridge_model, data["X_test_ridge"], data["y_test_dollars"], "Ridge")
    xgb_metrics, xgb_pred = evaluate(xgb_model, data["X_test_xgb"], data["y_test_dollars"], "XGBoost")

    print("[5/7] Validation croisée (KFold, 5 plis, RMSE en dollars) sur le train ...")
    ridge_cv = cross_validate_model(
        Ridge(alpha=10), data["X_train_ridge"], data["y_train_log"], data["numeric_cols"], scale=True
    )
    xgb_cv = cross_validate_model(
        XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                      random_state=RANDOM_STATE, n_jobs=-1),
        data["X_train_xgb"], data["y_train_log"], data["numeric_cols"], scale=False,
    )

    results = {
        "Ridge": {**ridge_metrics, **ridge_cv},
        "XGBoost": {**xgb_metrics, **xgb_cv},
    }

    print("\n=== Comparaison des modèles (métriques en dollars) ===")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(2).to_string())

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = ridge_model if best_name == "Ridge" else xgb_model
    best_pred = ridge_pred if best_name == "Ridge" else xgb_pred
    best_X_train = data["X_train_ridge"] if best_name == "Ridge" else data["X_train_xgb"]
    best_X_test = data["X_test_ridge"] if best_name == "Ridge" else data["X_test_xgb"]
    print(f"\nMeilleur modèle (RMSE, métrique de référence) : {best_name} "
          f"(RMSE=${results[best_name]['rmse']:,.0f})")

    print(f"\n[6/7] Analyse SHAP + scatter prédictions vs réel sur le meilleur modèle "
          f"({best_name}) ...")
    plot_predictions_vs_actual(data["y_test_dollars"], best_pred, best_name)
    top_features, case_summary = shap_analysis(
        best_model, best_X_train, best_X_test, best_name, data["y_test_dollars"], best_pred
    )
    print("      Variables les plus prédictives (SHAP, |valeur moyenne| décroissante):")
    for i, feat in enumerate(top_features, start=1):
        print(f"        {i}. {feat}")
    print(f"      Cas le mieux prédit   : réel=${case_summary['best_case']['real_price']:,.0f}  "
          f"prédit=${case_summary['best_case']['predicted_price']:,.0f}  "
          f"erreur=${case_summary['best_case']['abs_error']:,.0f}")
    print(f"      Cas le moins bien prédit : réel=${case_summary['worst_case']['real_price']:,.0f}  "
          f"prédit=${case_summary['worst_case']['predicted_price']:,.0f}  "
          f"erreur=${case_summary['worst_case']['abs_error']:,.0f}")

    print("\n[7/7] Sauvegarde des modèles et du rapport ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_model, MODELS_DIR / "ridge.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "results": results,
        "best_model": best_name,
        "top_shap_features": top_features,
        "shap_cases": case_summary,
        "n_features": len(data["feature_cols"]),
        "train_rows": len(data["X_train_ridge"]),
        "test_rows": len(data["X_test_ridge"]),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Résumé final ===")
    for name, m in results.items():
        marker = " <== meilleur modèle" if name == best_name else ""
        print(
            f"{name:10s} | RMSE=${m['rmse']:,.0f}  MAE=${m['mae']:,.0f}  R²={m['r2']:.4f}  "
            f"CV RMSE=${m['cv_rmse_mean']:,.0f} (±${m['cv_rmse_std']:,.0f}){marker}"
        )
    print(f"\nModèles sauvegardés dans {MODELS_DIR}/ (non commités)")
    print(f"Figures sauvegardées dans {FIGURES_DIR}/")
    print(f"Rapport de métriques sauvegardé dans {METRICS_PATH}")


if __name__ == "__main__":
    run()
