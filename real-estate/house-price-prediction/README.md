# House Price Prediction — Real Estate

## Contexte métier

L'estimation du prix de vente d'un bien immobilier à partir de ses caractéristiques
(surface, qualité de construction, année, quartier, etc.) est un problème central pour
les agences immobilières, les plateformes d'estimation en ligne et les acheteurs/vendeurs
souhaitant objectiver un prix. Ce projet construit un modèle de **régression** — cible
continue (`SalePrice`), et non une classe — sur le dataset **House Prices - Advanced
Regression Techniques** (Kaggle), qui décrit **1460 maisons** à Ames, Iowa, à travers
**79 variables explicatives** (taille du terrain, qualité et état général, année de
construction et de rénovation, quartier, type de garage, présence de sous-sol, etc.).

Il s'agit du premier projet de régression du portfolio (les projets précédents —
[`credit-scoring-finance`](../../finance/credit-scoring-finance/),
[`fraud-detection-finance`](../../finance/fraud-detection-finance/),
[`kidney-disease-healthcare`](../../health/kidney-disease-healthcare/) — étaient tous des
problèmes de classification). Les métriques d'évaluation seront donc différentes : RMSE,
MAE et R² plutôt que Precision/Recall/F1/AUC.

Dataset : [House Prices - Advanced Regression Techniques (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) —
`train.csv` (1460 lignes, cible `SalePrice`), `test.csv` (sans la cible, réservé à la
soumission Kaggle et non utilisé pour l'évaluation de ce projet), `data_description.txt`
(description détaillée des 79 variables).

## Structure du projet

```
house-price-prediction/
├── data/raw/          # train.csv (non versionné, à placer ici)
├── data/processed/    # train.csv / test.csv générés par le pipeline
├── src/                # nettoyage, feature engineering, entraînement (à venir)
├── models/             # modèles entraînés (.pkl, non versionnés)
├── notebooks/           # analyse exploratoire (à venir)
├── reports/figures/    # visualisations et diagnostics (à venir)
└── requirements.txt
```

## Statut

Squelette initial du projet : arborescence, dépendances et documentation de base. Le
pipeline de nettoyage/feature engineering (`src/data_pipeline.py`), l'entraînement des
modèles (`src/train_model.py`) et l'analyse exploratoire (`notebooks/`) seront ajoutés
dans une étape ultérieure, en commençant par un diagnostic des données.

## Installation

```bash
pip install -r requirements.txt

# Placer train.csv dans data/raw/ (déjà fait dans ce dépôt)
```
