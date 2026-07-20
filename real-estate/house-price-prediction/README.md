# House Price Prediction — Real Estate

## Contexte métier

L'estimation du prix de vente d'un bien immobilier à partir de ses caractéristiques
(surface, qualité de construction, année, quartier, etc.) est un problème central pour
les agences immobilières, les plateformes d'estimation en ligne et les acheteurs/vendeurs
souhaitant objectiver un prix. Ce projet construit un modèle de **régression** — cible
continue `SalePrice`, et non une classe — sur le dataset **House Prices - Advanced
Regression Techniques** (Kaggle), qui décrit **1460 maisons** à Ames, Iowa, à travers
**79 variables explicatives** (taille du terrain, qualité et état général, année de
construction et de rénovation, quartier, type de garage, présence de sous-sol, etc.).

Il s'agit du premier projet de **régression** du portfolio (les projets précédents —
[`credit-scoring-finance`](../../finance/credit-scoring-finance/),
[`fraud-detection-finance`](../../finance/fraud-detection-finance/),
[`kidney-disease-healthcare`](../../health/kidney-disease-healthcare/) — étaient tous des
problèmes de **classification**). Les métriques d'évaluation sont donc différentes : RMSE,
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
├── src/
│   ├── data_pipeline.py   # nettoyage, imputation, retrait outliers, log-target, split
│   └── train_model.py     # encodage, entraînement, évaluation, SHAP
├── models/             # modèles entraînés (.pkl, non versionnés)
├── notebooks/           # analyse exploratoire
├── reports/figures/    # scatter prédictions vs réel, SHAP summary/cas individuels
├── requirements.txt
└── data_description.txt   # description des 79 variables (référence Kaggle)
```

## Installation et lancement

```bash
pip install -r requirements.txt

# 1. Placer train.csv dans data/raw/
# 2. Nettoyage, imputation, retrait des outliers documentés, log-transform, split
python -m src.data_pipeline

# 3. Encodage, entraînement, évaluation et analyse SHAP
python -m src.train_model
```

## Méthodologie

**Diagnostic des valeurs manquantes** : ce dataset a la particularité de coder, pour de
nombreuses colonnes, une **absence de caractéristique** plutôt qu'une vraie donnée non
collectée — documenté explicitement dans `data_description.txt` via un code `NA` dédié
(ex. `PoolQC` vide = pas de piscine, `GarageType` vide = pas de garage, `BsmtQual` vide =
pas de sous-sol). 14 colonnes catégorielles sont dans ce cas (`PoolQC`, `MiscFeature`,
`Alley`, `Fence`, `FireplaceQu`, les 4 colonnes `Garage*`, les 5 colonnes `Bsmt*`) et sont
imputées par `"None"` ; `GarageYrBlt` (numérique, NaN sur exactement les mêmes 81 lignes
que les autres colonnes garage) est imputé par `0`. À l'inverse, 4 colonnes n'ont **aucun**
code `NA` documenté et sont donc traitées comme de vraies valeurs manquantes :
`LotFrontage` (imputée par la **médiane par quartier**, `groupby('Neighborhood')`, calculée
sur le train uniquement — plus pertinente qu'une médiane globale, la largeur de façade sur
rue variant fortement d'un quartier à l'autre), `MasVnrType`/`MasVnrArea` (`"None"`/`0`,
NaN signifiant très probablement l'absence de revêtement maçonné), et `Electrical` (1 seule
ligne, imputée par le mode du train).

**Outliers documentés** : les 2 maisons avec `GrLivArea` > 4000 pi² mais un prix
incohérent avec leur taille/qualité (Id 524 et 1299) sont retirées avant le split,
conformément à la recommandation officielle de l'auteur du dataset (De Cock, 2011) — ces
points, très éloignés de la relation surface/prix habituelle, ont une influence
disproportionnée sur une régression, linéaire en particulier.

**Anti-fuite de données** : le split train/test (80/20, aléatoire simple — pas de
stratification possible en régression) est effectué **avant** tout calcul de statistique
(médiane par quartier, mode `Electrical`, `StandardScaler`), qui sont ensuite calculées sur
le train uniquement et réappliquées au test.

**Cible transformée** : `SalePrice` est fortement asymétrique (skewness = 1.88 sur le
dataset complet). `SalePrice_log = log1p(SalePrice)` est utilisé comme cible
d'entraînement (skewness ramenée à 0.23 sur le train), et les prédictions sont reconverties
en dollars via `expm1` avant le calcul de toute métrique, pour rester interprétables dans
l'unité du problème.

**Encodage et scaling (`train_model.py`)** : One-Hot Encoding des variables catégorielles
(fit sur train uniquement, colonnes de test alignées sur celles du train), donnant 283
features. `StandardScaler` appliqué aux colonnes numériques uniquement, pour Ridge
seulement (inutile pour XGBoost, insensible à l'échelle des features).

## Résultats

Évaluation sur le test set, métriques reconverties en dollars (`expm1`) ; validation
croisée à 5 plis (`KFold`) sur le train, avec re-fit du `StandardScaler` à chaque pli pour
éviter toute fuite pendant la CV elle-même :

| Modèle | RMSE | MAE | R² | CV RMSE (moyenne ± écart-type) |
|---|---:|---:|---:|---:|
| **Ridge (alpha=10)** | **$20 048** | **$14 519** | **0.927** | **$20 222 (± $2 082)** |
| XGBoost | $21 736 | $15 604 | 0.915 | $24 543 (± $2 776) |

**Modèle retenu : Ridge**, pour deux raisons : il obtient le meilleur RMSE sur le test
($20 048 contre $21 736 pour XGBoost — RMSE choisi comme métrique de référence car elle est
dans la même unité que le prix et pénalise davantage les grosses erreurs), et il est
**plus stable en validation croisée** (écart-type $2 082 contre $2 776 pour XGBoost). Le
RMSE de $20 048 représente environ **11% du prix médian** (163 000$), un niveau d'erreur
raisonnable pour ce problème.

Scatter prédictions vs valeurs réelles (Ridge) :
[`reports/figures/pred_vs_actual_ridge.png`](reports/figures/pred_vs_actual_ridge.png).

## Interprétabilité (SHAP)

D'après le SHAP summary plot ([`reports/figures/shap_summary.png`](reports/figures/shap_summary.png))
calculé sur le modèle retenu (Ridge, `LinearExplainer`), les variables les plus
déterminantes sont, par ordre décroissant d'impact moyen :

1. `GrLivArea` (surface habitable hors sous-sol)
2. `OverallQual` (qualité globale des matériaux et finitions)
3. `YearBuilt` (année de construction)
4. `2ndFlrSF` (surface du 2e étage)
5. `1stFlrSF` (surface du rez-de-chaussée)

Ce classement est cohérent avec l'intuition immobilière : la surface habitable et la
qualité globale sont les deux premiers critères qu'un acheteur ou un évaluateur regarde,
et l'ancienneté du bien influence directement le prix. Deux cas individuels illustrent le
comportement du modèle :
[`reports/figures/shap_best_case.png`](reports/figures/shap_best_case.png) (maison la mieux
prédite : réel 176 000$, prédit 175 999$, erreur de 1$) et
[`reports/figures/shap_worst_case.png`](reports/figures/shap_worst_case.png) (maison la
moins bien prédite : réel 311 500$, prédit 237 001$, erreur de 74 499$).

## Limites et améliorations possibles

Le cas le moins bien prédit (écart d'environ 74 500$, soit près de 24% du prix réel) montre
que le modèle peut se tromper significativement sur certains biens, même avec un R² global
de 0.927. Ce type d'erreur est probablement dû à des facteurs qui influencent réellement le
prix de vente mais que le dataset ne capture pas : l'état intérieur réel du bien (au-delà
des notes `OverallQual`/`OverallCond`, qui restent des évaluations globales et non une
inspection détaillée pièce par pièce), le prestige précis ou la désirabilité d'un
sous-secteur au sein d'un même `Neighborhood`, et les conditions de négociation propres à
chaque vente (urgence du vendeur, enchère entre acheteurs, lien personnel, etc. — des
éléments par nature absents de toute base de données structurée). Une amélioration future
pourrait explorer des features d'interaction (ex. qualité × surface) ou un modèle non
linéaire mieux réglé (tuning des hyperparamètres XGBoost, actuellement fixés à des valeurs
raisonnables mais non optimisées), sans garantie de combler cet écart si l'information
manquante est simplement absente des données sources.
