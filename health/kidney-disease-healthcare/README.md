# Chronic Kidney Disease — Santé

## Contexte métier

La maladie rénale chronique (Chronic Kidney Disease, CKD) est une perte progressive et
souvent silencieuse de la fonction rénale, détectée tardivement dans de nombreux cas
faute de dépistage. Un modèle capable d'identifier les patients à risque à partir de
mesures cliniques courantes (tension, analyses sanguines et urinaires) pourrait appuyer
un dépistage plus précoce. Ce projet construit un modèle de classification binaire sur
le dataset **Chronic Kidney Disease** (Kaggle, `mansoordaku/ckdisease`), avec deux
contraintes fortes qui structurent toute la méthodologie : un échantillon très restreint
de **400 patients** (risque élevé de surapprentissage) et un taux de valeurs manquantes
important sur plusieurs variables (jusqu'à 38%).

Dataset : [Chronic Kidney Disease (Kaggle)](https://www.kaggle.com/datasets/mansoordaku/ckdisease) —
`kidney_disease.csv`, 25 variables cliniques, cible `classification` (`ckd` / `notckd`).

## Structure du projet

```
kidney-disease-healthcare/
├── data/raw/          # kidney_disease.csv (non versionné, à placer ici)
├── data/processed/    # train.csv / test.csv générés par le pipeline
├── src/
│   ├── data_pipeline.py   # nettoyage, indicateurs, imputation, split
│   └── train_model.py     # entraînement, évaluation, SHAP
├── models/             # modèles entraînés (.pkl, non versionnés)
├── notebooks/           # exploration, diagnostic, comparaison des méthodes d'imputation
├── reports/figures/    # matrices de confusion, SHAP summary/cas individuels
└── requirements.txt
```

## Installation et lancement

```bash
pip install -r requirements.txt

# 1. Placer kidney_disease.csv dans data/raw/
# 2. Nettoyage, indicateurs, imputation médiane/mode, split train/test
python -m src.data_pipeline

# 3. Entraînement, évaluation (cross-validation + test set) et analyse SHAP
python -m src.train_model
```

## Méthodologie

**Nettoyage (`src/data_pipeline.py`)** : `pcv`, `wc`, `rc` sont converties en numérique
(`'\t?'` → `NaN`) ; les incohérences de texte sur `dm`, `cad` et `classification`
(espaces/tabs résiduels comme `'\tyes'` ou `'ckd\t'`) sont normalisées. Les valeurs
médicalement impossibles (`sc>20` mg/dL, `sod<100` mEq/L, `pot>15` mEq/L — incompatibles
avec la vie, donc traitées comme des erreurs de saisie) sont mises en `NaN` plutôt que
supprimées, avec un indicateur dédié `valeur_medicale_aberrante`. Le split train/test
(80/20, stratifié) est effectué **avant** tout calcul statistique ; un indicateur
`{colonne}_missing` est créé pour chacune des 24 variables avant imputation par la
médiane (numérique) / le mode (catégorielle), calculés sur le train uniquement et
réappliqués tels quels au test.

**Choix de la méthode d'imputation** : avant d'écrire le pipeline final, deux méthodes
(médiane/mode vs `KNNImputer`) ont été comparées par validation croisée dans
[`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb). Les deux atteignent
une AUC-PR quasi parfaite (1.0000 vs 0.9996) — la médiane/mode a été retenue car
au moins aussi performante et nettement plus simple sur un dataset de 400 lignes.

**Score quasi parfait** : les modèles atteignent une AUC-PR ≈ 1.0, aussi bien en
validation croisée que sur le test set. Ce n'est **pas une fuite de données** : vérifié
explicitement (colonne `id` absente des features, indépendance à l'ordre du CSV,
répartition des coefficients de la régression logistique sur ~8 variables sans
dominante isolée — cf. `notebooks/01_exploration.ipynb`). C'est une caractéristique
réelle de ce dataset : plusieurs variables (`sg`, `al`, `hemo`, `pcv`) sont des critères
diagnostiques cliniques directs de la CKD, ce qui rend la séparation quasi triviale pour
un modèle linéaire.

## Résultats

Validation croisée (`StratifiedKFold`, 5 folds, sur le train) :

| Modèle | AUC-PR moyen | Écart-type |
|---|---:|---:|
| **Logistic Regression** | **1.0000** | 0.0000 |
| XGBoost (max_depth=3, n_estimators=50) | 0.9993 | 0.0009 |

Évaluation finale sur le test set (80 patients, jamais vus pendant l'entraînement) :

| Modèle | Precision | Recall | F1 | AUC-PR |
|---|---:|---:|---:|---:|
| **Logistic Regression** | 1.0000 | 0.9800 | 0.9899 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Modèle retenu : Logistic Regression**, la mieux classée en validation croisée (moyenne
la plus haute, écart-type nul). Elle ne manque qu'un seul patient CKD sur 50 dans le
test set, contre 0 pour XGBoost sur ce test précis — un écart non significatif sur 80
lignes, insuffisant pour préférer un modèle plus complexe et moins interprétable.

Matrices de confusion :
[`reports/figures/confusion_matrix_logistic_regression.png`](reports/figures/confusion_matrix_logistic_regression.png),
[`reports/figures/confusion_matrix_xgboost.png`](reports/figures/confusion_matrix_xgboost.png).

## Interprétabilité (SHAP)

D'après le SHAP summary plot
([`reports/figures/shap_summary.png`](reports/figures/shap_summary.png)) calculé sur le
modèle retenu, les variables les plus déterminantes sont : `hemo` (hémoglobine), `sg`
(densité urinaire), `pcv` (hématocrite), `al` (albuminurie) et `rbc_missing`.

**Insight notable — `rbc_missing`** : l'indicateur de valeur manquante sur `rbc`
(globules rouges dans les urines) apparaît en 5ᵉ position, avant plusieurs variables
numériques directement mesurées. Cette absence n'est vraisemblablement **pas aléatoire** :
`rbc` est l'une des variables les plus souvent manquantes du dataset (38%), et il est
plausible qu'un médecin ne prescrive cet examen d'urine que lorsqu'un premier signal
clinique l'y incite déjà — auquel cas le simple fait que le test ait été demandé (ou pas)
reflète indirectement un jugement clinique précoce, antérieur au résultat du test
lui-même. Si cette hypothèse se confirme sur un dataset plus large, `rbc_missing`
constituerait un signal prédictif à part entière plutôt qu'un artefact de saisie —
cohérent avec la décision de le conserver comme indicateur plutôt que de simplement
imputer et masquer l'information.

## Limites et améliorations possibles

Le dataset est petit (400 patients, 80 en test) et les scores quasi parfaits laissent peu
de marge pour distinguer objectivement les modèles entre eux ou pour détecter un éventuel
surapprentissage résiduel — une évaluation sur un second dataset CKD indépendant serait
nécessaire avant tout usage au-delà du cadre pédagogique de ce projet. L'hypothèse sur
`rbc_missing` (jugement clinique implicite) n'a pas été testée formellement (pas
d'information sur l'ordre chronologique des examens dans ce dataset) et reste une piste
d'investigation plutôt qu'une conclusion validée.
