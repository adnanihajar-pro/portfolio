# Fraud Detection — Finance

## Contexte métier

La détection de fraude par carte bancaire est un enjeu critique pour les institutions
financières : chaque transaction frauduleuse non détectée représente une perte directe,
tandis que trop de faux positifs dégrade l'expérience client et sature les équipes
antifraude. Ce projet construit un modèle de classification capable d'identifier les
transactions frauduleuses parmi des transactions européennes, avec un défi central : la
fraude ne représente qu'**environ 0,17% des transactions**, ce qui impose une gestion
explicite du déséquilibre de classes (pondération, métriques adaptées) plutôt qu'une
approche de classification standard.

Dataset : [Credit Card Fraud Detection (Kaggle, mlg-ulb)](https://www.kaggle.com/mlg-ulb/creditcardfraud) —
`V1`-`V28` anonymisées par PCA, `Time` (secondes écoulées), `Amount` (montant),
`Class` (0 = légitime, 1 = fraude).

## Structure du projet

```
fraud-detection-finance/
├── data/raw/          # creditcard.csv (non versionné, à placer ici)
├── data/processed/    # train.csv / test.csv générés par le pipeline (non versionnés)
├── src/
│   ├── data_pipeline.py   # nettoyage, feature engineering, split, scaling
│   └── train_model.py     # entraînement, évaluation, comparaison, SHAP
├── models/             # modèles entraînés (.pkl, non versionnés)
├── notebooks/
│   └── 01_exploration.ipynb   # analyse exploratoire
├── reports/figures/    # matrices de confusion, SHAP summary/cas individuels
└── requirements.txt
```

## Installation et lancement

```bash
pip install -r requirements.txt

# 1. Placer creditcard.csv dans data/raw/
# 2. Nettoyage + feature engineering + split + scaling
python -m src.data_pipeline

# 3. Entraînement, évaluation, comparaison et analyse SHAP
python -m src.train_model
```

`src/data_pipeline.py` :
- supprime les doublons exacts et impute les valeurs manquantes par la médiane (aucune
  valeur manquante détectée sur ce dataset, mais la logique est en place) ;
- construit des features cycliques `Hour_sin` / `Hour_cos` à partir de `Time`, et
  `Amount_log = log1p(Amount)` ;
- applique un split stratifié `train/test` (80/20, `stratify=Class`, `random_state=42`) ;
- applique un `RobustScaler` sur `Amount` et `Amount_log` uniquement (fit sur train,
  appliqué sur train et test) ;
- sauvegarde `data/processed/train.csv` et `data/processed/test.csv`.

`src/train_model.py` :
- entraîne un `RandomForestClassifier` (`class_weight='balanced'`) et un `XGBClassifier`
  (`scale_pos_weight` calculé à partir du ratio classes négatives/positives) ;
- évalue les deux modèles (accuracy, precision, recall, F1, AUC-PR, matrice de
  confusion sauvegardée en image) ;
- sélectionne le meilleur modèle selon l'AUC-PR (métrique adaptée au déséquilibre
  extrême, contrairement à l'accuracy) ;
- calcule une analyse SHAP (summary plot + cas individuels) sur le meilleur modèle ;
- sauvegarde les modèles dans `models/` et un résumé complet dans `reports/metrics.json`.

## Résultats obtenus

Sur 284 807 transactions brutes (1 081 doublons supprimés → 283 726 lignes), split
stratifié 80/20 (226 980 lignes train / 56 746 lignes test, ~0,17% de fraude conservé
dans les deux jeux) :

| Modèle        | Accuracy | Precision | Recall | F1     | AUC-PR |
|---------------|---------:|----------:|-------:|-------:|-------:|
| Random Forest |   0.9995 |    0.9589 | 0.7368 | 0.8333 | 0.8095 |
| **XGBoost**   |   0.9995 |    0.9241 | 0.7684 | 0.8391 | **0.8252** |

**Meilleur modèle : XGBoost** (retenu sur l'AUC-PR, la métrique la plus fiable ici vu
le déséquilibre extrême des classes — l'accuracy est proche de 100% pour les deux
modèles et n'est pas discriminante).

Matrices de confusion : [`reports/figures/confusion_matrix_random_forest.png`](reports/figures/confusion_matrix_random_forest.png),
[`reports/figures/confusion_matrix_xgboost.png`](reports/figures/confusion_matrix_xgboost.png).

### Variables les plus prédictives (analyse SHAP sur XGBoost)

D'après le SHAP summary plot ([`reports/figures/shap_summary.png`](reports/figures/shap_summary.png))
et l'explication d'un cas individuel de fraude ([`reports/figures/shap_case_1.png`](reports/figures/shap_case_1.png)),
les variables les plus déterminantes pour la prédiction sont, par ordre décroissant
d'impact moyen :

1. `V14`
2. `V4`
3. `V12`
4. `V11`
5. `V10`
6. `V3`
7. `V8`
8. `V26`
9. `V19`
10. `Amount`

Ces variables (issues de la transformation PCA, à l'exception de `Amount`) sont
cohérentes avec les corrélations observées dès l'étape d'exploration (`V14`, `V12`,
`V10`, `V4`, `V11` figuraient déjà parmi les corrélations les plus fortes avec `Class`).

## Détail complet des métriques

Le détail complet (métriques par modèle, top features SHAP, tailles des jeux de
données) est disponible dans [`reports/metrics.json`](reports/metrics.json), régénéré
à chaque exécution de `python -m src.train_model`.
