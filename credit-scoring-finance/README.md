# Credit Scoring — Finance

## Contexte métier

Le credit scoring consiste à estimer la probabilité qu'un emprunteur connaisse un
défaut de paiement grave dans les deux prochaines années, afin d'éclairer les
décisions d'octroi de crédit. Un mauvais scoring coûte cher dans les deux sens :
refuser un bon emprunteur fait perdre du revenu, accepter un mauvais emprunteur
génère des pertes sur créances. Ce projet construit un modèle de classification
binaire sur le dataset **Give Me Some Credit** (compétition Kaggle), avec un
déséquilibre de classes marqué (~6,7% de défauts) qui impose des métriques
adaptées plutôt que la seule accuracy.

Dataset : [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit) —
`cs-training.csv`, cible `SeriousDlqin2yrs` (0 = pas de défaut grave, 1 = défaut
grave dans les 2 ans), et des variables telles que `RevolvingUtilizationOfUnsecuredLines`,
`age`, `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`, les compteurs
de retards de paiement (`NumberOfTime30-59DaysPastDueNotWorse`, etc.) et
`NumberOfDependents`.

## Structure du projet

```
credit-scoring-finance/
├── data/raw/          # cs-training.csv (non versionné, à placer ici)
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

# 1. cs-training.csv est déjà présent dans data/raw/
# 2. Nettoyage + feature engineering + split + scaling
python -m src.data_pipeline

# 3. Entraînement, évaluation, comparaison et analyse SHAP
python -m src.train_model
```

`src/data_pipeline.py` :
- supprime les doublons exacts et impute `MonthlyIncome` / `NumberOfDependents`
  (seules colonnes avec des valeurs manquantes) par la médiane ;
- plafonne (`clip`) les compteurs de retard de paiement au 99,9e percentile pour
  limiter l'effet des valeurs sentinelles extrêmes connues du dataset ;
- construit `DebtRatio_log` et `MonthlyIncome_log` (log1p) ainsi que
  `TotalPastDue`, somme des trois compteurs de retard ;
- applique un split stratifié `train/test` (80/20, `stratify=SeriousDlqin2yrs`,
  `random_state=42`) ;
- applique un `RobustScaler` sur les variables log-transformées (fit sur train,
  appliqué sur train et test) ;
- sauvegarde `data/processed/train.csv` et `data/processed/test.csv`.

`src/train_model.py` :
- entraîne un `RandomForestClassifier` (`class_weight='balanced'`) et un
  `XGBClassifier` (`scale_pos_weight` calculé à partir du ratio classes
  négatives/positives) ;
- évalue les deux modèles (accuracy, precision, recall, F1, AUC-PR, matrice de
  confusion sauvegardée en image) ;
- sélectionne le meilleur modèle selon l'AUC-PR (métrique adaptée au
  déséquilibre de classes, contrairement à l'accuracy) ;
- calcule une analyse SHAP (summary plot) sur le meilleur modèle ;
- sauvegarde les modèles dans `models/` et un résumé complet dans
  `reports/metrics.json`.

## Résultats

Les métriques détaillées (accuracy, precision, recall, F1, AUC-PR par modèle) et
les features SHAP les plus prédictives sont générées dans
[`reports/metrics.json`](reports/metrics.json) à chaque exécution de
`python -m src.train_model`.
