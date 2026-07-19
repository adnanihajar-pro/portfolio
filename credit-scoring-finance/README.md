# Credit Scoring — Finance

## Contexte métier

Le credit scoring consiste à estimer, au moment de l'octroi d'un crédit, la probabilité
qu'un emprunteur connaisse un défaut de paiement grave (90 jours de retard ou plus) dans
les deux années suivantes. Une mauvaise estimation coûte cher dans les deux sens :
refuser un bon emprunteur fait perdre du revenu à la banque, accepter un mauvais
emprunteur génère des pertes sur créances. Ce projet construit un modèle de
classification binaire sur le dataset **Give Me Some Credit** (Kaggle, ~150 000
emprunteurs), avec un déséquilibre de classes marqué (~6,7% de défauts) qui impose des
métriques et une gestion adaptées plutôt qu'une approche de classification standard.

Dataset : [Give Me Some Credit (Kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit) —
`cs-training.csv`, cible `SeriousDlqin2yrs` (0 = pas de défaut grave, 1 = défaut grave
dans les 2 ans).

## Structure du projet

```
credit-scoring-finance/
├── data/raw/          # cs-training.csv (non versionné, à placer ici)
├── data/processed/    # train.csv / test.csv générés par le pipeline
├── src/
│   ├── data_pipeline.py   # nettoyage, feature engineering, split, scaling
│   └── train_model.py     # sélection de features, entraînement, évaluation, SHAP
├── models/             # modèles entraînés (.pkl, non versionnés)
├── notebooks/           # analyse exploratoire
├── reports/figures/    # matrices de confusion, SHAP summary/cas individuels
└── requirements.txt
```

## Installation et lancement

```bash
pip install -r requirements.txt

# 1. Placer cs-training.csv dans data/raw/
# 2. Nettoyage + feature engineering + split + scaling
python -m src.data_pipeline

# 3. Sélection de features, entraînement, évaluation et analyse SHAP
python -m src.train_model
```

## Méthodologie

**Nettoyage (`src/data_pipeline.py`)** : les doublons exacts sont supprimés avant le
split (pour éviter qu'une même ligne se retrouve à la fois en train et en test). Les
valeurs manquantes (`MonthlyIncome`, `NumberOfDependents`) et les valeurs aberrantes
connues du dataset (utilisation de crédit > 100%, codes de retard sentinelles 96/98,
`DebtRatio` extrême) sont **imputées/plafonnées plutôt que supprimées**, chacune
accompagnée d'un indicateur binaire dédié (`MonthlyIncome_missing`,
`utilisation_aberrante`, `retard_code_suspect`, `debtratio_aberrant`, etc.) afin de
préserver le signal que représente le fait même qu'une valeur soit manquante ou
aberrante. Toutes les statistiques (médianes, scaler) sont calculées sur le train
uniquement et réappliquées au test, pour éviter toute fuite de données.

**Sélection de features (`src/train_model.py`)** : parmi toutes les colonnes
disponibles, seules **16 variables** alignées sur les critères bancaires classiques
d'octroi de crédit (les "5 C du crédit", en particulier *Capacity* — capacité de
remboursement — et *Character* — historique de paiement) sont retenues comme features
de modélisation : utilisation du crédit renouvelable, historique de retards de paiement,
âge, revenu, taux d'endettement, nombre de lignes de crédit ouvertes, et leurs
indicateurs associés. `DebtRatio` est explicitement exclue au profit de `DebtRatio_log`
(sa transformation logarithmique), les deux étant redondantes et leur coexistence
faussant l'interprétation SHAP. Les variables restantes non alignées sur ces critères
(`NumberRealEstateLoansOrLines`, `NumberOfDependents`) sont écartées.

**Gestion du déséquilibre** : la classe positive (défaut) ne représente que ~6,7% des
emprunteurs. Les deux modèles intègrent une pondération de classe (`class_weight
='balanced'` pour la régression logistique, `scale_pos_weight` calculé dynamiquement
sur le ratio classes négatives/positives pour XGBoost) plutôt que de traiter les deux
classes à égalité.

## Résultats

| Modèle | Precision | Recall | F1 | AUC-PR |
|---|---:|---:|---:|---:|
| **Logistic Regression** | 0.2193 | 0.7507 | 0.3394 | **0.3879** |
| XGBoost | 0.2458 | 0.6179 | 0.3517 | 0.3318 |

**Modèle retenu : Logistic Regression**, pour deux raisons : elle obtient la meilleure
AUC-PR (0.3879 contre 0.3318 pour XGBoost, un écart significatif sur cette métrique de
référence en contexte fortement déséquilibré), et elle offre des coefficients
directement interprétables — un atout important dans un contexte réglementé comme le
credit scoring, où la justification d'une décision de refus de crédit est souvent
requise.

Matrices de confusion :
[`reports/figures/confusion_matrix_logistic_regression.png`](reports/figures/confusion_matrix_logistic_regression.png),
[`reports/figures/confusion_matrix_xgboost.png`](reports/figures/confusion_matrix_xgboost.png).

## Interprétabilité (SHAP)

D'après le SHAP summary plot
([`reports/figures/shap_summary.png`](reports/figures/shap_summary.png)) calculé sur le
modèle retenu (Logistic Regression), les 5 variables les plus déterminantes sont :

1. `RevolvingUtilizationOfUnsecuredLines`
2. `age`
3. `TotalPastDue`
4. `MonthlyIncome_missing`
5. `DebtRatio_log`

Ce classement est cohérent avec la logique bancaire : l'utilisation du crédit
renouvelable et le cumul des retards de paiement passés sont des indicateurs directs de
la *Capacity*/*Character* de l'emprunteur, l'âge est un facteur de risque reconnu
(les emprunteurs plus jeunes présentant historiquement un risque de défaut plus élevé),
et le fait même que le revenu mensuel soit manquant (`MonthlyIncome_missing`) s'avère
être un signal prédictif à part entière, ce qui justifie a posteriori le choix de le
conserver comme indicateur plutôt que de simplement supprimer les lignes concernées.

## Limites et améliorations possibles

Le seuil de décision utilisé pour convertir les probabilités prédites en classe
(0,5 par défaut) n'a pas été optimisé : il traite implicitement un faux négatif (accorder
un crédit à un emprunteur qui fera défaut) et un faux positif (refuser un crédit à un bon
emprunteur) comme équivalents en coût, ce qui n'est généralement pas le cas pour une
banque. Une amélioration future consisterait à ajuster ce seuil à partir d'une analyse de
coût métier explicite (coût moyen d'une perte sur créance vs manque à gagner d'un refus),
plutôt que de conserver la valeur par défaut — piste non encore implémentée dans ce
projet.
