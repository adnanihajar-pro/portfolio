# Customer Churn Prediction — Retail / Télécom

## Contexte métier

La rétention client est un enjeu majeur pour les opérateurs télécoms : acquérir un
nouveau client coûte significativement plus cher que d'en conserver un existant. Ce
projet construit un modèle de **classification binaire** — cible `Churn` — permettant
d'identifier en amont les clients à risque de résiliation à partir de leurs
caractéristiques (données démographiques, services souscrits, type de contrat,
facturation).

Dataset : [Telco Customer Churn (Kaggle, `blastchar/telco-customer-churn`)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) —
**7043 clients** à l'origine, **7021 après suppression des doublons stricts** (voir
Méthodologie), **19 variables explicatives** (données démographiques, services
téléphoniques et internet souscrits, type et durée de contrat, mode de facturation,
charges mensuelles et totales), avec un **déséquilibre de classes modéré** (~26.5% de
clients ayant résilié).

## Structure du projet

```
customer-churn-retail/
├── data/raw/           # WA_Fn-UseC_-Telco-Customer-Churn.csv (non versionné)
├── data/processed/     # train.csv / test.csv générés par le pipeline
├── src/
│   ├── data_pipeline.py   # nettoyage, indicateurs, split
│   └── train_model.py     # encodage, entraînement, évaluation, SHAP
├── models/              # modèle retenu (.pkl, non versionné)
├── notebooks/            # analyse exploratoire
├── reports/figures/     # matrices de confusion, SHAP summary/cas individuels
├── reports/metrics.json # métriques + résultats SHAP
└── requirements.txt
```

## Installation et lancement

```bash
pip install -r requirements.txt

# 1. Placer WA_Fn-UseC_-Telco-Customer-Churn.csv dans data/raw/
# 2. Nettoyage, indicateurs, suppression des doublons, split
python -m src.data_pipeline

# 3. Entraînement, évaluation et analyse SHAP
python -m src.train_model
```

## Méthodologie

**Nettoyage (`data_pipeline.py`)** :

- `TotalCharges` était stockée en texte (`object`) ; convertie en numérique
  (`pd.to_numeric`), ce qui fait apparaître 11 valeurs non convertibles — toutes
  correspondant exactement à des clients `tenure=0` (nouveaux clients sans encore de
  facture cumulée). Remplacées par `0`, une règle métier plutôt qu'une imputation
  statistique.
- **22 lignes dupliquées** (identiques sur les 20 colonnes hors `customerID`, formant 20
  groupes : 18 paires + 2 triplets) supprimées **avant le split**, pour qu'un même profil
  ne se retrouve pas à la fois en train et en test. La correspondance exacte constatée y
  compris sur les variables continues (`tenure`, `MonthlyCharges`, `TotalCharges`) rendait
  une simple coïncidence statistique très improbable.
- Catégories redondantes fusionnées en `"No"` : `"No internet service"` et `"No phone
  service"` sur les 7 colonnes concernées (`MultipleLines`, `OnlineSecurity`,
  `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) ne
  sont pas de vraies 3e catégories informatives, seulement des doublons fonctionnels de
  `"No"` liés à l'absence du service parent.
- Indicateur `charges_incoherentes` ajouté (écart relatif > 10% entre `TotalCharges` et
  `tenure × MonthlyCharges`), sans suppression ni modification des valeurs sous-jacentes.
- `customerID` exclu des features (identifiant unique par ligne, aucune valeur
  prédictive).
- Split 80/20 stratifié sur `Churn`, effectué **avant** toute statistique.

**Choix des modèles (`train_model.py`)** : ce dataset se distingue des projets précédents
du portfolio ([`credit-scoring-finance`](../../finance/credit-scoring-finance/),
[`fraud-detection-finance`](../../finance/fraud-detection-finance/)) par sa composition —
**16 des 19 variables sont catégorielles** (15 colonnes `object` + `SeniorCitizen`,
binaire). Plutôt que Logistic Regression + XGBoost, ce projet retient **Random Forest**
(`class_weight='balanced'`, One-Hot Encoding classique) et **CatBoost**
(`auto_class_weights='Balanced'`), qui gère les colonnes catégorielles **nativement** via
`cat_features` sans les faire passer par un One-Hot Encoding — évitant la dimensionnalité
éparse (20 → 40 colonnes avec OHE) et la perte d'information associée sur un dataset
majoritairement catégoriel.

## Résultats

Évaluation sur le test set (1405 clients) :

| Modèle | Precision | Recall | F1 | AUC-PR |
|---|---:|---:|---:|---:|
| Random Forest (OHE) | 0.5542 | 0.6048 | 0.5784 | 0.5877 |
| **CatBoost (natif)** | **0.5474** | **0.7608** | **0.6367** | **0.6576** |

**Modèle retenu : CatBoost**, meilleur AUC-PR (métrique de référence, cohérente avec le
déséquilibre modéré de classes) et surtout meilleur recall — 76% des clients qui
résilient réellement sont détectés, contre 60% pour Random Forest. La gestion native des
variables catégorielles, majoritaires dans ce dataset, explique cet écart.

Matrices de confusion :
[`reports/figures/confusion_matrix_random_forest.png`](reports/figures/confusion_matrix_random_forest.png),
[`reports/figures/confusion_matrix_catboost.png`](reports/figures/confusion_matrix_catboost.png).

## Interprétabilité (SHAP)

D'après le SHAP summary plot ([`reports/figures/shap_summary.png`](reports/figures/shap_summary.png))
calculé sur le modèle retenu (CatBoost, `TreeExplainer`), les variables les plus
déterminantes sont, par ordre décroissant d'impact moyen :

1. `Contract` (type de contrat)
2. `InternetService` (type de service internet)
3. `tenure` (ancienneté du client)
4. `TotalCharges` (charges cumulées)
5. `PaymentMethod` (mode de paiement)

Ce classement est cohérent avec l'intuition métier : un **contrat mensuel combiné à une
faible ancienneté** est le profil le plus caractéristique d'un client à risque, ce
qu'illustrent les deux cas individuels :

- [`reports/figures/shap_case_1_high_risk.png`](reports/figures/shap_case_1_high_risk.png)
  (client à haut risque bien identifié, probabilité de churn = 0.96) : nouveau client
  (`tenure=1`), fibre optique, contrat mensuel — chaque facteur pousse fortement vers le
  churn.
- [`reports/figures/shap_case_2_borderline.png`](reports/figures/shap_case_2_borderline.png)
  (cas limite, probabilité de churn = 0.50) : l'absence de service internet
  (`InternetService=No`) pousse fortement vers "pas de churn" (-1.06), mais la faible
  ancienneté et le contrat mensuel poussent vers le churn (+1.03, +0.78) — les deux forces
  s'annulent presque exactement, illustrant un cas où le modèle est réellement incertain.

## Limites et améliorations possibles

La precision du modèle retenu reste modérée (~55%) : environ **une alerte sur deux est
une fausse alerte** (un client identifié à risque qui ne résilie finalement pas). Ce
niveau de precision doit être mis en perspective avec le coût métier respectif des deux
types d'erreur — contacter à tort un client qui n'allait pas résilier (ex. offre de
rétention, appel commercial) coûte généralement bien moins cher que perdre un client sans
avoir tenté d'intervenir, ce qui justifie de privilégier le recall (capturer un maximum de
churners réels) au prix d'un nombre plus élevé de fausses alertes. Une amélioration future
pourrait explorer un ajustement du seuil de décision (actuellement 0.5) en fonction du
coût réel d'une action de rétention, ou un tuning des hyperparamètres CatBoost
(actuellement à leurs valeurs par défaut hormis `auto_class_weights`).
