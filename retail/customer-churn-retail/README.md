# Customer Churn Prediction — Retail / Télécom

## Contexte métier

La rétention client est un enjeu majeur pour les opérateurs télécoms et les entreprises
de retail par abonnement : acquérir un nouveau client coûte significativement plus cher
que d'en conserver un existant. Ce projet construit un modèle de **classification
binaire** — cible `Churn` (le client résilie son abonnement ou non) — permettant
d'identifier en amont les clients à risque de résiliation à partir de leurs
caractéristiques (données démographiques, services souscrits, type de contrat,
facturation).

Dataset : [Telco Customer Churn (Kaggle, `blastchar/telco-customer-churn`)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) —
**7043 clients**, **21 colonnes** (données démographiques, services téléphoniques et
internet souscrits, type et durée de contrat, mode de facturation, charges mensuelles et
totales, et la cible `Churn`), avec un **déséquilibre de classes modéré** (~26.5% de
clients ayant résilié).

## Structure du projet

```
customer-churn-retail/
├── data/raw/          # WA_Fn-UseC_-Telco-Customer-Churn.csv
├── data/processed/    # train.csv / test.csv générés par le pipeline
├── src/                # nettoyage, encodage, entraînement, évaluation
├── models/             # modèles entraînés (.pkl, non versionnés)
├── notebooks/           # analyse exploratoire
├── reports/figures/    # matrices de confusion, SHAP summary/cas individuels
└── requirements.txt
```

## Installation et lancement

```bash
pip install -r requirements.txt
```

Le CSV brut est déjà présent dans `data/raw/`. Le pipeline de nettoyage/encodage et
l'entraînement des modèles seront ajoutés dans `src/` dans une prochaine étape.

## Statut

Structure initiale du projet — dataset brut en place, pipeline de données et
entraînement des modèles à venir.
