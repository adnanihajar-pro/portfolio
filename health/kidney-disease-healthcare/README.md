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
├── data/processed/    # jeux de données nettoyés générés par le pipeline
├── src/                # pipeline de nettoyage, feature engineering, entraînement
├── models/             # modèles entraînés (.pkl, non versionnés)
├── notebooks/           # analyse exploratoire, diagnostic des données
├── reports/figures/    # visualisations, matrices de confusion, SHAP
└── requirements.txt
```

## Installation et lancement

```bash
pip install -r requirements.txt

# Placer kidney_disease.csv dans data/raw/
```

## Points d'attention identifiés

- **Taille du dataset** : seulement 400 patients pour 25 variables, ce qui impose une
  validation prudente (validation croisée, régularisation) plutôt qu'un simple split
  train/test.
- **Valeurs manquantes** : plusieurs colonnes cliniques présentent jusqu'à 38% de
  valeurs manquantes, à traiter avec une stratégie d'imputation adaptée plutôt qu'une
  suppression de lignes qui réduirait encore l'échantillon.
- **Colonnes mal typées** : `pcv`, `wc` et `rc` sont stockées en texte à cause de
  valeurs `'\t?'` non reconnues comme `NaN` par pandas, à corriger avant toute
  modélisation.

## Statut

Projet en cours. Le diagnostic complet des données (structure, doublons, valeurs
manquantes, incohérences de texte, cohérence médicale) est la prochaine étape, suivi du
nettoyage, du feature engineering, de l'entraînement et de l'évaluation des modèles.
