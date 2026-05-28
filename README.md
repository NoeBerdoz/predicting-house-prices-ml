# Inved Corp — Predicting House Prices

Académie HES-SO ARC, Master en science des données (64-61.1).
Équipe : Noé Berdoz, Cyrille Dos Ghali, Steeve Leuba.

Roleplay : société de conseil immobilier fictive **Inved Corp**. Modèle de régression supervisée pour prédire le prix de vente d'un bien résidentiel à partir de 79 caractéristiques (jeu de données Kaggle "House Prices — Advanced Regression Techniques", Ames Iowa).

## Structure (CRISP-DM)

| Notebook                          | Phase(s) CRISP   | Rôle                                                        |
|-----------------------------------|------------------|------------------------------------------------------------|
| `1_ideation_phase.ipynb`          | Phase 1 + 2      | Business understanding + EDA                                |
| `2_data_prep.ipynb`               | Phase 3          | Préparation canonique : 3 variantes de préprocesseur + helpers partagés |
| `3a_scaled_models.ipynb`          | Phase 4          | Modèles sensibles à l'échelle (OLS, Ridge, Lasso, ElasticNet, KNN, MLP) |
| `3b_sklearn_trees.ipynb`          | Phase 4          | Arbres sklearn (DecisionTree, RandomForest, GradientBoosting, AdaBoost) |
| `3c_native_boosting.ipynb`        | Phase 4          | Boosting moderne (XGBoost natif vs OneHot, Optuna, SHAP)    |
| `3d_stacking.ipynb`               | Phase 4          | Stacking à préparations mixtes (Lasso + GBR + XGB → RidgeCV) |
| `4_evaluation.ipynb`              | Phase 5          | Comparaison inter-familles + dissertation + désignation du champion |
| `5_Assessment_Blueprint.ipynb`    | Phase 6          | Synthèse blueprint (rapport direction) + soumission Kaggle  |

> **Ordre d'exécution.** `2_data_prep.ipynb` est la source canonique de la préparation. Chaque notebook de modélisation / évaluation le ré-exécute automatiquement via `%run 2_data_prep.ipynb` en première cellule — il n'y a donc pas besoin de le lancer à la main au préalable, mais il doit rester exécutable. `4_evaluation.ipynb` lit les `results/family_*.json` produits par `3a`–`3d` ; `5_Assessment_Blueprint.ipynb` lit `results/winning_model.json` produit par `4_evaluation.ipynb`.

Documentation détaillée du modèle dans `docs/MLCanvas_v1.2.pdf` (canvas OWNML rempli).

## Setup environnement

Python **3.13** requis (les versions pinnées dans `requirements.txt` le ciblent).

```bash
# 1. Créer le virtual env (depuis la racine du repo)
python3.13 -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate             # Windows (PowerShell / cmd)

# 2. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt    # outils Jupyter

# 3. Enregistrer le kernel Jupyter
python -m ipykernel install --user --name inved-corp \
    --display-name "Python (Inved Corp .venv)"
```

Dans Jupyter / PyCharm, sélectionner le kernel **"Python (Inved Corp .venv)"** lors de l'ouverture des notebooks.

> Si vous n'avez pas Python 3.13 nativement, vous pouvez bootstrap depuis un environnement conda existant :
> ```bash
> /chemin/vers/python3.13 -m venv .venv
> ```

## Lancer Jupyter

```bash
source .venv/bin/activate            # Linux / macOS  (Windows : .venv\Scripts\activate)
jupyter lab    # ou : jupyter notebook
```

## Données

Le dossier `data/` contient les CSV Kaggle d'origine (`train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt`). Ils sont versionnés dans le repo.

## Soumission Kaggle

La soumission est générée par `5_Assessment_Blueprint.ipynb` (section 6.4), qui réentraîne le champion sur 100 % des données puis produit un fichier `submission.csv` à la racine du repo. Ce fichier est git-ignoré — il faut l'uploader manuellement sur Kaggle.
