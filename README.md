# Inved Corp — Predicting House Prices

Académie HES-SO ARC, Master en science des données (64-61.1).
Équipe : Noé Berdoz, Cyrille Dos Ghali, Steeve Leuba.

Roleplay : société de conseil immobilier fictive **Inved Corp**. Modèle de régression supervisée pour prédire le prix de vente d'un bien résidentiel à partir de 79 caractéristiques (jeu de données Kaggle "House Prices — Advanced Regression Techniques", Ames Iowa).

## Structure (CRISP-DM)

| Notebook                          | Phases CRISP                                   |
|-----------------------------------|------------------------------------------------|
| `1_ideation_phase.ipynb`          | Phase 1 (Business) + Phase 2 (EDA)             |
| `2_Design_phase.ipynb`            | Phase 3 (Prep) + Phase 4 (Modélisation) + Phase 5 (Évaluation) |
| `3_Assessment_Blueprint.ipynb`    | Phase 6 — synthèse blueprint + soumission Kaggle |

Documentation détaillée du modèle dans `docs/MLCanvas_v1.2.pdf` (canvas OWNML rempli).

## Setup environnement

Python **3.13** requis (les versions pinnées dans `requirements.txt` le ciblent).

```bash
# 1. Créer le virtual env (depuis la racine du repo)
python3.13 -m venv .venv
source .venv/bin/activate

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
source .venv/bin/activate
jupyter lab    # ou : jupyter notebook
```

## Données

Le dossier `data/` contient les CSV Kaggle d'origine (`train.csv`, `test.csv`, `sample_submission.csv`, `data_description.txt`). Ils sont versionnés dans le repo.

## Soumission Kaggle

La soumission est générée par `3_Assessment_Blueprint.ipynb` (cellule du bas), qui produit un fichier `submission.csv` à la racine du repo. Ce fichier est git-ignoré — il faut l'uploader manuellement sur Kaggle.
