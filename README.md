# Optimisation Discrète — UC Hydro-Thermique

Ce projet met en œuvre un modèle d’Unit Commitment hydro-thermique avec Pyomo et propose un tableau de bord Streamlit interactif pour explorer les résultats (dispatch, engagements, volumes des réservoirs et animation des niveaux d’eau).

## Prérequis
- Python 3.10+ recommandé.
- Un solveur linéaire compatible Pyomo. Le projet utilise GLPK par défaut.
  - macOS (brew) : `brew install glpk`
  - Debian/Ubuntu : `sudo apt-get install glpk-utils`
- Accès au fichier de données inclus : `data/10_0_1_w.nc4`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Lancer le dashboard
```bash
./.venv/bin/streamlit run main.py
# ou si le venv est activé
streamlit run main.py
```
Le tableau de bord affiche :
- Coût total, statut et condition d’arrêt du solveur.
- Graphes Plotly : dispatch empilé, engagement on/off, volumes de réservoirs.
- Animation des niveaux de réservoirs (play/pause + slider).
- Tables de données (dispatch détaillé, commitment, volumes).

## Structure rapide
- `main.py` : point d’entrée, construction du modèle, résolution, génération des graphiques Plotly et du dashboard.
- `src/model/pyomo_uc.py` : définition du modèle UC hydro-thermique Pyomo.
- `src/loaders/ucblock_thermal.py` : chargement des données depuis le NetCDF.
- `data/10_0_1_w.nc4` : données d’entrée.
- `outputs/` : répertoire de sortie (LP exporté, etc.), créé au lancement.

## Points d’attention
- Si Streamlit signale « No module named pyomo », c’est que l’exécutable Streamlit n’utilise pas le venv. Lancer avec `./.venv/bin/streamlit …` ou activer le venv avant.
- Le solveur doit être présent dans le PATH (`glpsol` pour GLPK). Sinon, installer GLPK ou modifier `SolverFactory("glpk")` pour un autre solveur disponible.

## Tests / validation
Il n’y a pas de suite de tests automatisés. Pour vérifier l’environnement :
```bash
./.venv/bin/python -c "import pyomo, streamlit, plotly; print('deps OK')"
./.venv/bin/python -m pyomo.environ
```

## Licence
Non spécifiée dans le dépôt. Ajouter un fichier `LICENSE` si nécessaire.
