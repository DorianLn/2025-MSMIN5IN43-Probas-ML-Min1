# Conformal Prediction for House Prices

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) [![Angular](https://img.shields.io/badge/Angular-17-red.svg)](https://angular.io/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/) [![MAPIE](https://img.shields.io/badge/MAPIE-0.7.0-purple.svg)](https://mapie.readthedocs.io/)

Ce projet est une démonstration complète de l'application de la **prédiction conforme** pour des problèmes de régression. L'objectif est de prédire les prix de l'immobilier (en utilisant le dataset _Ames Housing_) et, plus important encore, de fournir des **intervalles de confiance** rigoureux pour chaque prédiction.

Le modèle sous-jacent est traité comme une "boîte noire", et la bibliothèque `MAPIE` est utilisée pour calibrer ses sorties et générer les intervalles.

## ✨ Architecture

Le projet est structuré en deux parties principales :

- **`backend/`** : Une API RESTful construite avec **FastAPI** qui sert le modèle de machine learning. Elle expose un endpoint pour recevoir les caractéristiques d'une maison et retourne une prédiction de prix avec son intervalle de confiance.
- **`frontend/`** : Une application web monopage (SPA) développée avec **Angular** qui fournit une interface utilisateur pour interagir avec l'API, soumettre des données et visualiser les résultats.

---

## 🚀 Démarrage rapide (Getting Started)

Suivez ces étapes pour configurer et lancer le projet en local.

### 1. Prérequis

Assurez-vous d'avoir les outils suivants installés sur votre machine :

- **Python** (version 3.12)
- **Node.js** (version 18 ou supérieure)
- **npm** (généralement inclus avec Node.js)

### 2. Configuration du Backend

Ouvrez un premier terminal et suivez ces instructions :

1. **Accédez au dossier backend :**

   ```bash
   cd backend
   ```

2. **Créez et activez un environnement virtuel :**

   - _macOS / Linux_
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - _Windows (PowerShell)_
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. **Installez les dépendances Python :**

   ```bash
   pip install -r requirements.txt
   ```

### 3. Entraînement du Modèle (Optionnel)

Le backend est configuré pour utiliser un modèle pré-entraîné. Si vous souhaitez ré-entraîner les modèles vous-même, vous pouvez exécuter l'un des scripts suivants depuis la racine du projet (`Conformal_Prediction/`) :

- **Pour entraîner le modèle `GradientBoosting` (recommandé, actuellement utilisé) :**

  ```bash
  python backend/scripts/train_ames_gradient.py
  ```

  _Cela génère `backend/models/ames_gb_mapie.joblib`._

- **Pour entraîner le modèle `RandomForest` (alternatif) :**

  ```bash
  python backend/scripts/train_ames.py
  ```

  _Cela génère `backend/models/ames_rf_mapie.joblib`. Pensez à mettre à jour `backend/app/main.py` si vous voulez utiliser ce modèle._

### 4. Lancement du Backend

Dans le terminal où votre environnement virtuel est activé :

```bash
# Assurez-vous d'être dans le dossier `backend/`
uvicorn app.main:app --reload --port 8000
```

L'API est maintenant accessible à l'adresse `http://127.0.0.1:8000`.

### 5. Configuration et Lancement du Frontend

Ouvrez un **second terminal** :

1. **Accédez au dossier frontend :**

   ```bash
   cd frontend
   ```

2. **Installez les dépendances Node.js :**

   ```bash
   npm install
   ```

3. **Lancez l'application Angular :**

   ```bash
   npm start
   ```

L'application web est maintenant accessible à l'adresse `http://localhost:4200`.

---

## 🔧 Utilisation de l'API

Une fois le backend démarré, vous pouvez interroger l'endpoint `/predict/` avec une requête `POST`.

Voici un exemple d'appel avec `curl` (les valeurs `null` sont gérées par le pipeline d'imputation) :

```bash
curl -X POST "http://127.0.0.1:8000/predict/"
-H "Content-Type: application/json"
-d
'{
      "MS SubClass": 60,
      "Lot Frontage": null,
      "Lot Area": 12000,
      "Overall Qual": 7,
      "Overall Cond": 5,
      "Year Built": 2005,
      "Gr Liv Area": 1500,
      "Full Bath": 2,
      "Bedroom AbvGr": 3,
      "Kitchen Qual": "Gd",
      "Garage Cars": 2,
      "Pool Area": 0
    }'
```

**Réponse attendue :**

```json
{
  "prediction": 195000.0,
  "lower_bound": 175000.0,
  "upper_bound": 215000.0
}
```

_Les valeurs sont des exemples._

---

## 📝 Remarques

- Le backend est configuré avec une politique **CORS** permissive pour le développement local, autorisant les requêtes depuis `http://localhost:4200`.
- Pour une mise en production, il serait nécessaire de restreindre CORS, d'ajouter une authentification et de conteneuriser les applications (par exemple avec Docker).
