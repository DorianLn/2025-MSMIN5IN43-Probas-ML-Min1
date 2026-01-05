# 🛡️ FactGuard API - Backend de Détection de Fake News

Ce projet est une API haute performance conçue pour analyser la crédibilité d'articles de presse en temps réel. Elle utilise des modèles de langage **SOTA (State-of-the-Art)** basés sur l'architecture Transformer.

## 🌟 Points Forts

- **Multi-langues** : Support du Français (**CamemBERT**) et de l'Anglais (**BERT**, **RoBERTa**).
- **Optimisation GPU** : Gestion intelligente de la mémoire VRAM pour les cartes graphiques **NVIDIA RTX 3050 Ti** (4 Go).
- **Nettoyage Automatique** : Prétraitement du texte pour éliminer les bruits (URLs, espaces superflus).
- **Validation stricte** : Utilisation de Pydantic pour garantir des échanges de données sécurisés avec le Frontend.

---

## 🏗️ Architecture du Projet

```text
backend/
├── main.py              # Point d'entrée de l'application
├── requirements.txt     # Dépendances du projet
├── .gitignore           # Fichiers à exclure de Git
└── app/
    ├── config.py        # Configuration (Modèles Hugging Face, GPU)
    ├── schemas.py       # Modèles de données Pydantic
    ├── utils.py         # Fonctions d'aide (Nettoyage, mapping labels)
    ├── models/
    │   └── model_loader.py # Chargeur de modèles avec gestion de VRAM
    └── routes/
        └── predict.py   # Logique de la route API /predict
```

---

## 🛠️ Installation et Configuration

### 1. Prérequis

- Python **3.9+**
- Pilotes **NVIDIA CUDA** installés (pour l'accélération GPU)

### 2. Installation

Clonez le dépôt et créez un environnement virtuel :

```bash
python -m venv venv
```

Activation de l'environnement :

```bash
# Windows
.\venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

### 3. Exécution

Lancez le serveur de développement :

```bash
uvicorn main:app --reload
```

L'API sera accessible sur :  
👉 http://127.0.0.1:8000

---

## 🔌 API Endpoints

### POST `/predict`

Analyse un texte avec le modèle spécifié.

#### Exemple de requête

```json
{
  "text": "L'intelligence artificielle va révolutionner le monde.",
  "model": "camembert"
}
```

#### Exemple de réponse

```json
{
  "isReliable": true,
  "confidence": 98.45,
  "factors": {
    "style": { "score": 99.0, "label": "Analysé" },
    "vocabulary": { "score": 97.5, "label": "Vérifié" },
    "source": { "score": 85.0, "label": "Évalué" }
  },
  "summary": "Modèle CAMEMBERT : Article jugé FIABLE."
}
```

---

## 🧠 Spécifications des Modèles IA

L'API gère automatiquement les différences de labels entre les modèles entraînés :

| Modèle     | Identifiant | Langue | Logique de Fiabilité |
|-----------|------------|--------|----------------------|
| CamemBERT | camembert  | 🇫🇷 FR | LABEL_0 = FIABLE |
| BERT      | bert       | 🇬🇧 EN | LABEL_1 = FIABLE |
| RoBERTa   | roberta    | 🇺🇸 EN | LABEL_1 = FIABLE |
