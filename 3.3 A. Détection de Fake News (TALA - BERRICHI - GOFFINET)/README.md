# 🛡️ FactGuard : Détection de Fake News par NLP Avancé & Transformers

---

## 📌 Présentation du projet

Ce projet, réalisé dans le cadre du **module de NLP avancé**, propose un système intelligent de **détection de Fake News**.  
L’objectif est de distinguer les articles **vrais** et **faux** en **anglais** et en **français**, en s’appuyant sur des modèles **Transformers de l’état de l’art**.

Une attention particulière a été portée à la **résilience face à la désinformation sophistiquée** (contenus complotistes bien rédigés) via des stratégies avancées de **calibration** et de **pondération des erreurs**.

---

## 👥 Membres du groupe

- Lamyae TALA 
- Safae BERRICHI
- Pauline GOFFINET

---

## 🎯 Objectifs techniques & Méthodologie

Pour faire face aux Fake News **« haute fidélité »**, nous avons implémenté des techniques de pointe :

### 🔹 Multilinguisme & Data Augmentation
- Utilisation de la **Back-Translation (FR ↔ EN)** via **Helsinki-NLP**
- Enrichissement et équilibrage des jeux de données
- Réduction du sur-apprentissage sur des patterns lexicaux spécifiques

### 🔹 Weighted Cross-Entropy
Implémentation d’un **Weighted Trainer** pour pénaliser davantage les faux négatifs :

- **Poids classe VRAI** : `1.0`
- **Poids classe FAKE** : `3.0`  
  *(Vigilance accrue face à la désinformation)*

### 🔹 Ultra-Suspicious Threshold
- Ajustement dynamique du seuil de décision à l’inférence
- Un article est signalé comme suspect dès que la **confiance en la véracité** passe sous un seuil critique

---

## 🧠 Modèles & Inférence (Hugging Face)

Les modèles sont entraînés, calibrés et hébergés sur le **Hub Hugging Face**.

| Modèle     | Langue | Base Transformer        | Lien Hugging Face | Logique Label        |
|-----------|--------|-------------------------|-------------------|----------------------|
| CamemBERT | 🇫🇷 FR | camembert-base          | Consulter le modèle | 0 = Vrai / 1 = Fake |
| BERT      | 🇬🇧 EN | bert-base-uncased       | Consulter le modèle | 1 = Vrai / 0 = Fake |
| RoBERTa   | 🇺🇸 EN | roberta-base            | Consulter le modèle | 1 = Vrai / 0 = Fake |

---

## 🖥️ Architecture du Système (Full-Stack)

Le projet est divisé en **trois briques technologiques** :

### 1️⃣ Backend — FastAPI & PyTorch
API robuste optimisée pour l’inférence sur **NVIDIA RTX 3050 Ti (4GB)** :

- Gestion intelligente de la **VRAM**
  - `torch.cuda.empty_cache()` lors du changement de modèle
- **Normalisation du texte**
  - Nettoyage via Regex (URLs, espaces, caractères parasites)

### 2️⃣ Frontend — React & Tailwind CSS
Interface utilisateur moderne et réactive :

- Diagnostic immédiat avec **score de confiance**
- Facteurs d’analyse : *Style*, *Vocabulaire*, *Source*
- **UX dynamique** avec animations Framer Motion

### 3️⃣ Notebooks — Recherche & Training
- `EN_Fakenews_Bert.ipynb` : Pipeline anglais BERT
- `EN_fakenews_RoBERTa.ipynb` : Pipeline anglais RoBERTa
- `FR_Fake.ipynb` : Pipeline français (Augmentation + Calibration CamemBERT)

---

## 🛠️ Installation et Lancement

### ▶ Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### ▶ Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🚦 Structure des Fichiers

```text
.
├── notebooks/                      # Phase de Recherche & Entraînement
│   ├── EN_Fakenews_Bert.ipynb      # Pipeline anglais — BERT
│   ├── EN_fakenews_RoBERTa.ipynb   # Pipeline anglais — RoBERTa
│   └── FR_Fake.ipynb               # Pipeline français — CamemBERT (Back-translation)
│
├── backend/                        # API FastAPI (Python)
│   ├── main.py                     # Point d’entrée, configuration CORS et routes
│   ├── requirements.txt            # Dépendances (FastAPI, Torch, Transformers, Pydantic)
│   ├── .gitignore                  # Exclusion venv, __pycache__, fichiers .env
│   └── app/
│       ├── config.py               # Configuration (DEVICE GPU, modèles Hugging Face)
│       ├── schemas.py              # Modèles Pydantic (AnalysisRequest, AnalysisResponse)
│       ├── utils.py                # Nettoyage Regex & mapping des labels (0/1)
│       ├── models/
│       │   └── model_loader.py     # Inférence & gestion VRAM (RTX 3050 Ti)
│       └── routes/
│           └── predict.py          # Route POST /predict (IA ↔ API)
│
├── frontend/                       # Interface Utilisateur (React + Vite)
│   ├── package.json                # Dépendances (Tailwind, Framer Motion, Lucide)
│   ├── tailwind.config.js          # Configuration UI (couleurs, typographie)
│   ├── src/
│   │   ├── components/             # Composants UI
│   │   │   ├── AnalysisLoader.tsx  # Animation de chargement
│   │   │   └── AnalysisResults.tsx # Affichage des scores & jauges
│   │   └── pages/
│   │       └── Index.tsx           # Page principale (state + appels API)
│   └── public/                     # Assets statiques
│
└── README.md                       # Documentation complète du projet

```

---

## 🛡️ Licence

Projet réalisé dans un **cadre académique** pour le module de **NLP Avancé**.  
Modèles optimisés pour la **recherche** et la **prévention contre la désinformation**.
