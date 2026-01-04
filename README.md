# 📰 Détection de Fake News par NLP Avancé & Transformers

## 📌 Présentation du projet
Ce projet a été réalisé dans le cadre du **module de NLP avancé**.  
L’objectif principal est de concevoir un **système intelligent de détection de Fake News**, capable de distinguer des articles **vrais** et **faux** en **anglais** et en **français**, en s’appuyant sur des **modèles Transformers de l’état de l’art**.

Une attention particulière a été portée à la **résilience des modèles face à la désinformation sophistiquée**, notamment les contenus complotistes bien rédigés, via des **stratégies avancées de calibration et de pondération des erreurs**.

---

## 👥 Membres du groupe
- **Nom Prénom**
- **Nom Prénom**
- **Nom Prénom**

*(à compléter)*

---

## 🎯 Objectifs techniques
- **Multilinguisme**  
  Fine-tuning de modèles spécifiques pour l’anglais et le français.

- **Data Augmentation**  
  Utilisation de la **Back-Translation (FR ↔ EN)** pour enrichir et équilibrer les jeux de données d’entraînement.

- **Optimisation de la précision**  
  Implémentation d’une **fonction de perte pondérée (Weighted Cross-Entropy)** afin de pénaliser davantage les faux négatifs.

- **Calibration de l’inférence**  
  Mise en place d’un **seuil de suspicion personnalisé** pour détecter des signaux faibles de désinformation.

---

## 🧠 Modèles & stratégies

### 🔹 Modèles pour l’anglais
- **BERT** (`bert-base-uncased`)
- **RoBERTa** (`roberta-base`)  
  → Meilleure compréhension contextuelle et robustesse linguistique.

### 🔹 Modèle pour le français
- **CamemBERT** (`camembert-base`)  
  → Fine-tuning avec **régularisation stricte (Weight Decay)** afin de limiter le biais stylistique et le sur-apprentissage.

---

## 🧪 Méthodologie avancée
Pour faire face aux **Fake News très bien rédigées**, nous avons mis en œuvre les techniques suivantes :

- **Back-Translation**  
  Traduction automatique via *Helsinki-NLP* pour enrichir la classe minoritaire.

- **Weighted Trainer**  
  Pondération des classes :
  - VRAI : **1.0**
  - FAKE : **3.0**  
  afin de rendre le modèle plus vigilant face à la désinformation.

- **Ultra-Suspicious Threshold**  
  Ajustement du seuil de décision lors de l’inférence :  
  un article est signalé comme **suspect** dès que la confiance en la classe *VRAI* descend sous **99.99%**.

---

## 🖥️ Interface utilisateur
Une **interface interactive** permet à l’utilisateur de saisir un texte et d’obtenir un diagnostic immédiat selon le modèle choisi.

| Bouton | Langue | Modèle |
|------|------|------|
| 🇫🇷 CamemBERT | Français | CamemBERT v2 (calibré) |
| 🇬🇧 BERT | Anglais | BERT-base |
| 🇬🇧 RoBERTa | Anglais | RoBERTa-base |

---

## 🗂️ Structure du projet
```text
.
├── notebooks/
│   ├── EN_Fakenews_Bert.ipynb      # Pipeline anglais - BERT
│   ├── EN_fakenews_RoBERTa.ipynb   # Pipeline anglais - RoBERTa
│   └── FR_Fake.ipynb               # Pipeline français (augmentation + calibration)
├── interface/
│   └── app.py                     # Application Streamlit / Gradio
├── .gitignore                     # Exclusion des modèles > 100 Mo
└── README.md
