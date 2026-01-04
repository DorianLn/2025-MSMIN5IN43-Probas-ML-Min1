📰 Détection de Fake News par NLP Avancé et Transformers
📌 Présentation du projet

Ce projet s’inscrit dans le cadre du module de NLP avancé pour le traitement de bases de données.
L’objectif est de concevoir un système intelligent de détection de fake news, basé sur des modèles Transformers pré-entraînés et fine-tunés, capables de traiter des articles de presse en anglais et en français.

Le projet est réalisé en groupe et combine :

des techniques avancées de Traitement Automatique du Langage Naturel (TALN),

l’exploitation de datasets et modèles via Hugging Face,

et le développement d’une interface utilisateur pour une utilisation concrète.

👥 Membres du groupe

Lamyae TALA
Safe BERRICHI
Pauline GOFFINET

🎯 Objectifs du projet

Détecter automatiquement si une information est vraie ou fausse

Appliquer des techniques de NLP avancé sur de grandes bases de données textuelles

Fine-tuner et comparer plusieurs modèles Transformers

Gérer le multilinguisme (anglais / français)

Mettre en place une interface interactive de vérification des news

🧠 Modèles utilisés
🔹 Données en anglais

Deux modèles Transformers ont été fine-tunés pour la détection de fake news en anglais :

BERT (bert-base-uncased)

RoBERTa (roberta-base)

Ces modèles permettent une comparaison des performances sur les données anglophones.

🔹 Données en français

Pour les articles en français, nous avons utilisé :

CamemBERT (camembert-base)

CamemBERT est un modèle spécifiquement entraîné pour la langue française, ce qui le rend particulièrement adapté à la détection de fake news en français.

🗄️ Données & Stockage des modèles

Les datasets sont chargés depuis Hugging Face Datasets

Les modèles fine-tunés sont :

sauvegardés localement,

puis stockés et versionnés sur Hugging Face Hub pour faciliter le partage, la réutilisation et la reproductibilité

🖥️ Interface utilisateur

Une interface interactive permet à l’utilisateur de vérifier une news en quelques clics.

🎛️ Fonctionnalités de l’interface

L’utilisateur peut :

saisir le texte d’une news,

choisir le modèle de vérification via trois boutons :

Bouton	Langue	Modèle
🇫🇷 CamemBERT	Français	CamemBERT
🇬🇧 BERT	Anglais	BERT
🇬🇧 RoBERTa	Anglais	RoBERTa

L’interface retourne :

la prédiction (Fake / Real),

un score de confiance associé.

🗂️ Structure du projet
FakeNews-Detection/
│
├── notebooks/
│   ├── notebook_1_BERT.ipynb
│   ├── notebook_2_RoBERTa.ipynb
│   └── notebook_3_CamemBERT.ipynb
│
├── interface/
│   └── app.py
│
├── data/
│   └── datasets (Hugging Face)
│
├── models/
│   ├── bert/
│   ├── roberta/
│   └── camembert/
│
├── results/
│   └── metrics_et_evaluations/
│
└── README.md

⚙️ Environnement technique

Langage : Python

Frameworks & bibliothèques :

PyTorch

Hugging Face Transformers & Datasets

Scikit-learn

Accélération matérielle :

Entraînement sur GPU (CUDA)

🧪 Méthodologie

Chargement des données depuis Hugging Face

Nettoyage et prétraitement professionnel des textes

Tokenisation adaptée à chaque modèle

Fine-tuning des modèles Transformers

Évaluation à l’aide de métriques standard

Intégration des modèles dans une interface utilisateur

📊 Évaluation

Les modèles sont évalués à l’aide de :

Accuracy

Precision

Recall

F1-score

Matrice de confusion

Une analyse comparative est réalisée entre BERT et RoBERTa pour les données anglaises, et CamemBERT pour les données françaises.

🚀 Perspectives d’amélioration

Ajout d’autres langues

Déploiement de l’application en ligne

Amélioration de l’explicabilité des prédictions

Intégration de nouvelles sources de données