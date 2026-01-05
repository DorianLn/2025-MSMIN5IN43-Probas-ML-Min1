# 📁 Dossier Data

## Structure

```
data/
├── raw/                    # Données brutes téléchargées
│   ├── allocine_sentiment.csv
│   ├── emotions.csv
│   ├── irony.csv
│   └── combined_multitask.csv
│
├── processed/              # Données traitées et splitées
│   ├── train.csv          # 70% - Entraînement
│   ├── val.csv            # 15% - Validation
│   └── test.csv           # 15% - Test
│
├── dataset_stats.json      # Statistiques globales
└── exploration_report.json # Rapport d'exploration
```

## Description des fichiers

### Fichiers RAW

- **allocine_sentiment.csv** : Critiques de films français (sentiment positif/négatif)
- **emotions.csv** : Textes annotés avec 7 émotions
- **irony.csv** : Textes ironiques vs non-ironiques
- **combined_multitask.csv** : Dataset combiné avec les 3 tâches

### Fichiers PROCESSED

- **train.csv** : Données d'entraînement (stratifiées)
- **val.csv** : Données de validation (stratifiées)
- **test.csv** : Données de test (stratifiées) - **NE PAS TOUCHER JUSQU'À L'ÉVALUATION FINALE !**

## Format des données

Chaque fichier CSV contient :
- `text` : Le texte à analyser
- `emotion` : L'émotion (joie, tristesse, colere, peur, surprise, degout, neutre)
- `emotion_id` : ID numérique de l'émotion (0-6)
- `sentiment` : Le sentiment (0=négatif, 1=neutre, 2=positif)
- `is_ironic` : Ironie (0=non-ironique, 1=ironique)

## ⚠️ Important

- **NE PAS** modifier les fichiers dans `processed/` manuellement
- **NE PAS** utiliser le test set pendant le développement
- Les données sont stratifiées pour garder la distribution des classes
