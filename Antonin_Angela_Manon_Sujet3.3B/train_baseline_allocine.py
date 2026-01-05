"""
Script d'entraînement du modèle Baseline uniquement sur Allociné (comme CamemBERT)
Pour comparaison équitable avec le modèle CamemBERT
"""

import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.models.baseline import BaselineModel

def train_baseline_allocine():
    print("=" * 70)
    print("🚀 ENTRAÎNEMENT BASELINE SUR ALLOCINÉ (Sentiment uniquement)")
    print("=" * 70)
    
    # Load data
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print("❌ Dossier data/processed non trouvé. Lancez d'abord le téléchargement des données.")
        return

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Filter only Allociné data (sentiment task, no synthetic data)
    # Les données Allociné ont sentiment_label != -1
    train_sentiment = train_df[train_df['sentiment_label'] != -1].copy()
    val_sentiment = val_df[val_df['sentiment_label'] != -1].copy()
    test_sentiment = test_df[test_df['sentiment_label'] != -1].copy()
    
    print(f"\n📊 Dataset Allociné (Sentiment):")
    print(f"   Train: {len(train_sentiment)} exemples")
    print(f"   Val:   {len(val_sentiment)} exemples")
    print(f"   Test:  {len(test_sentiment)} exemples")
    
    if len(train_sentiment) == 0:
        print("❌ Pas de données Allociné trouvées!")
        return
    
    # Distribution des classes
    print(f"\n📈 Distribution des classes (Train):")
    print(train_sentiment['sentiment_label'].value_counts().sort_index())
    print("   (0=négatif, 2=positif)")
    
    # Entraînement
    print(f"\n🔧 Entraînement du modèle Baseline (TF-IDF + LogReg)...")
    model = BaselineModel(task_name='sentiment')
    
    X_train = train_sentiment['text'].astype(str).tolist()
    y_train = train_sentiment['sentiment_label'].tolist()
    
    X_val = val_sentiment['text'].astype(str).tolist()
    y_val = val_sentiment['sentiment_label'].tolist()
    
    X_test = test_sentiment['text'].astype(str).tolist()
    y_test = test_sentiment['sentiment_label'].tolist()
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print(f"\n{'='*70}")
    print("📊 RÉSULTATS SUR LE SET DE VALIDATION")
    print(f"{'='*70}")
    
    y_pred_val = model.predict(X_val)
    y_pred_proba_val = model.predict_proba(X_val)
    
    accuracy_val = accuracy_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val, average='weighted')
    
    # Pour le log loss, on a besoin de probabilités pour toutes les classes
    # Créer un mapping des prédictions aux indices
    classes = model.label_encoder.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Créer une matrice de probabilités au bon format
    y_val_encoded = [class_to_idx[label] for label in y_val]
    loss_val = log_loss(y_val_encoded, y_pred_proba_val)
    
    print(f"\n✨ Métriques Validation:")
    print(f"   Accuracy: {accuracy_val:.4f} ({accuracy_val*100:.2f}%)")
    print(f"   F1 Score (weighted): {f1_val:.4f}")
    print(f"   Log Loss: {loss_val:.4f}")
    
    print(f"\n📋 Rapport de classification détaillé (Validation):")
    print(classification_report(y_val, y_pred_val, target_names=['Négatif', 'Positif']))
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("📊 RÉSULTATS SUR LE SET DE TEST")
    print(f"{'='*70}")
    
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)
    
    accuracy_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    
    y_test_encoded = [class_to_idx[label] for label in y_test]
    loss_test = log_loss(y_test_encoded, y_pred_proba_test)
    
    print(f"\n✨ Métriques Test:")
    print(f"   Accuracy: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
    print(f"   F1 Score (weighted): {f1_test:.4f}")
    print(f"   Log Loss: {loss_test:.4f}")
    
    print(f"\n📋 Rapport de classification détaillé (Test):")
    print(classification_report(y_test, y_pred_test, target_names=['Négatif', 'Positif']))
    
    # Sauvegarder les résultats
    results = {
        'model': 'Baseline (TF-IDF + LogisticRegression)',
        'dataset': 'Allociné',
        'task': 'sentiment',
        'timestamp': datetime.now().isoformat(),
        'data_size': {
            'train': len(train_sentiment),
            'val': len(val_sentiment),
            'test': len(test_sentiment)
        },
        'validation_metrics': {
            'accuracy': float(accuracy_val),
            'f1_weighted': float(f1_val),
            'log_loss': float(loss_val)
        },
        'test_metrics': {
            'accuracy': float(accuracy_test),
            'f1_weighted': float(f1_test),
            'log_loss': float(loss_test)
        }
    }
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_path = results_dir / "baseline_allocine_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résultats sauvegardés dans: {results_path}")
    
    # Sauvegarder le modèle
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model.save(models_dir)
    print(f"💾 Modèle sauvegardé dans: {models_dir / f'baseline_{model.task_name}.pkl'}")
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*70)
    print("\n💡 Pour comparer avec CamemBERT, consultez:")
    print(f"   - Baseline: {results_path}")
    print(f"   - CamemBERT: models/training_history.json")

if __name__ == "__main__":
    train_baseline_allocine()
