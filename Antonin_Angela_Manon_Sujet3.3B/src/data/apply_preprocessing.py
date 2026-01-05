"""
Script pour appliquer le preprocessing aux datasets et afficher des statistiques
"""

import os
import pandas as pd
import sys

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import preprocess_dataset, clean_text, get_text_stats


def main():
    print("=" * 70)
    print("PREPROCESSING DES DATASETS")
    print("=" * 70)
    
    # Chemins
    data_dir = "data/processed"
    
    # Charger les datasets
    print("\n📂 Chargement des données...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    print(f"   ✓ Train: {len(train_df)} exemples")
    print(f"   ✓ Val:   {len(val_df)} exemples")
    print(f"   ✓ Test:  {len(test_df)} exemples")
    
    # =============================================================================
    # PREPROCESSING TRAIN
    # =============================================================================
    print("\n" + "=" * 70)
    print("1️⃣ PREPROCESSING TRAIN SET")
    print("=" * 70)
    
    train_df = preprocess_dataset(train_df)
    
    # Afficher quelques exemples
    print("\n📋 Exemples de textes nettoyés (Train) :\n")
    for i in range(min(5, len(train_df))):
        row = train_df.iloc[i]
        print(f"{i+1}. Original : {row['text'][:80]}...")
        print(f"   Nettoyé  : {row['text_clean'][:80]}...")
        print(f"   Émotion  : {row['emotion']}, Sentiment: {row['sentiment']}, Ironie: {row['is_ironic']}")
        print()
    
    # =============================================================================
    # PREPROCESSING VAL
    # =============================================================================
    print("\n" + "=" * 70)
    print("2️⃣ PREPROCESSING VALIDATION SET")
    print("=" * 70)
    
    val_df = preprocess_dataset(val_df)
    
    # =============================================================================
    # PREPROCESSING TEST
    # =============================================================================
    print("\n" + "=" * 70)
    print("3️⃣ PREPROCESSING TEST SET")
    print("=" * 70)
    
    test_df = preprocess_dataset(test_df)
    
    # =============================================================================
    # STATISTIQUES COMPARATIVES
    # =============================================================================
    print("\n" + "=" * 70)
    print("📊 STATISTIQUES COMPARATIVES")
    print("=" * 70)
    
    print("\n🔍 Longueur des textes :")
    print(f"   Train : Moy={train_df['text_length'].mean():.1f}, "
          f"Min={train_df['text_length'].min()}, Max={train_df['text_length'].max()}")
    print(f"   Val   : Moy={val_df['text_length'].mean():.1f}, "
          f"Min={val_df['text_length'].min()}, Max={val_df['text_length'].max()}")
    print(f"   Test  : Moy={test_df['text_length'].mean():.1f}, "
          f"Min={test_df['text_length'].min()}, Max={test_df['text_length'].max()}")
    
    print("\n😊 Présence d'emojis :")
    print(f"   Train : {(train_df['emoji_count'] > 0).sum()} textes ({(train_df['emoji_count'] > 0).sum() / len(train_df) * 100:.1f}%)")
    print(f"   Val   : {(val_df['emoji_count'] > 0).sum()} textes ({(val_df['emoji_count'] > 0).sum() / len(val_df) * 100:.1f}%)")
    print(f"   Test  : {(test_df['emoji_count'] > 0).sum()} textes ({(test_df['emoji_count'] > 0).sum() / len(test_df) * 100:.1f}%)")
    
    print("\n❗ Ponctuation expressive :")
    print(f"   Train : {(train_df['exclamation_count'] > 0).sum()} avec '!' ({(train_df['exclamation_count'] > 0).sum() / len(train_df) * 100:.1f}%)")
    print(f"   Val   : {(val_df['exclamation_count'] > 0).sum()} avec '!' ({(val_df['exclamation_count'] > 0).sum() / len(val_df) * 100:.1f}%)")
    print(f"   Test  : {(test_df['exclamation_count'] > 0).sum()} avec '!' ({(test_df['exclamation_count'] > 0).sum() / len(test_df) * 100:.1f}%)")
    
    # =============================================================================
    # DISTRIBUTION DES CLASSES
    # =============================================================================
    print("\n" + "=" * 70)
    print("🏷️ DISTRIBUTION DES CLASSES")
    print("=" * 70)
    
    print("\n📊 Émotions (Train) :")
    emotion_dist = train_df['emotion'].value_counts().sort_index()
    for emotion, count in emotion_dist.items():
        print(f"   {emotion:12} : {count:3} ({count/len(train_df)*100:5.1f}%)")
    
    print("\n📊 Sentiment (Train) :")
    sentiment_map = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
    sentiment_dist = train_df['sentiment'].value_counts().sort_index()
    for sent_id, count in sentiment_dist.items():
        print(f"   {sentiment_map[sent_id]:12} : {count:3} ({count/len(train_df)*100:5.1f}%)")
    
    print("\n📊 Ironie (Train) :")
    irony_map = {0: 'Non-ironique', 1: 'Ironique'}
    irony_dist = train_df['is_ironic'].value_counts().sort_index()
    for irony_id, count in irony_dist.items():
        print(f"   {irony_map[irony_id]:14} : {count:3} ({count/len(train_df)*100:5.1f}%)")
    
    # =============================================================================
    # SAUVEGARDE
    # =============================================================================
    print("\n" + "=" * 70)
    print("💾 SAUVEGARDE DES DONNÉES PREPROCESSÉES")
    print("=" * 70)
    
    # Sauvegarder avec les colonnes nettoyées
    train_df.to_csv(os.path.join(data_dir, "train_preprocessed.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val_preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test_preprocessed.csv"), index=False)
    
    print(f"\n   ✓ {data_dir}/train_preprocessed.csv")
    print(f"   ✓ {data_dir}/val_preprocessed.csv")
    print(f"   ✓ {data_dir}/test_preprocessed.csv")
    
    # =============================================================================
    # RÉSUMÉ FINAL
    # =============================================================================
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING TERMINÉ !")
    print("=" * 70)
    
    print(f"\n📈 Résumé :")
    print(f"   • {len(train_df)} exemples d'entraînement preprocessés")
    print(f"   • {len(val_df)} exemples de validation preprocessés")
    print(f"   • {len(test_df)} exemples de test preprocessés")
    print(f"   • Emojis préservés : ✓")
    print(f"   • Ponctuation préservée : ✓")
    print(f"   • URLs supprimées : ✓")
    print(f"   • Mentions supprimées : ✓")
    
    print(f"\n🎯 Prochaine étape :")
    print(f"   • Créer le modèle baseline (TF-IDF + Logistic Regression)")
    print("=" * 70)


if __name__ == "__main__":
    main()
