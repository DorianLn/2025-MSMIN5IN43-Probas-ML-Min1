"""
Module de preprocessing des textes pour l'analyse de sentiment multi-tâches

IMPORTANT : On GARDE les emojis et la ponctuation car ils sont essentiels 
pour détecter les émotions et l'ironie !
"""

import re
import pandas as pd
from typing import List, Dict


# =============================================================================
# FONCTIONS DE NETTOYAGE
# =============================================================================

def clean_text(text: str, keep_emojis: bool = True, keep_punctuation: bool = True) -> str:
    """
    Nettoie un texte en supprimant les éléments inutiles mais en gardant
    les emojis et la ponctuation expressive.
    
    Args:
        text: Texte à nettoyer
        keep_emojis: Si True, garde les emojis (recommandé pour sentiment)
        keep_punctuation: Si True, garde la ponctuation (recommandé)
    
    Returns:
        Texte nettoyé
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Supprimer les URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 2. Supprimer les mentions (@utilisateur)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Supprimer les hashtags mais garder le texte
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 4. Nettoyer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # 5. Supprimer les espaces en début/fin
    text = text.strip()
    
    # Note: On NE supprime PAS les emojis ni la ponctuation !
    # Ils sont cruciaux pour l'analyse de sentiment
    
    return text


def normalize_text(text: str) -> str:
    """
    Normalise le texte (minuscules, etc.) - ATTENTION : peut perdre de l'info !
    À utiliser avec précaution.
    
    Args:
        text: Texte à normaliser
    
    Returns:
        Texte normalisé
    """
    # Pour l'instant, on ne fait PAS de lowercasing car les majuscules
    # peuvent indiquer de la colère ("JE SUIS EN COLÈRE !!!")
    # On pourrait le faire plus tard si nécessaire
    
    return text


def get_text_stats(text: str) -> Dict[str, int]:
    """
    Calcule des statistiques sur un texte.
    
    Args:
        text: Texte à analyser
    
    Returns:
        Dictionnaire avec les stats
    """
    # Compter les emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    emoji_count = len(emoji_pattern.findall(text))
    
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'emoji_count': emoji_count,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
    }


# =============================================================================
# PREPROCESSING DE DATASETS
# =============================================================================

def preprocess_dataset(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Applique le preprocessing à un dataset complet.
    
    Args:
        df: DataFrame avec les textes
        text_column: Nom de la colonne contenant les textes
    
    Returns:
        DataFrame avec une nouvelle colonne 'text_clean'
    """
    print(f"📝 Preprocessing de {len(df)} exemples...")
    
    # Appliquer le nettoyage
    df['text_clean'] = df[text_column].apply(clean_text)
    
    # Calculer les stats
    print("📊 Calcul des statistiques...")
    stats = df['text_clean'].apply(get_text_stats)
    
    # Ajouter les stats comme colonnes
    df['text_length'] = stats.apply(lambda x: x['length'])
    df['word_count'] = stats.apply(lambda x: x['word_count'])
    df['emoji_count'] = stats.apply(lambda x: x['emoji_count'])
    df['exclamation_count'] = stats.apply(lambda x: x['exclamation_count'])
    df['question_count'] = stats.apply(lambda x: x['question_count'])
    
    # Afficher quelques stats
    print(f"   ✓ Longueur moyenne : {df['text_length'].mean():.1f} caractères")
    print(f"   ✓ Nombre moyen de mots : {df['word_count'].mean():.1f}")
    print(f"   ✓ Textes avec emojis : {(df['emoji_count'] > 0).sum()} ({(df['emoji_count'] > 0).sum() / len(df) * 100:.1f}%)")
    
    return df


# =============================================================================
# TOKENIZATION CAMEMBERT (à venir)
# =============================================================================

def tokenize_text_camembert(text: str, tokenizer=None, max_length: int = 128):
    """
    Tokenize un texte avec le tokenizer de CamemBERT.
    
    Args:
        text: Texte à tokenizer
        tokenizer: Tokenizer CamemBERT (de transformers)
        max_length: Longueur maximale des séquences
    
    Returns:
        Tokens encodés
    """
    if tokenizer is None:
        # Charger le tokenizer CamemBERT
        try:
            from transformers import CamembertTokenizer
            tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        except ImportError:
            print("⚠️ Transformers non installé. Installez-le avec: pip install transformers")
            return None
    
    # Tokenizer le texte
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return encoded


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

if __name__ == "__main__":
    """
    Test des fonctions de preprocessing
    """
    print("=" * 70)
    print("TEST DU PREPROCESSING")
    print("=" * 70)
    
    # Exemples de textes
    test_texts = [
        "Regardez cette vidéo géniale ! http://example.com 😍",
        "@utilisateur Tu as vu le film ? C'est INCROYABLE !!!",
        "#cinema J'adore ce film, vraiment top ! 🎬❤️",
        "Bof... pas terrible 😐",
        "C'EST NUL !!! Je déteste ça 😡😡😡"
    ]
    
    print("\n📝 Nettoyage des textes :\n")
    for i, text in enumerate(test_texts, 1):
        cleaned = clean_text(text)
        stats = get_text_stats(cleaned)
        
        print(f"{i}. Original : {text}")
        print(f"   Nettoyé  : {cleaned}")
        print(f"   Stats    : {stats['word_count']} mots, {stats['emoji_count']} emojis, "
              f"{stats['exclamation_count']} !, {stats['question_count']} ?")
        print()
    
    print("=" * 70)
    print("✅ Tests terminés !")
    print("=" * 70)
