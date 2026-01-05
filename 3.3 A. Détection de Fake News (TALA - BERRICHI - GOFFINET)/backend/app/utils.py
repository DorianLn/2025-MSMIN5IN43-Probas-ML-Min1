import re

def clean_text(text: str) -> str:
    """Nettoie le texte avant l'inférence."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_label_to_reliability(label: str, model_type: str) -> bool:
    """
    Traduit le label selon la logique spécifique de chaque modèle :
    - BERT/RoBERTa : 1 = True, 0 = Fake
    - CamemBERT : 0 = True, 1 = Fake
    """
    label_id = "1" if "1" in label else "0"
    
    if model_type.lower() == "camembert":
        # Logique CamemBERT : 0 est True
        return True if label_id == "0" else False
    else:
        # Logique BERT et RoBERTa : 1 est True
        return True if label_id == "1" else False