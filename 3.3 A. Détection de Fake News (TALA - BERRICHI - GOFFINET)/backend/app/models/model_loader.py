import torch
from transformers import pipeline
import gc
from app.config import settings

# Variables globales pour conserver le modèle en mémoire (Cache)
# Cela évite de recharger le modèle à chaque phrase si c'est le même modèle
current_model_id = None
pipe = None

def get_prediction(text: str, model_type: str):
    """
    Charge le modèle demandé s'il n'est pas déjà actif, 
    libère la mémoire GPU si nécessaire, et effectue l'analyse.
    """
    global current_model_id, pipe
    
    # 1. Récupération du chemin Hugging Face via le dictionnaire dans config.py
    model_repo = settings.MODELS.get(model_type.lower())
    
    if not model_repo:
        raise ValueError(f"Le modèle '{model_type}' n'est pas configuré dans settings.MODELS")

    # 2. Gestion du changement de modèle pour économiser la VRAM (4Go de ta 3050 Ti)
    if current_model_id != model_repo:
        print(f"🔄 Changement de modèle détecté...")
        print(f"📥 Chargement de : {model_repo}")
        
        # On détruit l'ancien pipeline s'il existe
        pipe = None
        
        # Nettoyage forcé de la mémoire RAM et VRAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 VRAM libérée avec succès.")
            
        # Chargement du nouveau pipeline sur le périphérique défini (GPU ou CPU)
        try:
            pipe = pipeline(
                "text-classification", 
                model=model_repo, 
                device=settings.DEVICE  # Utilise 0 pour GPU, -1 pour CPU
            )
            current_model_id = model_repo
            print(f"✅ Modèle {model_type.upper()} prêt sur {'GPU' if settings.DEVICE == 0 else 'CPU'}")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
            raise

    # 3. Exécution de l'inférence
    # On limite le texte à 512 tokens pour respecter la limite native de BERT/RoBERTa
    try:
        # L'analyse est effectuée ici
        results = pipe(text[:512])
        
        # Le résultat est une liste, on prend le premier élément
        # Format attendu : {'label': 'LABEL_X', 'score': 0.99}
        prediction = results[0]
        
        print(f"🔍 Analyse terminée : {prediction['label']} (Confiance: {prediction['score']:.2%})")
        return prediction

    except Exception as e:
        print(f"❌ Erreur lors de l'inférence : {e}")
        raise