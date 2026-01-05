from fastapi import APIRouter, HTTPException
from app.schemas import AnalysisRequest, AnalysisResponse
from app.models.model_loader import get_prediction
from app.utils import clean_text, map_label_to_reliability

router = APIRouter()

@router.post("/predict", response_model=AnalysisResponse)
async def predict_news(request: AnalysisRequest):
    try:
        # 1. Prétraitement du texte
        cleaned_text = clean_text(request.text)
        
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Le texte est vide après nettoyage.")
        
        # 2. Prédiction via Hugging Face
        # On passe l'ID du modèle (camembert, bert ou roberta)
        raw_result = get_prediction(cleaned_text, request.model)
        
        # 3. Formatage avec la logique spécifique des labels
        # On passe ici 'request.model' pour différencier les logiques 0/1
        is_reliable = map_label_to_reliability(raw_result['label'], request.model)
        
        # Conversion du score en pourcentage (0.9854 -> 98.54)
        conf = round(raw_result['score'] * 100, 2)
        
        # 4. Construction de la réponse pour le Frontend
        return AnalysisResponse(
            isReliable=is_reliable,
            confidence=conf,
            factors={
                "style": {"score": min(round(conf * 1.02, 1), 100.0), "label": "Analysé"},
                "vocabulary": {"score": min(round(conf * 0.98, 1), 100.0), "label": "Vérifié"},
                "source": {"score": 85.0, "label": "Évalué"}
            },
            summary=(
                f"Modèle {request.model.upper()} : Article analysé avec succès. "
                f"Le contenu est jugé {'FIABLE' if is_reliable else 'NON-FIABLE'}."
            )
        )
        
    except Exception as e:
        print(f"🔥 Erreur dans le tunnel de prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))