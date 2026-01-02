"""
API FastAPI minimale pour entraîner et consommer un modèle conformal (MAPIE).
Endpoints:
  - POST /train : upload CSV + target_col -> entraîne et sauvegarde un modèle MAPIE
  - POST /predict : charger un modèle existant et retourner intervalles
  - GET  /models : lister modèles sauvegardés et métadonnées
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import os
import joblib
from app.conformal import train_mapie_from_dataframe, load_model, predict_with_intervals, MODELS_DIR

app = FastAPI(title="Conformal Prediction API", version="0.2")

# CORS - en dev je permets tout, en production restreins allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(MODELS_DIR, exist_ok=True)


class TrainResponse(BaseModel):
    path: str
    feature_names: List[str]
    target_name: str
    coverage: float
    alpha_default: float


@app.post("/train", response_model=TrainResponse)
async def train_endpoint(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    model_name: Optional[str] = Form("rf_mapie"),
    calibration_size: Optional[float] = Form(0.2),
    alpha: Optional[float] = Form(0.05),
):
    """
    Entraîner à partir d'un CSV envoyé.
    Form fields:
      - file: fichier CSV (multipart/form-data)
      - target_col: nom de la colonne cible dans le CSV
      - model_name: (optionnel) nom sous lequel sauvegarder le modèle (models/<model_name>.joblib)
      - calibration_size: fraction utilisée pour calibration (ex: 0.2)
      - alpha: niveau de risque pour la couverture calculée (ex: 0.05)
    """
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire le CSV: {e}")

    try:
        meta = train_mapie_from_dataframe(
            df,
            target_col=target_col,
            model_name=model_name,
            calibration_size=calibration_size,
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'entraînement: {e}")

    return TrainResponse(**meta)


class PredictRequest(BaseModel):
    model_path: str  # chemin relatif ex: models/rf_mapie.joblib
    instances: List[Dict[str, Any]]  # liste de dicts {feature1: val, feature2: val, ...}
    alpha: Optional[float] = None


class PredictResponseItem(BaseModel):
    prediction: float
    lower: float
    upper: float


@app.post("/predict", response_model=List[PredictResponseItem])
def predict_endpoint(req: PredictRequest):
    """
    Prédire des intervalles pour les instances fournies.
    - model_path: chemin vers le joblib créé par /train, ex: models/rf_mapie.joblib
    - instances: liste de dict (même colonnes que celles du jeu d'entraînement)
    - alpha: optionnel, si non fourni on utilise alpha par défaut sauvegardé
    """
    if not os.path.exists(req.model_path):
        raise HTTPException(status_code=404, detail="model_path introuvable")

    try:
        model_obj = load_model(req.model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement modèle: {e}")

    try:
        X = pd.DataFrame(req.instances)
        results = predict_with_intervals(model_obj, X, alpha=req.alpha)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {e}")

    return results


@app.get("/models")
def list_models():
    """
    Lister tous les modèles joblib présents dans le dossier models/ avec quelques métadonnées.
    """
    models = []
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith(".joblib"):
            continue
        path = os.path.join(MODELS_DIR, fname)
        try:
            meta = joblib.load(path)
            models.append(
                {
                    "filename": fname,
                    "path": path,
                    "feature_names": meta.get("feature_names"),
                    "target_name": meta.get("target_name"),
                    "alpha_default": meta.get("alpha_default"),
                }
            )
        except Exception:
            models.append({"filename": fname, "path": path, "error": "unable to load metadata"})
    return models


@app.get("/")
def root():
    return {"msg": "Conformal Prediction API - endpoints: /train, /predict, /models"}