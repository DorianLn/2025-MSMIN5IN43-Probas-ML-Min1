"""
API FastAPI pour modèles MAPIE.
Endpoints:
  - POST /train
  - POST /predict
  - GET  /models
  - GET  /models/{model_filename}/features
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import os
import joblib
from app.conformal import train_mapie_from_dataframe, load_model, predict_with_intervals, MODELS_DIR

app = FastAPI(title="Conformal Prediction API", version="0.4")

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

# Entraîne un modèle depuis un CSV uploadé et retourne les métadonnées


@app.post("/train", response_model=TrainResponse)
async def train_endpoint(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    model_name: Optional[str] = Form("rf_mapie"),
    calibration_size: Optional[float] = Form(0.2),
    alpha: Optional[float] = Form(0.05),
):
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Impossible de lire le CSV: {e}")

    try:
        meta = train_mapie_from_dataframe(
            df,
            target_col=target_col,
            model_name=model_name,
            calibration_size=calibration_size,
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de l'entraînement: {e}")

    return TrainResponse(**meta)


class PredictRequest(BaseModel):
    # maintenant on met un nom de fichier de modèle (ex: "ames_rf_mapie.joblib")
    model_filename: str
    instances: List[Dict[str, Any]]
    alpha: Optional[float] = None


class PredictResponseItem(BaseModel):
    prediction: float
    lower: float
    upper: float


class PredictSingleRequest(BaseModel):
    alpha: float
    features: Dict[str, Any]


# Prédit pour un batch d'instances et renvoie prediction + intervalle pour chaque instance


@app.post("/predict", response_model=List[PredictResponseItem])
def predict_endpoint(req: PredictRequest):
    path = os.path.join(MODELS_DIR, req.model_filename)
    try:
        model_obj = load_model(path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="model_filename introuvable")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur chargement modèle: {e}")

    try:
        X = pd.DataFrame(req.instances)
        results = predict_with_intervals(model_obj, X, alpha=req.alpha)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erreur lors de la prédiction: {e}")

    return results


@app.post("/predict_single", response_model=PredictResponseItem)
def predict_single_endpoint(req: PredictSingleRequest):
    # model_filename = "ames_rf_mapie.joblib"
    model_filename = "ames_gb_mapie.joblib"
    path = os.path.join(MODELS_DIR, model_filename)
    try:
        model_obj = load_model(path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Modèle par défaut '{model_filename}' introuvable.")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur chargement modèle: {e}")

    try:
        # Le modèle attend un DataFrame, même pour une seule instance
        X = pd.DataFrame([req.features])
        # Les colonnes numériques envoyées en JSON peuvent être des strings, on les convertit
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.to_numeric(X[col], errors='ignore')

        results = predict_with_intervals(model_obj, X, alpha=req.alpha)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erreur lors de la prédiction: {e}")

    if not results:
        raise HTTPException(
            status_code=500, detail="La prédiction n'a retourné aucun résultat.")

    return results[0]


# Liste les modèles sauvegardés avec leurs métadonnées


@app.get("/models")
def list_models():
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
                    "feature_types": meta.get("feature_types"),  # ajouté
                    "target_name": meta.get("target_name"),
                    "alpha_default": meta.get("alpha_default"),
                }
            )
        except Exception:
            models.append({"filename": fname, "path": path,
                          "error": "unable to load metadata"})
    return models

# Retourne la liste des features et leur type estimé pour un modèle donné


@app.get("/models/{model_filename}/features")
def model_features(model_filename: str):
    path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model file not found")

    try:
        saved = joblib.load(path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Impossible de charger le modèle: {e}")

    feature_names = saved.get("feature_names", [])
    feature_types_saved = saved.get("feature_types")
    if feature_types_saved:
        return [{"name": fn, "type": feature_types_saved.get(fn, "unknown")} for fn in feature_names]

    # fallback: best-effort inference (existing logic omitted for brevity)
    result = [{"name": fn, "type": "unknown"} for fn in feature_names]
    return result

# Page d'accueil minimaliste listant les endpoints


@app.get("/")
def root():
    return {"msg": "Conformal Prediction API - endpoints: /train, /predict, /models, /models/{model_filename}/features"}
