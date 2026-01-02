# app/conformal.py
"""
Utilities pour entraîner un modèle (RandomForest par défaut) et calibrer avec MAPIE,
puis prédire des intervalles conformes.

Usage principal :
  - train_mapie_from_dataframe(df, target_col, model_name=...)
  - load_model(path)
  - predict_with_intervals(model_obj, X_df, alpha=None)
"""
from typing import Dict, Any, List, Tuple
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Import direct (ta version mapie 0.7.0 expose MapieRegressor ici)
from mapie.regression import MapieRegressor

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _extract_lower_upper(y_pred: np.ndarray, y_pis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extraire lower et upper à partir de y_pis retourné par MAPIE.
    Supporte plusieurs formats possibles de y_pis :
      - (n_samples, n_alpha, 2)  -> lower = y_pis[:,0,0], upper = y_pis[:,0,1]
      - (n_samples, 2)           -> lower = y_pis[:,0], upper = y_pis[:,1]
      - (n_samples, n_alpha, 1)  -> on interprète comme delta -> pred +/- delta
      - (n_samples, 1)           -> idem
    Retourne (lower, upper) numpy arrays.
    """
    arr = np.asarray(y_pis)
    # debug shape if needed:
    # print("DEBUG _extract_lower_upper shape:", arr.shape)
    if arr.ndim == 3:
        n_samples, n_alpha, last = arr.shape
        if last == 2:
            lower = arr[:, 0, 0]
            upper = arr[:, 0, 1]
            return lower, upper
        if last == 1:
            # if n_alpha == 2 and last==1, squeeze then treat as (n_samples,2)
            if n_alpha == 2:
                squeezed = arr.squeeze(axis=2)  # (n_samples,2)
                lower = squeezed[:, 0]
                upper = squeezed[:, 1]
                return lower, upper
            # otherwise interpret as delta
            delta = arr[:, 0, 0]
            lower = y_pred - delta
            upper = y_pred + delta
            return lower, upper
        raise RuntimeError(f"Format inattendu de y_pis (ndim==3) shape={arr.shape}")

    if arr.ndim == 2:
        if arr.shape[1] == 2:
            lower = arr[:, 0]
            upper = arr[:, 1]
            return lower, upper
        if arr.shape[1] == 1:
            delta = arr[:, 0]
            lower = y_pred - delta
            upper = y_pred + delta
            return lower, upper
        raise RuntimeError(f"Format inattendu de y_pis (ndim==2) shape={arr.shape}")

    raise RuntimeError(f"Format inattendu de y_pis (ndim={arr.ndim})")


def train_mapie_from_dataframe(
    df: pd.DataFrame,
    target_col: str,
    model_name: str = "rf_mapie",
    rf_kwargs: Dict[str, Any] = None,
    calibration_size: float = 0.2,
    random_state: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Entraîne un RandomForest sur une portion d'entraînement, calibre avec MAPIE
    sur une portion de calibration (cv='prefit', method='plus'), sauvegarde et retourne les métadonnées.

    Retour:
      dict contenant:
        - 'path': chemin du fichier joblib sauvegardé
        - 'feature_names': liste des colonnes/features
        - 'target_name': nom de la target
        - 'coverage': couverture empirique sur l'ensemble de calibration (float)
        - 'alpha_default': alpha sauvegardé
    """
    if rf_kwargs is None:
        rf_kwargs = {"n_estimators": 100, "n_jobs": -1}

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' introuvable dans le DataFrame")

    X = df.drop(columns=[target_col]).reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)

    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=calibration_size, random_state=random_state
    )

    base = RandomForestRegressor(random_state=random_state, **rf_kwargs)
    base.fit(X_train, y_train)

    # MAPIE en mode 'prefit'
    mapie = MapieRegressor(base, cv="prefit", method="plus")
    # Fournir DataFrame/Series si base est pipeline qui sélectionne par nom
    try:
        mapie.fit(X_cal, y_cal)
    except Exception:
        # fallback to numpy arrays if necessary
        mapie.fit(X_cal.values, y_cal.values)

    # calcul coverage empirique
    try:
        y_pred_cal, y_pis_cal = mapie.predict(X_cal, alpha=alpha)
    except Exception:
        y_pred_cal, y_pis_cal = mapie.predict(X_cal.values, alpha=alpha)

    lower, upper = _extract_lower_upper(y_pred_cal, y_pis_cal)
    coverage = float(np.mean((y_cal.values >= lower) & (y_cal.values <= upper)))

    saved = {
        "model": mapie,
        "feature_names": X.columns.tolist(),
        "target_name": target_col,
        "alpha_default": alpha,
    }
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(saved, model_path)

    return {
        "path": model_path,
        "feature_names": saved["feature_names"],
        "target_name": target_col,
        "coverage": coverage,
        "alpha_default": alpha,
    }


def load_model(model_path: str) -> Dict[str, Any]:
    """Charge le dict sauvegardé contenant 'model' (Mapie) et metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_with_intervals(
    model_obj: Dict[str, Any], X: pd.DataFrame, alpha: float = None
) -> List[Dict[str, float]]:
    """
    Prédit valeurs ponctuelles + intervalles via l'objet MAPIE chargé.
    - model_obj: dict retourné par load_model (contenant 'model' et 'feature_names')
    - X: DataFrame d'entrée (colonnes doivent correspondre à feature_names)
    - alpha: niveau de risque (ex: 0.05 pour 95% intervalle). Si None -> alpha_default sauvegardé.
    Retour: liste de dicts {prediction, lower, upper}
    """
    mapie = model_obj["model"]
    feature_names = model_obj["feature_names"]

    if alpha is None:
        alpha = float(model_obj.get("alpha_default", 0.05))

    # Vérifier colonnes
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans les instances: {missing}")

    X_ordered = X[feature_names].reset_index(drop=True)

    # Essayer DataFrame, sinon fallback arrays
    try:
        y_pred, y_pis = mapie.predict(X_ordered, alpha=alpha)
    except Exception:
        y_pred, y_pis = mapie.predict(X_ordered.values, alpha=alpha)

    lower, upper = _extract_lower_upper(y_pred, y_pis)

    results = []
    for pred, lo, up in zip(y_pred, lower, upper):
        results.append({"prediction": float(pred), "lower": float(lo), "upper": float(up)})

    return results