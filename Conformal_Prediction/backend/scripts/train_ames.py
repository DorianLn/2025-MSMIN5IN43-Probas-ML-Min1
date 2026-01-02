# scripts/train_ames.py
"""
Entraîne un modèle RandomForest + calibration MAPIE sur le dataset Ames Housing.
Exige : avoir placé le CSV Ames (par ex. 'data/ames.csv') avec la colonne cible 'SalePrice'.
Usage:
    python scripts/train_ames.py
"""
import sys
from pathlib import Path
import os

# Permettre l'import du package 'app' (backend) depuis le parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np

# Import MAPIE de façon résiliente
try:
    from mapie.regression import MapieRegressor
except Exception:
    try:
        from mapie.regression.mapie_regressor import MapieRegressor
    except Exception as exc:
        raise ImportError(
            "Impossible d'importer MapieRegressor depuis le paquet 'mapie'. "
            "Vérifie que 'mapie' est installé et compatible (ex: pip install mapie==0.7.0). "
            f"Détails: {exc}"
        )

DATA_PATH = Path("data/ames.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_ames(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV Ames non trouvé : {csv_path}. Place le fichier et relance.")
    # Lecture prudente (gestion encodage)
    df = pd.read_csv(csv_path)
    return df


def make_onehot_encoder_compat():
    """
    Retourne un OneHotEncoder compatible avec la version installée de scikit-learn.
    Certaines versions acceptent `sparse=False`, d'autres `sparse_output=False`.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_pipeline(df: pd.DataFrame):
    # Identifier colonnes (exclure explicitement la cible)
    X = df.drop(columns=["SalePrice"])
    # Utiliser select_dtypes pour couvrir plusieurs types numériques
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Transformers
    num_tf = SimpleImputer(strategy="median")
    ohe = make_onehot_encoder_compat()
    cat_tf = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("ohe", ohe)
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0  # forcer la sortie dense
    )

    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)

    pipeline = Pipeline(steps=[("preproc", preproc), ("model", rf)])
    # feature_cols = noms originaux des features (ordre : numériques puis catégoriques)
    feature_cols = numeric_cols + categorical_cols
    return pipeline, feature_cols


def extract_lower_upper_from_mapie_y_pis(y_pred: np.ndarray, y_pis: np.ndarray):
    """
    Fonction robuste pour extraire lower et upper à partir de y_pis retourné par MAPIE.
    gère plusieurs formats de sortie :
      - (n_samples, n_alpha, 2)  -> lower = y_pis[:,0,0], upper = y_pis[:,0,1]
      - (n_samples, 2)           -> lower = y_pis[:,0], upper = y_pis[:,1]
      - (n_samples, n_alpha, 1)  -> on interprète la valeur comme delta et on fait pred +/- delta
      - (n_samples, 1)           -> idem
      - (n_samples, 2, 1)        -> squeeze last dim -> (n_samples,2)
    Retour: lower, upper (np.ndarray)
    """
    y_pis = np.asarray(y_pis)
    print(f"DEBUG: y_pis.shape = {y_pis.shape}")

    if y_pis.ndim == 3:
        n_samples, n_alpha, last = y_pis.shape
        # cas attendu : last == 2
        if last == 2:
            lower = y_pis[:, 0, 0]
            upper = y_pis[:, 0, 1]
            return lower, upper
        # cas (n_samples, 2, 1) ou similaire
        if last == 1 and n_alpha == 2:
            arr = y_pis.squeeze(axis=2)  # shape (n_samples, 2)
            lower = arr[:, 0]
            upper = arr[:, 1]
            return lower, upper
        # cas (n_samples, n_alpha, 1) -> on utilise delta = y_pis[:,0,0] comme demi-largeur
        if last == 1:
            delta = y_pis[:, 0, 0]
            lower = y_pred - delta
            upper = y_pred + delta
            return lower, upper
        raise RuntimeError(f"Format inattendu y_pis (ndim==3) : shape={y_pis.shape}")

    elif y_pis.ndim == 2:
        # cas fréquent (n_samples, 2)
        if y_pis.shape[1] == 2:
            lower = y_pis[:, 0]
            upper = y_pis[:, 1]
            return lower, upper
        # cas (n_samples, 1) -> delta
        if y_pis.shape[1] == 1:
            delta = y_pis[:, 0]
            lower = y_pred - delta
            upper = y_pred + delta
            return lower, upper
        raise RuntimeError(f"Format inattendu y_pis (ndim==2) : shape={y_pis.shape}")

    else:
        raise RuntimeError(f"Format inattendu y_pis (ndim={y_pis.ndim})")


def main():
    print("Chargement du dataset Ames...")
    df = load_ames(DATA_PATH)

    if "SalePrice" not in df.columns:
        raise ValueError("Le CSV doit contenir la colonne cible 'SalePrice'.")

    # Séparer X/y en DataFrame/Series (important pour ColumnTransformer)
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # Split train / calibration (on garde DataFrame pour X)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construire pipeline
    pipeline, feature_cols = build_pipeline(df)

    print("Entraînement du pipeline (préproc + RandomForest)...")
    pipeline.fit(X_train, y_train)

    print("Calibration MAPIE (cv='prefit', method='plus') sur l'ensemble de calibration...")
    mapie = MapieRegressor(pipeline, cv="prefit", method="plus")
    # Fournir DataFrame/Series (pas d'arrays) pour que ColumnTransformer sélectionne par nom
    mapie.fit(X_cal, y_cal)

    # Coverage empirique
    y_pred_cal, y_pis_cal = mapie.predict(X_cal, alpha=0.05)

    # Extraire bornes de manière robuste
    try:
        lower, upper = extract_lower_upper_from_mapie_y_pis(y_pred_cal, y_pis_cal)
    except RuntimeError as e:
        # Afficher diagnostic détaillé puis ré-élever
        print(f"Erreur lors de l'extraction des intervalles MAPIE: {e}")
        print("Types/shape debug :")
        print(" - type(y_pred_cal):", type(y_pred_cal), "shape:", getattr(y_pred_cal, "shape", None))
        print(" - type(y_pis_cal):", type(y_pis_cal), "shape:", getattr(np.asarray(y_pis_cal), "shape", None))
        raise

    coverage = float(np.mean((y_cal.values >= lower) & (y_cal.values <= upper)))
    print(f"Couverture empirique sur calibration (alpha=0.05): {coverage:.3f}")

    # Sauvegarde du modèle (même format que l'API attend)
    saved = {
        "model": mapie,
        "feature_names": feature_cols,  # liste des colonnes originales (X columns)
        "target_name": "SalePrice",
        "alpha_default": 0.05,
    }
    model_path = MODELS_DIR / "ames_rf_mapie.joblib"
    joblib.dump(saved, model_path)
    print(f"Modèle sauvegardé: {model_path}")

    # Résumé
    print("Résumé :")
    print(" - Nombre de features (originales) :", len(feature_cols))
    print(" - Exemples de features :", feature_cols[:10])
    print(" - Chemin modèle :", model_path)


if __name__ == "__main__":
    main()