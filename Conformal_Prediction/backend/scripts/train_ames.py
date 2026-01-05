"""
Entraîne un modèle RandomForest + calibration MAPIE sur le dataset Ames Housing.
- Exclut les colonnes d'identifiant 'Order' et 'PID' des features.
- Supprime le DEBUG.
- Sauvegarde, en plus du modèle MAPIE, un mapping 'feature_types' (numeric/categorical).
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

# Import de la fonction centralisée depuis app.conformal
from app.conformal import extract_lower_upper as extract_lower_upper_from_mapie

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


def build_pipeline_and_types(df: pd.DataFrame, exclude_ids: list = None):
    """
    Construit la pipeline et renvoie (pipeline, feature_cols, feature_types)
    feature_types: dict feature_name -> 'numeric'|'categorical'|'unknown'
    """
    if exclude_ids is None:
        exclude_ids = ["Order", "PID"]

    # Exclure explicitement les colonnes d'ID si elles existent
    df_cols = df.columns.tolist()
    cols_to_drop = [c for c in exclude_ids if c in df_cols]
    if cols_to_drop:
        print(f"Excluding ID columns from features: {cols_to_drop}")

    X = df.drop(columns=["SalePrice"] + cols_to_drop, errors="ignore")
    # select numeric and categorical
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build simple transformers
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

    # feature order: numeric then categorical (original column names)
    feature_cols = numeric_cols + categorical_cols

    # feature types mapping
    feature_types = {}
    for c in feature_cols:
        if c in numeric_cols:
            feature_types[c] = "numeric"
        elif c in categorical_cols:
            feature_types[c] = "categorical"
        else:
            feature_types[c] = "unknown"

    return pipeline, feature_cols, feature_types





def main():
    print("Chargement du dataset Ames...")
    df = load_ames(DATA_PATH)

    if "SalePrice" not in df.columns:
        raise ValueError("Le CSV doit contenir la colonne cible 'SalePrice'.")

    # Construire pipeline et inférer types, en excluant Order & PID
    pipeline, feature_cols, feature_types = build_pipeline_and_types(df, exclude_ids=["Order", "PID"])

    # Séparer X/y en DataFrame/Series (important pour ColumnTransformer)
    X = df.drop(columns=["SalePrice"] + [c for c in ("Order", "PID") if c in df.columns], errors="ignore")
    y = df["SalePrice"]

    # Split train / calibration (on garde DataFrame pour X)
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entraînement du pipeline (préproc + RandomForest)...")
    pipeline.fit(X_train, y_train)

    print("Calibration MAPIE (cv='prefit', method='plus') sur l'ensemble de calibration...")
    mapie = MapieRegressor(pipeline, cv="prefit", method="plus")
    # Fournir DataFrame/Series (pas d'arrays) pour que ColumnTransformer sélectionne par nom
    mapie.fit(X_cal, y_cal)

    # Coverage empirique
    y_pred_cal, y_pis_cal = mapie.predict(X_cal, alpha=0.05)
    lower, upper = extract_lower_upper_from_mapie(y_pred_cal, y_pis_cal)
    coverage = float(np.mean((y_cal.values >= lower) & (y_cal.values <= upper)))
    print(f"Couverture empirique sur calibration (alpha=0.05): {coverage:.3f}")

    # Sauvegarde du modèle (même format que l'API attend) avec feature_types
    saved = {
        "model": mapie,
        "feature_names": feature_cols,  # liste des colonnes originales (X columns)
        "target_name": "SalePrice",
        "alpha_default": 0.05,
        "feature_types": feature_types,
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