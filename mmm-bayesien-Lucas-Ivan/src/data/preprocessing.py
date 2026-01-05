"""
Data Preprocessing for Marketing Mix Modeling

Fonctions pour nettoyer et préparer les données MMM.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'forward_fill',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans le dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset avec potentiellement des valeurs manquantes.
    method : str, default='forward_fill'
        Méthode : 'forward_fill', 'backward_fill', 'interpolate', 'drop', 'zero'.
    columns : list of str, optional
        Colonnes spécifiques à traiter. Si None, traite toutes les colonnes.
    
    Returns
    -------
    pd.DataFrame
        Dataset sans valeurs manquantes.
    """
    df = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if df[col].isna().any():
            if method == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
            elif method == 'backward_fill':
                df[col] = df[col].fillna(method='bfill')
            elif method == 'interpolate':
                df[col] = df[col].interpolate(method='linear')
            elif method == 'zero':
                df[col] = df[col].fillna(0)
            elif method == 'drop':
                df = df.dropna(subset=[col])
            else:
                raise ValueError(f"Méthode inconnue : {method}")
    
    return df


def normalize_features(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'standardize',
    fit_on: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalise les features numériques.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset à normaliser.
    columns : list of str
        Colonnes à normaliser.
    method : str, default='standardize'
        'standardize' (mean=0, std=1) ou 'minmax' (range [0,1]).
    fit_on : pd.DataFrame, optional
        Dataset pour fitter le scaler (ex: train). Si None, fit sur df.
    
    Returns
    -------
    tuple
        (df_normalized, scaler_info) où scaler_info contient les paramètres.
    """
    df = df.copy()
    
    if method == 'standardize':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Méthode inconnue : {method}")
    
    # Fit sur le dataset spécifié
    fit_data = fit_on if fit_on is not None else df
    scaler.fit(fit_data[columns])
    
    # Transform
    df[columns] = scaler.transform(df[columns])
    
    # Sauvegarder les paramètres
    scaler_info = {
        'method': method,
        'columns': columns,
        'scaler': scaler
    }
    
    return df, scaler_info


def add_time_features(
    df: pd.DataFrame,
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Ajoute des features temporelles (tendance, saisonnalité).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset avec colonne de dates.
    date_column : str, default='date'
        Nom de la colonne de dates.
    
    Returns
    -------
    pd.DataFrame
        Dataset avec features temporelles ajoutées.
    """
    df = df.copy()
    
    if date_column not in df.columns:
        raise ValueError(f"Colonne '{date_column}' manquante")
    
    # Convertir en datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Tendance linéaire
    df['trend'] = np.arange(len(df))
    
    # Saisonnalité (sin/cos pour capturer la périodicité annuelle)
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['seasonality_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['seasonality_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Mois (utile pour certains patterns)
    df['month'] = df[date_column].dt.month
    
    # Quarter
    df['quarter'] = df[date_column].dt.quarter
    
    return df


def create_lagged_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 4]
) -> pd.DataFrame:
    """
    Crée des features retardées (lagged features).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset source.
    columns : list of str
        Colonnes pour lesquelles créer des lags.
    lags : list of int, default=[1, 2, 4]
        Liste des retards à appliquer (en périodes).
    
    Returns
    -------
    pd.DataFrame
        Dataset avec features laggées ajoutées.
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Identifie et traite les outliers.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset à nettoyer.
    columns : list of str
        Colonnes à vérifier pour outliers.
    method : str, default='iqr'
        'iqr' (Interquartile Range) ou 'zscore'.
    threshold : float, default=1.5
        Seuil pour IQR (1.5) ou z-score (3.0).
    
    Returns
    -------
    pd.DataFrame
        Dataset avec outliers gérés (capped aux limites).
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Cap les valeurs aux bornes
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        else:
            raise ValueError(f"Méthode inconnue : {method}")
    
    return df


def prepare_mmm_data(
    df: pd.DataFrame,
    target_column: str = 'sales',
    media_columns: Optional[List[str]] = None,
    control_columns: Optional[List[str]] = None,
    add_time_vars: bool = True,
    normalize: bool = True,
    handle_missing: str = 'interpolate'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Pipeline complet de préparation des données MMM.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset brut.
    target_column : str, default='sales'
        Colonne cible.
    media_columns : list of str, optional
        Colonnes media. Si None, détection automatique.
    control_columns : list of str, optional
        Colonnes de contrôle.
    add_time_vars : bool, default=True
        Ajouter les variables temporelles.
    normalize : bool, default=True
        Normaliser les features.
    handle_missing : str, default='interpolate'
        Méthode pour gérer les valeurs manquantes.
    
    Returns
    -------
    tuple
        (df_prepared, metadata) avec métadonnées de préparation.
    """
    df = df.copy()
    metadata = {}
    
    # Détection automatique des colonnes media si nécessaire
    if media_columns is None:
        media_keywords = ['spend', 'cost', 'impression', 'budget']
        media_columns = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in media_keywords)
        ]
        metadata['media_columns_detected'] = media_columns
    
    # 1. Gérer les valeurs manquantes
    if handle_missing:
        df = handle_missing_values(df, method=handle_missing)
        metadata['missing_values_handled'] = handle_missing
    
    # 2. Ajouter les features temporelles
    if add_time_vars and 'date' in df.columns:
        df = add_time_features(df)
        metadata['time_features_added'] = True
    
    # 3. Normaliser les features
    if normalize:
        # Normaliser les dépenses media
        if media_columns:
            df, scaler_media = normalize_features(df, media_columns, method='standardize')
            metadata['media_scaler'] = scaler_media
        
        # Normaliser la cible (log-transform souvent utilisé pour les ventes)
        if target_column in df.columns:
            # Log-transform pour stabiliser la variance
            df[f'{target_column}_log'] = np.log1p(df[target_column])
            metadata['target_log_transformed'] = True
    
    metadata['n_periods'] = len(df)
    metadata['prepared_columns'] = list(df.columns)
    
    return df, metadata


def inverse_transform_predictions(
    predictions: np.ndarray,
    log_transformed: bool = False
) -> np.ndarray:
    """
    Inverse les transformations appliquées aux prédictions.
    
    Parameters
    ----------
    predictions : np.ndarray
        Prédictions du modèle (potentiellement transformées).
    log_transformed : bool, default=False
        Si True, applique exp(x) - 1 pour inverser log1p.
    
    Returns
    -------
    np.ndarray
        Prédictions dans l'échelle originale.
    """
    if log_transformed:
        return np.expm1(predictions)
    
    return predictions
