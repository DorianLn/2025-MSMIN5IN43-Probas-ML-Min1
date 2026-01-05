"""
Data Loader for Marketing Mix Modeling

Ce module fournit des fonctions pour charger et valider les datasets MMM.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
import warnings


def load_csv_data(
    filepath: Union[str, Path],
    date_column: str = "date",
    target_column: str = "sales",
    media_columns: Optional[List[str]] = None,
    control_columns: Optional[List[str]] = None,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Charge un dataset MMM depuis un fichier CSV.
    
    Parameters
    ----------
    filepath : str or Path
        Chemin vers le fichier CSV.
    date_column : str, default='date'
        Nom de la colonne de dates.
    target_column : str, default='sales'
        Nom de la colonne cible (ventes).
    media_columns : list of str, optional
        Liste des colonnes de dépenses media. Si None, détection automatique.
    control_columns : list of str, optional
        Liste des colonnes de contrôle (tendances, saison, etc.).
    parse_dates : bool, default=True
        Si True, parse la colonne de dates.
    
    Returns
    -------
    pd.DataFrame
        Dataset chargé avec colonnes validées.
    
    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si les colonnes requises sont manquantes.
    
    Examples
    --------
    >>> df = load_csv_data('data/raw/marketing_data.csv')
    >>> print(df.columns)
    Index(['date', 'sales', 'tv_spend', 'facebook_spend', ...])
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé : {filepath}")
    
    # Charger le CSV
    df = pd.read_csv(filepath)
    
    # Parser les dates si demandé
    if parse_dates and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
    
    # Valider la présence des colonnes essentielles
    if date_column not in df.columns:
        raise ValueError(f"Colonne '{date_column}' manquante dans le dataset")
    
    if target_column not in df.columns:
        raise ValueError(f"Colonne cible '{target_column}' manquante dans le dataset")
    
    # Détecter les colonnes media si non spécifiées
    if media_columns is None:
        # Heuristique : colonnes contenant 'spend', 'cost', 'impression' (case insensitive)
        media_keywords = ['spend', 'cost', 'impression', 'budget']
        media_columns = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in media_keywords)
        ]
        
        if media_columns:
            print(f"✓ Colonnes media détectées automatiquement : {media_columns}")
    
    # Vérifier que les colonnes media existent
    if media_columns:
        missing_media = set(media_columns) - set(df.columns)
        if missing_media:
            raise ValueError(f"Colonnes media manquantes : {missing_media}")
    
    # Vérifier les colonnes de contrôle si spécifiées
    if control_columns:
        missing_control = set(control_columns) - set(df.columns)
        if missing_control:
            warnings.warn(f"Colonnes de contrôle manquantes : {missing_control}")
    
    return df


def validate_mmm_data(
    df: pd.DataFrame,
    target_column: str = "sales",
    media_columns: Optional[List[str]] = None,
    min_periods: int = 52
) -> Dict[str, any]:
    """
    Valide un dataset MMM et retourne un rapport de validation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset à valider.
    target_column : str, default='sales'
        Nom de la colonne cible.
    media_columns : list of str, optional
        Liste des colonnes media à valider.
    min_periods : int, default=52
        Nombre minimum de périodes recommandé (défaut = 1 an hebdomadaire).
    
    Returns
    -------
    dict
        Rapport de validation avec clés : 'valid', 'warnings', 'n_periods', etc.
    
    Examples
    --------
    >>> report = validate_mmm_data(df, media_columns=['tv_spend', 'facebook_spend'])
    >>> if report['valid']:
    ...     print("Dataset valide !")
    """
    report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'n_periods': len(df),
        'n_media_channels': len(media_columns) if media_columns else 0
    }
    
    # Vérifier le nombre de périodes
    if len(df) < min_periods:
        report['warnings'].append(
            f"Seulement {len(df)} périodes (recommandé : >= {min_periods})"
        )
    
    # Vérifier les valeurs manquantes
    if df[target_column].isna().any():
        n_missing = df[target_column].isna().sum()
        report['errors'].append(f"{n_missing} valeurs manquantes dans '{target_column}'")
        report['valid'] = False
    
    # Vérifier les valeurs négatives dans les ventes
    if (df[target_column] < 0).any():
        report['warnings'].append("Valeurs négatives détectées dans les ventes")
    
    # Vérifier les colonnes media
    if media_columns:
        for col in media_columns:
            if col not in df.columns:
                report['errors'].append(f"Colonne media '{col}' manquante")
                report['valid'] = False
                continue
            
            # Valeurs manquantes
            if df[col].isna().any():
                n_missing = df[col].isna().sum()
                report['warnings'].append(f"{n_missing} valeurs manquantes dans '{col}'")
            
            # Valeurs négatives
            if (df[col] < 0).any():
                report['warnings'].append(f"Valeurs négatives dans '{col}'")
            
            # Dépenses nulles
            if (df[col] == 0).all():
                report['warnings'].append(f"Toutes les dépenses sont nulles pour '{col}'")
    
    return report


def create_sample_data(
    n_periods: int = 104,
    n_media_channels: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Génère un dataset MMM synthétique pour tests et démonstrations.
    
    Le dataset généré simule :
    - Tendance croissante
    - Saisonnalité
    - Effet des dépenses media sur les ventes (avec adstock et saturation)
    - Bruit aléatoire
    
    Parameters
    ----------
    n_periods : int, default=104
        Nombre de périodes (défaut = 2 ans hebdomadaires).
    n_media_channels : int, default=3
        Nombre de canaux media.
    seed : int, default=42
        Seed pour la reproductibilité.
    
    Returns
    -------
    pd.DataFrame
        Dataset synthétique avec colonnes : date, sales, media_1_spend, ...
    
    Examples
    --------
    >>> df = create_sample_data(n_periods=52, n_media_channels=2)
    >>> df.head()
    """
    np.random.seed(seed)
    
    # Dates
    dates = pd.date_range(start='2022-01-01', periods=n_periods, freq='W')
    
    # Baseline des ventes (tendance + saisonnalité)
    trend = np.linspace(1000, 1500, n_periods)
    seasonality = 200 * np.sin(2 * np.pi * np.arange(n_periods) / 52)
    baseline = trend + seasonality
    
    # Générer les dépenses media
    media_data = {}
    media_contributions = np.zeros(n_periods)
    
    for i in range(n_media_channels):
        # Dépenses média avec variation
        mean_spend = np.random.uniform(50, 200)
        spend = np.maximum(0, mean_spend + np.random.normal(0, mean_spend * 0.3, n_periods))
        
        # Simuler l'effet avec adstock simplifié
        alpha = np.random.uniform(0.3, 0.7)
        adstocked_spend = spend.copy()
        for t in range(1, n_periods):
            adstocked_spend[t] += alpha * adstocked_spend[t-1]
        
        # Simuler la saturation
        k = mean_spend * 2
        saturated = adstocked_spend / (k + adstocked_spend)
        
        # Contribution aux ventes (coefficient aléatoire)
        coef = np.random.uniform(100, 300)
        media_contributions += coef * saturated
        
        media_data[f'media_{i+1}_spend'] = spend
    
    # Ventes totales = baseline + contributions media + bruit
    noise = np.random.normal(0, 50, n_periods)
    sales = baseline + media_contributions + noise
    sales = np.maximum(0, sales)  # Pas de ventes négatives
    
    # Construire le DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        **media_data
    })
    
    return df


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    date_column: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divise le dataset en train/test chronologiquement.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset à diviser.
    train_ratio : float, default=0.8
        Proportion des données pour l'entraînement.
    date_column : str, default='date'
        Colonne de dates pour trier.
    
    Returns
    -------
    tuple of pd.DataFrame
        (train_df, test_df)
    
    Examples
    --------
    >>> train, test = split_train_test(df, train_ratio=0.8)
    >>> print(f"Train: {len(train)} periods, Test: {len(test)} periods")
    """
    if date_column in df.columns:
        df = df.sort_values(date_column).reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Génère un résumé statistique du dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset à résumer.
    
    Returns
    -------
    dict
        Statistiques descriptives.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = {
        'n_periods': len(df),
        'n_columns': len(df.columns),
        'date_range': None,
        'numeric_stats': {}
    }
    
    # Plage de dates
    if 'date' in df.columns:
        summary['date_range'] = (df['date'].min(), df['date'].max())
    
    # Stats pour colonnes numériques
    for col in numeric_cols:
        summary['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'missing': df[col].isna().sum()
        }
    
    return summary
