"""
Prior Distributions for MMM Bayesian Model

Définit les distributions a priori pour les paramètres du modèle MMM.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import pymc as pm
import numpy as np
from typing import Dict, Optional, List


def get_default_priors(n_media_channels: int) -> Dict:
    """
    Retourne les priors par défaut pour un modèle MMM.
    
    Ces priors sont faiblement informatifs (weakly informative) basés sur
    les bonnes pratiques MMM.
    
    Parameters
    ----------
    n_media_channels : int
        Nombre de canaux media.
    
    Returns
    -------
    dict
        Configuration des priors.
    """
    return {
        'intercept': {
            'dist': 'Normal',
            'mu': 0,
            'sigma': 100
        },
        'media_coefficients': {
            'dist': 'HalfNormal',
            'sigma': 1.0,
            'shape': n_media_channels
        },
        'control_coefficients': {
            'dist': 'Normal',
            'mu': 0,
            'sigma': 1.0
        },
        'sigma': {
            'dist': 'HalfNormal',
            'sigma': 50
        }
    }


def get_informative_priors(
    n_media_channels: int,
    expected_roi: Optional[List[float]] = None
) -> Dict:
    """
    Retourne des priors informatifs basés sur des connaissances métier.
    
    Parameters
    ----------
    n_media_channels : int
        Nombre de canaux media.
    expected_roi : list of float, optional
        ROI attendus par canal (ex: [2.0, 1.5, 3.0] pour 200%, 150%, 300%).
    
    Returns
    -------
    dict
        Configuration des priors informatifs.
    """
    if expected_roi is None:
        # ROI par défaut basés sur benchmarks industrie
        expected_roi = [2.0] * n_media_channels
    
    return {
        'intercept': {
            'dist': 'Normal',
            'mu': 0,
            'sigma': 50
        },
        'media_coefficients': {
            'dist': 'LogNormal',
            'mu': np.log(expected_roi),
            'sigma': 0.5,
            'shape': n_media_channels
        },
        'control_coefficients': {
            'dist': 'Normal',
            'mu': 0,
            'sigma': 0.5
        },
        'sigma': {
            'dist': 'HalfNormal',
            'sigma': 25
        }
    }


def create_prior(prior_config: Dict, name: str) -> pm.Distribution:
    """
    Crée une distribution PyMC à partir de la configuration.
    
    Parameters
    ----------
    prior_config : dict
        Configuration du prior (dist, paramètres).
    name : str
        Nom de la variable dans le modèle PyMC.
    
    Returns
    -------
    pm.Distribution
        Distribution PyMC.
    """
    dist_name = prior_config['dist']
    params = {k: v for k, v in prior_config.items() if k != 'dist'}
    
    if dist_name == 'Normal':
        return pm.Normal(name, **params)
    elif dist_name == 'HalfNormal':
        return pm.HalfNormal(name, **params)
    elif dist_name == 'LogNormal':
        return pm.LogNormal(name, **params)
    elif dist_name == 'Beta':
        return pm.Beta(name, **params)
    elif dist_name == 'Gamma':
        return pm.Gamma(name, **params)
    else:
        raise ValueError(f"Distribution inconnue : {dist_name}")


def get_adstock_priors() -> Dict:
    """
    Retourne les priors pour les paramètres d'adstock.
    
    Returns
    -------
    dict
        Configuration des priors pour alpha (taux de rétention).
    """
    return {
        'alpha': {
            'dist': 'Beta',
            'alpha': 2,
            'beta': 2,
            'testval': 0.5  # Valeur initiale pour sampling
        }
    }


def get_saturation_priors(mean_spend: Optional[float] = None) -> Dict:
    """
    Retourne les priors pour les paramètres de saturation.
    
    Parameters
    ----------
    mean_spend : float, optional
        Dépense moyenne pour calibrer half_saturation.
    
    Returns
    -------
    dict
        Configuration des priors pour k (half_saturation) et s (slope).
    """
    if mean_spend is None:
        mean_spend = 100  # Valeur par défaut
    
    return {
        'half_saturation': {
            'dist': 'Gamma',
            'alpha': 2.0,
            'beta': 2.0 / mean_spend,  # Centré autour de mean_spend
            'testval': mean_spend
        },
        'slope': {
            'dist': 'Gamma',
            'alpha': 2.0,
            'beta': 2.0,  # Centré autour de 1.0
            'testval': 1.0
        }
    }


def validate_priors(prior_config: Dict, n_media_channels: int) -> bool:
    """
    Valide la configuration des priors.
    
    Parameters
    ----------
    prior_config : dict
        Configuration à valider.
    n_media_channels : int
        Nombre attendu de canaux media.
    
    Returns
    -------
    bool
        True si valide.
    
    Raises
    ------
    ValueError
        Si la configuration est invalide.
    """
    required_keys = ['intercept', 'media_coefficients', 'sigma']
    
    for key in required_keys:
        if key not in prior_config:
            raise ValueError(f"Clé manquante dans prior_config : {key}")
    
    # Vérifier que media_coefficients a la bonne shape
    media_prior = prior_config['media_coefficients']
    if 'shape' in media_prior:
        if media_prior['shape'] != n_media_channels:
            raise ValueError(
                f"Shape incorrecte pour media_coefficients : "
                f"{media_prior['shape']} != {n_media_channels}"
            )
    
    return True
