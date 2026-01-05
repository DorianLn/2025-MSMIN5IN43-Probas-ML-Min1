"""
MCMC Sampler - Inference Module

Fonctions pour configurer et exécuter l'inférence MCMC.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import pymc as pm
import arviz as az
import numpy as np
from typing import Optional, Dict


def sample_model(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.95,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
    **kwargs
) -> az.InferenceData:
    """
    Échantillonne un modèle PyMC avec NUTS.
    
    Parameters
    ----------
    model : pm.Model
        Modèle PyMC à échantillonner.
    draws : int, default=2000
        Nombre d'échantillons après warm-up.
    tune : int, default=1000
        Nombre d'échantillons de warm-up (tuning).
    chains : int, default=4
        Nombre de chaînes MCMC indépendantes.
    target_accept : float, default=0.95
        Taux d'acceptation cible (0.8-0.99).
        Plus élevé = plus conservateur = moins de divergences.
    random_seed : int, optional
        Seed pour reproductibilité.
    progressbar : bool, default=True
        Afficher la barre de progression.
    **kwargs
        Arguments supplémentaires pour pm.sample.
    
    Returns
    -------
    az.InferenceData
        Trace avec échantillons a posteriori.
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=progressbar,
            **kwargs
        )
    
    return trace


def sample_prior_predictive(
    model: pm.Model,
    samples: int = 1000
) -> az.InferenceData:
    """
    Échantillonne la distribution a priori prédictive.
    
    Utile pour vérifier que les priors sont raisonnables avant le fitting.
    
    Parameters
    ----------
    model : pm.Model
        Modèle PyMC.
    samples : int, default=1000
        Nombre d'échantillons a priori.
    
    Returns
    -------
    az.InferenceData
        Échantillons a priori.
    """
    with model:
        prior_predictive = pm.sample_prior_predictive(samples=samples)
    
    return prior_predictive


def sample_posterior_predictive(
    trace: az.InferenceData,
    model: pm.Model,
    progressbar: bool = True
) -> az.InferenceData:
    """
    Échantillonne la distribution a posteriori prédictive.
    
    Permet de faire des posterior predictive checks et de générer
    des prédictions avec incertitude.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC de l'inférence.
    model : pm.Model
        Modèle PyMC.
    progressbar : bool, default=True
        Afficher la barre de progression.
    
    Returns
    -------
    az.InferenceData
        Échantillons a posteriori prédictifs.
    """
    with model:
        posterior_predictive = pm.sample_posterior_predictive(
            trace,
            progressbar=progressbar
        )
    
    return posterior_predictive


def get_sampling_recommendations(n_samples: int, n_params: int) -> Dict:
    """
    Recommande les paramètres de sampling selon la taille du problème.
    
    Parameters
    ----------
    n_samples : int
        Nombre d'observations.
    n_params : int
        Nombre de paramètres à estimer.
    
    Returns
    -------
    dict
        Configuration recommandée.
    """
    # Règles empiriques
    if n_samples < 50:
        draws = 3000
        tune = 1500
        chains = 4
    elif n_samples < 200:
        draws = 2000
        tune = 1000
        chains = 4
    else:
        draws = 1500
        tune = 1000
        chains = 4
    
    # Plus de paramètres = plus de samples
    if n_params > 20:
        draws = int(draws * 1.5)
        tune = int(tune * 1.5)
    
    return {
        'draws': draws,
        'tune': tune,
        'chains': chains,
        'target_accept': 0.95
    }
