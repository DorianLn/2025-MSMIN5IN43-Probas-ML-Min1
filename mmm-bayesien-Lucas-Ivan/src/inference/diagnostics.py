"""
MCMC Diagnostics - Convergence Checks

Fonctions pour diagnostiquer la qualité de l'inférence MCMC.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import arviz as az
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def check_convergence(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None,
    r_hat_threshold: float = 1.01,
    ess_threshold: int = 400
) -> Dict:
    """
    Vérifie la convergence de l'inférence MCMC.
    
    Critères :
    - R-hat (Gelman-Rubin) < 1.01 : convergence des chaînes
    - ESS (Effective Sample Size) > 400 : échantillons indépendants suffisants
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_names : list of str, optional
        Variables à vérifier. Si None, toutes.
    r_hat_threshold : float, default=1.01
        Seuil pour R-hat.
    ess_threshold : int, default=400
        Seuil pour ESS.
    
    Returns
    -------
    dict
        Rapport de convergence avec warnings si problèmes.
    """
    # Calculer les diagnostics
    summary = az.summary(trace, var_names=var_names)
    
    # Vérifier R-hat
    r_hat_issues = summary[summary['r_hat'] > r_hat_threshold]
    
    # Vérifier ESS bulk (nombre d'échantillons effectifs)
    ess_bulk_issues = summary[summary['ess_bulk'] < ess_threshold]
    
    # Vérifier ESS tail (pour les queues de distribution)
    ess_tail_issues = summary[summary['ess_tail'] < ess_threshold]
    
    # Divergences
    n_divergences = trace.sample_stats['diverging'].sum().values
    
    report = {
        'converged': True,
        'warnings': [],
        'n_divergences': int(n_divergences),
        'r_hat_max': float(summary['r_hat'].max()),
        'ess_bulk_min': float(summary['ess_bulk'].min()),
        'ess_tail_min': float(summary['ess_tail'].min())
    }
    
    # Ajouter les warnings
    if len(r_hat_issues) > 0:
        report['converged'] = False
        report['warnings'].append(
            f"{len(r_hat_issues)} variable(s) avec R-hat > {r_hat_threshold}"
        )
    
    if len(ess_bulk_issues) > 0:
        report['warnings'].append(
            f"{len(ess_bulk_issues)} variable(s) avec ESS_bulk < {ess_threshold}"
        )
    
    if len(ess_tail_issues) > 0:
        report['warnings'].append(
            f"{len(ess_tail_issues)} variable(s) avec ESS_tail < {ess_threshold}"
        )
    
    if n_divergences > 0:
        report['warnings'].append(
            f"{n_divergences} divergences détectées"
        )
    
    return report


def get_diagnostic_summary(trace: az.InferenceData) -> pd.DataFrame:
    """
    Résumé complet des diagnostics MCMC.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    
    Returns
    -------
    pd.DataFrame
        Tableau avec tous les diagnostics.
    """
    summary = az.summary(trace)
    
    # Ajouter des colonnes d'alerte
    summary['r_hat_ok'] = summary['r_hat'] < 1.01
    summary['ess_ok'] = (summary['ess_bulk'] > 400) & (summary['ess_tail'] > 400)
    
    return summary


def posterior_predictive_check(
    trace: az.InferenceData,
    y_obs: np.ndarray,
    var_name: str = 'y_obs'
) -> Dict:
    """
    Posterior predictive check (PPC) pour vérifier la qualité du modèle.
    
    Compare les données observées aux données simulées par le modèle.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace avec posterior_predictive.
    y_obs : np.ndarray
        Données observées.
    var_name : str, default='y_obs'
        Nom de la variable observée dans le modèle.
    
    Returns
    -------
    dict
        Métriques de comparaison.
    """
    # Extraire les prédictions a posteriori
    if not hasattr(trace, 'posterior_predictive'):
        return {'error': 'Posterior predictive non disponible'}
    
    y_pred = trace.posterior_predictive[var_name].values
    
    # Calculer des statistiques
    y_pred_mean = y_pred.mean(axis=(0, 1))  # Moyenne sur chains et draws
    
    # Intervalles de crédibilité
    y_pred_lower = np.percentile(y_pred, 2.5, axis=(0, 1))
    y_pred_upper = np.percentile(y_pred, 97.5, axis=(0, 1))
    
    # Proportion d'observations dans l'intervalle
    in_interval = np.mean((y_obs >= y_pred_lower) & (y_obs <= y_pred_upper))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_obs - y_pred_mean) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_obs - y_pred_mean))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'coverage_95': in_interval,
        'y_pred_mean': y_pred_mean
    }


def check_divergences(trace: az.InferenceData) -> Dict:
    """
    Analyse détaillée des divergences MCMC.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    
    Returns
    -------
    dict
        Informations sur les divergences.
    """
    diverging = trace.sample_stats['diverging'].values
    n_total = diverging.size
    n_divergences = diverging.sum()
    
    report = {
        'n_divergences': int(n_divergences),
        'pct_divergences': float(n_divergences / n_total * 100),
        'has_divergences': n_divergences > 0
    }
    
    if n_divergences > 0:
        report['recommendation'] = (
            "Augmenter target_accept (ex: 0.99) ou "
            "reparamétrer le modèle"
        )
    
    return report


def compare_chains(trace: az.InferenceData, var_name: str) -> Dict:
    """
    Compare les différentes chaînes MCMC pour une variable.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_name : str
        Nom de la variable à analyser.
    
    Returns
    -------
    dict
        Statistiques par chaîne.
    """
    var_data = trace.posterior[var_name].values
    n_chains = var_data.shape[0]
    
    chains_stats = {}
    for i in range(n_chains):
        chain_data = var_data[i]
        chains_stats[f'chain_{i}'] = {
            'mean': float(chain_data.mean()),
            'std': float(chain_data.std()),
            'min': float(chain_data.min()),
            'max': float(chain_data.max())
        }
    
    return chains_stats


def get_effective_sample_size(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calcule l'ESS pour les variables d'intérêt.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_names : list of str, optional
        Variables à analyser.
    
    Returns
    -------
    pd.DataFrame
        ESS bulk et tail par variable.
    """
    summary = az.summary(trace, var_names=var_names)
    
    return summary[['ess_bulk', 'ess_tail']]
