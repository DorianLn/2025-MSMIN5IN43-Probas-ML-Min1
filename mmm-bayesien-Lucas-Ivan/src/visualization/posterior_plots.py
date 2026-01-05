"""
Posterior Plots - Bayesian Analysis Visualizations

Visualisations pour l'analyse des distributions a posteriori.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from typing import Optional, List, Tuple


def plot_trace(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Trace plots pour vérifier la convergence MCMC.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_names : list of str, optional
        Variables à visualiser.
    figsize : tuple, default=(14, 10)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec trace plots.
    """
    if var_names is None:
        var_names = ['intercept', 'beta_media', 'sigma']
    
    axes = az.plot_trace(
        trace,
        var_names=var_names,
        figsize=figsize,
        rug=True
    )
    
    plt.suptitle('Trace Plots - Convergence MCMC', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return axes.ravel()[0].figure


def plot_posterior(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None,
    hdi_prob: float = 0.95,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Distributions a posteriori avec intervalles de crédibilité.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_names : list of str, optional
        Variables à visualiser.
    hdi_prob : float, default=0.95
        Probabilité pour l'intervalle HDI.
    figsize : tuple, default=(14, 8)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec distributions posteriors.
    """
    if var_names is None:
        var_names = ['intercept', 'beta_media', 'sigma']
    
    axes = az.plot_posterior(
        trace,
        var_names=var_names,
        hdi_prob=hdi_prob,
        figsize=figsize,
        textsize=10
    )
    
    plt.suptitle(f'Distributions a posteriori (HDI {hdi_prob*100:.0f}%)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return axes.ravel()[0].figure if hasattr(axes, 'ravel') else axes.figure


def plot_forest(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None,
    hdi_prob: float = 0.95,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Forest plot pour comparer les paramètres.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_names : list of str, optional
        Variables à visualiser.
    hdi_prob : float, default=0.95
        Probabilité pour l'intervalle HDI.
    figsize : tuple, default=(10, 8)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec forest plot.
    """
    if var_names is None:
        var_names = ['beta_media']
    
    axes = az.plot_forest(
        trace,
        var_names=var_names,
        hdi_prob=hdi_prob,
        figsize=figsize,
        combined=True
    )
    
    plt.title('Forest Plot - Coefficients Media',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Valeur du paramètre', fontsize=11)
    plt.tight_layout()
    
    # Gérer le cas où axes est un array ou un seul axe
    if isinstance(axes, np.ndarray):
        return axes.ravel()[0].figure
    else:
        return axes.figure if hasattr(axes, 'figure') else plt.gcf()


def plot_pairplot(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 12)
) -> plt.Figure:
    """
    Pairplot pour visualiser les corrélations entre paramètres.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_names : list of str, optional
        Variables à visualiser.
    figsize : tuple, default=(12, 12)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec pairplot.
    """
    if var_names is None:
        var_names = ['intercept', 'beta_media', 'sigma']
    
    axes = az.plot_pair(
        trace,
        var_names=var_names,
        figsize=figsize,
        divergences=True
    )
    
    plt.suptitle('Pairplot - Corrélations entre paramètres',
                 fontsize=14, fontweight='bold', y=0.995)
    
    return axes.ravel()[0].figure


def plot_energy(
    trace: az.InferenceData,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Energy plot pour diagnostiquer les problèmes de géométrie.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    figsize : tuple, default=(10, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec energy plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    az.plot_energy(trace, ax=ax)
    
    ax.set_title('Energy Plot - Diagnostic MCMC',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_rank(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Rank plots pour détecter les problèmes de convergence.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_names : list of str, optional
        Variables à visualiser.
    figsize : tuple, default=(14, 8)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec rank plots.
    """
    if var_names is None:
        var_names = ['intercept', 'beta_media', 'sigma']
    
    axes = az.plot_rank(
        trace,
        var_names=var_names,
        figsize=figsize
    )
    
    plt.suptitle('Rank Plots - Distribution uniforme = bonne convergence',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return axes.ravel()[0].figure


def plot_autocorr(
    trace: az.InferenceData,
    var_names: Optional[List[str]] = None,
    max_lag: int = 100,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Autocorrélation pour vérifier l'indépendance des échantillons.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    var_names : list of str, optional
        Variables à visualiser.
    max_lag : int, default=100
        Lag maximum.
    figsize : tuple, default=(14, 8)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec autocorrélations.
    """
    if var_names is None:
        var_names = ['beta_media']
    
    axes = az.plot_autocorr(
        trace,
        var_names=var_names,
        max_lag=max_lag,
        figsize=figsize
    )
    
    plt.suptitle('Autocorrélation - Décroissance rapide = bonne indépendance',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return axes.ravel()[0].figure


def create_diagnostic_report(
    trace: az.InferenceData,
    save_path: Optional[str] = None
) -> dict:
    """
    Génère un rapport de diagnostic complet.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    save_path : str, optional
        Chemin pour sauvegarder les figures.
    
    Returns
    -------
    dict
        Dictionnaire {nom: figure} de tous les graphiques.
    """
    figures = {}
    
    # 1. Trace plots
    figures['trace'] = plot_trace(trace)
    
    # 2. Posteriors
    figures['posterior'] = plot_posterior(trace)
    
    # 3. Forest plot
    figures['forest'] = plot_forest(trace)
    
    # 4. Energy
    figures['energy'] = plot_energy(trace)
    
    # 5. Rank
    figures['rank'] = plot_rank(trace)
    
    # Sauvegarder si demandé
    if save_path:
        for name, fig in figures.items():
            fig.savefig(f'{save_path}/diagnostic_{name}.png', 
                       dpi=300, bbox_inches='tight')
    
    return figures
