"""
Optimization Plots

Visualisations pour l'optimisation budgétaire.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict


def plot_budget_comparison(
    current_spend: np.ndarray,
    optimal_spend: np.ndarray,
    channel_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Compare l'allocation actuelle vs optimale.
    
    Parameters
    ----------
    current_spend : np.ndarray
        Dépenses actuelles par canal.
    optimal_spend : np.ndarray
        Dépenses optimales par canal.
    channel_names : list of str, optional
        Noms des canaux.
    figsize : tuple, default=(12, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec comparaison.
    """
    n_channels = len(current_spend)
    
    if channel_names is None:
        channel_names = [f'Canal {i+1}' for i in range(n_channels)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_channels)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, current_spend, width, label='Actuel', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimal_spend, width, label='Optimal', alpha=0.8)
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}€',
                   ha='center', va='bottom', fontsize=9)
    
    # Calculer le changement en %
    changes = ((optimal_spend - current_spend) / current_spend * 100)
    
    for i, change in enumerate(changes):
        color = 'green' if change > 0 else 'red'
        ax.text(i, max(current_spend[i], optimal_spend[i]) * 1.15,
               f'{change:+.1f}%',
               ha='center', color=color, fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Canal', fontsize=12)
    ax.set_ylabel('Budget (€)', fontsize=12)
    ax.set_title('Allocation budgétaire : Actuelle vs Optimale',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_marginal_roi(
    spend_range: np.ndarray,
    marginal_rois: np.ndarray,
    current_spend: Optional[float] = None,
    optimal_spend: Optional[float] = None,
    channel_name: str = 'Canal',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Visualise le ROI marginal en fonction des dépenses.
    
    Parameters
    ----------
    spend_range : np.ndarray
        Plage de dépenses.
    marginal_rois : np.ndarray
        ROI marginal correspondant.
    current_spend : float, optional
        Dépenses actuelles (pour marqueur).
    optimal_spend : float, optional
        Dépenses optimales (pour marqueur).
    channel_name : str, default='Canal'
        Nom du canal.
    figsize : tuple, default=(10, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec courbe de ROI marginal.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(spend_range, marginal_rois, linewidth=2.5, color='steelblue')
    
    # Marquer les points clés
    if current_spend is not None:
        idx_current = np.argmin(np.abs(spend_range - current_spend))
        ax.scatter(current_spend, marginal_rois[idx_current],
                  s=150, color='orange', zorder=5, label='Actuel')
        ax.axvline(current_spend, color='orange', linestyle='--', alpha=0.5)
    
    if optimal_spend is not None:
        idx_optimal = np.argmin(np.abs(spend_range - optimal_spend))
        ax.scatter(optimal_spend, marginal_rois[idx_optimal],
                  s=150, color='green', zorder=5, label='Optimal')
        ax.axvline(optimal_spend, color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Dépenses (€)', fontsize=12)
    ax.set_ylabel('ROI Marginal', fontsize=12)
    ax.set_title(f'ROI Marginal - {channel_name}',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_budget_scenarios(
    scenarios_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Compare plusieurs scénarios de budget.
    
    Parameters
    ----------
    scenarios_df : pd.DataFrame
        DataFrame avec colonnes: budget_total, sales_predicted, roi_overall.
    figsize : tuple, default=(12, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec comparaison des scénarios.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Ventes vs Budget
    ax1.plot(scenarios_df['budget_total'], scenarios_df['sales_predicted'],
            marker='o', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Budget Total (€)', fontsize=11)
    ax1.set_ylabel('Ventes Prédites', fontsize=11)
    ax1.set_title('Ventes en fonction du budget',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. ROI vs Budget
    ax2.plot(scenarios_df['budget_total'], scenarios_df['roi_overall'],
            marker='s', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Budget Total (€)', fontsize=11)
    ax2.set_ylabel('ROI Global', fontsize=11)
    ax2.set_title('ROI en fonction du budget',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax2.legend(fontsize=10)
    
    fig.suptitle('Analyse de scénarios budgétaires',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_incremental_allocation(
    increments_df: pd.DataFrame,
    channel_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualise l'allocation incrémentale du budget.
    
    Parameters
    ----------
    increments_df : pd.DataFrame
        DataFrame retourné par get_optimal_increments().
    channel_names : list of str, optional
        Noms des canaux.
    figsize : tuple, default=(12, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure montrant l'allocation progressive.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. Distribution des incréments
    channel_counts = increments_df['channel'].value_counts().sort_index()
    
    if channel_names is None:
        labels = [f'Canal {i}' for i in channel_counts.index]
    else:
        labels = [channel_names[i] for i in channel_counts.index]
    
    ax1.bar(labels, channel_counts.values, color=sns.color_palette('husl', len(labels)))
    ax1.set_xlabel('Canal', fontsize=11)
    ax1.set_ylabel('Nombre d\'incréments', fontsize=11)
    ax1.set_title('Répartition des incréments budgétaires',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. ROI marginal décroissant
    ax2.plot(increments_df['step'], increments_df['marginal_roi'],
            marker='o', linewidth=2, markersize=6, color='steelblue')
    ax2.set_xlabel('Étape', fontsize=11)
    ax2.set_ylabel('ROI Marginal', fontsize=11)
    ax2.set_title('Évolution du ROI marginal',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Allocation incrémentale du budget additionnel',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_channel_roi_comparison(
    channel_names: List[str],
    roi_values: np.ndarray,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Compare le ROI entre canaux.
    
    Parameters
    ----------
    channel_names : list of str
        Noms des canaux.
    roi_values : np.ndarray
        ROI par canal.
    figsize : tuple, default=(10, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec comparaison des ROI.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Trier par ROI décroissant
    sorted_indices = np.argsort(roi_values)[::-1]
    sorted_names = [channel_names[i] for i in sorted_indices]
    sorted_rois = roi_values[sorted_indices]
    
    # Couleurs (vert si ROI > 0, rouge sinon)
    colors = ['green' if r > 0 else 'red' for r in sorted_rois]
    
    bars = ax.barh(sorted_names, sorted_rois, color=colors, alpha=0.7)
    
    # Ajouter les valeurs
    for i, (bar, roi) in enumerate(zip(bars, sorted_rois)):
        width = bar.get_width()
        ax.text(width, i, f'{roi:.2f}',
               ha='left' if roi > 0 else 'right',
               va='center', fontsize=10, fontweight='bold')
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('ROI', fontsize=12)
    ax.set_title('ROI par canal marketing',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def create_optimization_dashboard(
    optimization_result: Dict,
    channel_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Crée un dashboard complet d'optimisation.
    
    Parameters
    ----------
    optimization_result : dict
        Résultat de optimize_budget_allocation().
    channel_names : list of str, optional
        Noms des canaux.
    save_path : str, optional
        Chemin pour sauvegarder les figures.
    
    Returns
    -------
    dict
        Dictionnaire {nom: figure} de tous les graphiques.
    """
    figures = {}
    
    # 1. Comparaison actuel vs optimal
    if optimization_result['current_spend'] is not None:
        figures['comparison'] = plot_budget_comparison(
            optimization_result['current_spend'],
            optimization_result['optimal_spend'],
            channel_names
        )
    
    # Sauvegarder si demandé
    if save_path:
        for name, fig in figures.items():
            fig.savefig(f'{save_path}/optimization_{name}.png',
                       dpi=300, bbox_inches='tight')
    
    return figures
