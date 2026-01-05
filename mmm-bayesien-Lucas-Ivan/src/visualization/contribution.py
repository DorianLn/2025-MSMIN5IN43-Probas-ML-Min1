"""
Channel Contribution Analysis

Analyse et visualisation des contributions de chaque canal aux ventes.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from typing import Optional, List, Tuple, Dict


def calculate_channel_contributions(
    trace: az.InferenceData,
    X_media: np.ndarray,
    channel_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calcule les contributions de chaque canal aux ventes.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC avec résultats.
    X_media : np.ndarray
        Dépenses media (après transformations).
    channel_names : list of str, optional
        Noms des canaux.
    
    Returns
    -------
    pd.DataFrame
        Contributions par canal.
    """
    n_media = X_media.shape[1]
    
    if channel_names is None:
        channel_names = [f'Canal {i+1}' for i in range(n_media)]
    
    # Extraire les coefficients
    beta_media = trace.posterior['beta_media'].mean(dim=['chain', 'draw']).values
    
    # Calculer les contributions
    contributions = []
    for i in range(n_media):
        contrib = X_media[:, i] * beta_media[i]
        contributions.append({
            'channel': channel_names[i],
            'total_contribution': contrib.sum(),
            'mean_contribution': contrib.mean(),
            'coefficient': beta_media[i]
        })
    
    df = pd.DataFrame(contributions)
    df['pct_total'] = (df['total_contribution'] / df['total_contribution'].sum()) * 100
    
    return df.sort_values('total_contribution', ascending=False)


def plot_contribution_bars(
    contributions: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Barplot des contributions par canal.
    
    Parameters
    ----------
    contributions : pd.DataFrame
        Contributions calculées.
    figsize : tuple, default=(10, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec barplot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(
        contributions['channel'],
        contributions['pct_total'],
        color=sns.color_palette('husl', len(contributions))
    )
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xlabel('Canal', fontsize=12)
    ax.set_ylabel('Contribution aux ventes (%)', fontsize=12)
    ax.set_title('Attribution des ventes par canal marketing',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def plot_contribution_pie(
    contributions: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Pie chart des contributions.
    
    Parameters
    ----------
    contributions : pd.DataFrame
        Contributions calculées.
    figsize : tuple, default=(8, 8)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec pie chart.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette('husl', len(contributions))
    
    wedges, texts, autotexts = ax.pie(
        contributions['pct_total'],
        labels=contributions['channel'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10}
    )
    
    # Mettre en gras les pourcentages
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Répartition des contributions par canal',
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig


def plot_contribution_waterfall(
    contributions: pd.DataFrame,
    baseline: float,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Waterfall chart montrant l'accumulation des contributions.
    
    Parameters
    ----------
    contributions : pd.DataFrame
        Contributions calculées.
    baseline : float
        Baseline (intercept du modèle).
    figsize : tuple, default=(12, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec waterfall chart.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Préparer les données
    categories = ['Baseline'] + list(contributions['channel']) + ['Total']
    values = [baseline] + list(contributions['total_contribution'])
    
    # Calculer le total
    total = baseline + contributions['total_contribution'].sum()
    values.append(total)
    
    # Calculer les positions
    cumulative = [0]
    for val in values[:-1]:
        cumulative.append(cumulative[-1] + val)
    
    # Dessiner les barres
    colors = ['lightblue'] + ['steelblue'] * len(contributions) + ['darkgreen']
    
    for i, (cat, val, cum) in enumerate(zip(categories, values, cumulative)):
        if i == len(categories) - 1:  # Total
            ax.bar(i, val, color=colors[i], edgecolor='black', linewidth=1.5)
            ax.text(i, val / 2, f'{val:.0f}', ha='center', va='center',
                   fontweight='bold', color='white', fontsize=11)
        else:
            ax.bar(i, val, bottom=cum, color=colors[i], 
                  edgecolor='black', linewidth=1)
            ax.text(i, cum + val / 2, f'+{val:.0f}', ha='center', va='center',
                   fontweight='bold', fontsize=10)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Contribution aux ventes', fontsize=12)
    ax.set_title('Waterfall - Décomposition des ventes par canal',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_contribution_time_series(
    trace: az.InferenceData,
    X_media: np.ndarray,
    dates: pd.DatetimeIndex,
    channel_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Évolution temporelle des contributions.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    X_media : np.ndarray
        Dépenses media transformées.
    dates : pd.DatetimeIndex
        Dates des périodes.
    channel_names : list of str, optional
        Noms des canaux.
    figsize : tuple, default=(14, 8)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec séries temporelles empilées.
    """
    n_media = X_media.shape[1]
    
    if channel_names is None:
        channel_names = [f'Canal {i+1}' for i in range(n_media)]
    
    # Extraire coefficients
    beta_media = trace.posterior['beta_media'].mean(dim=['chain', 'draw']).values
    
    # Calculer contributions temporelles
    contributions_ts = np.zeros((len(dates), n_media))
    for i in range(n_media):
        contributions_ts[:, i] = X_media[:, i] * beta_media[i]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.stackplot(
        dates,
        *[contributions_ts[:, i] for i in range(n_media)],
        labels=channel_names,
        alpha=0.8
    )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Contribution aux ventes', fontsize=12)
    ax.set_title('Évolution temporelle des contributions par canal',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_response_curves(
    spend_range: np.ndarray,
    alpha: Optional[float] = None,
    k: Optional[float] = None,
    s: Optional[float] = None,
    channel_name: str = 'Canal',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Courbes de réponse (saturation) pour un canal.
    
    Parameters
    ----------
    spend_range : np.ndarray
        Plage de dépenses à visualiser.
    alpha : float, optional
        Paramètre adstock.
    k : float, optional
        Half-saturation point.
    s : float, optional
        Slope de saturation.
    channel_name : str, default='Canal'
        Nom du canal.
    figsize : tuple, default=(10, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec courbe de réponse.
    """
    from models.transformations import geometric_adstock, hill_saturation
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Dépenses brutes
    axes[0].plot(spend_range, spend_range, linewidth=2)
    axes[0].set_xlabel('Dépenses (€)', fontsize=10)
    axes[0].set_ylabel('Impact brut', fontsize=10)
    axes[0].set_title('Sans transformation', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Avec adstock
    if alpha is not None:
        spend_adstocked = geometric_adstock(
            spend_range.reshape(-1, 1),
            alpha=np.array([alpha]),
            l_max=8
        ).flatten()
        axes[1].plot(spend_range, spend_adstocked, linewidth=2, color='orange')
        axes[1].set_xlabel('Dépenses (€)', fontsize=10)
        axes[1].set_ylabel('Impact avec adstock', fontsize=10)
        axes[1].set_title(f'Adstock (α={alpha:.2f})', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    # 3. Avec saturation
    if k is not None and s is not None:
        spend_saturated = hill_saturation(
            spend_range.reshape(-1, 1),
            half_saturation=np.array([k]),
            slope=np.array([s])
        ).flatten()
        axes[2].plot(spend_range, spend_saturated, linewidth=2, color='green')
        axes[2].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% saturation')
        axes[2].axvline(k, color='red', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Dépenses (€)', fontsize=10)
        axes[2].set_ylabel('Impact saturé [0,1]', fontsize=10)
        axes[2].set_title(f'Saturation (k={k:.0f}, s={s:.1f})', 
                         fontsize=11, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f'Courbes de réponse - {channel_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_contribution_report(
    trace: az.InferenceData,
    X_media: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    channel_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Génère un rapport complet d'attribution.
    
    Parameters
    ----------
    trace : az.InferenceData
        Trace MCMC.
    X_media : np.ndarray
        Dépenses media transformées.
    dates : pd.DatetimeIndex, optional
        Dates des périodes.
    channel_names : list of str, optional
        Noms des canaux.
    save_path : str, optional
        Chemin pour sauvegarder les figures.
    
    Returns
    -------
    dict
        Dictionnaire {nom: figure} de tous les graphiques.
    """
    # Calculer contributions
    contributions = calculate_channel_contributions(trace, X_media, channel_names)
    
    # Générer les figures
    figures = {}
    
    # 1. Barplot
    figures['bars'] = plot_contribution_bars(contributions)
    
    # 2. Pie chart
    figures['pie'] = plot_contribution_pie(contributions)
    
    # 3. Time series (si dates disponibles)
    if dates is not None:
        figures['timeseries'] = plot_contribution_time_series(
            trace, X_media, dates, channel_names
        )
    
    # Sauvegarder si demandé
    if save_path:
        for name, fig in figures.items():
            fig.savefig(f'{save_path}/contribution_{name}.png',
                       dpi=300, bbox_inches='tight')
    
    return figures
