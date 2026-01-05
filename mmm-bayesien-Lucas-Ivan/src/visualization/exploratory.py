"""
Exploratory Data Analysis - Visualizations

Fonctions de visualisation pour l'analyse exploratoire des données MMM.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict


def plot_time_series(
    df: pd.DataFrame,
    columns: List[str],
    date_column: str = 'date',
    figsize: Tuple[int, int] = (14, 8),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Visualise les séries temporelles.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset avec séries temporelles.
    columns : list of str
        Colonnes à visualiser.
    date_column : str, default='date'
        Colonne de dates.
    figsize : tuple, default=(14, 8)
        Taille de la figure.
    title : str, optional
        Titre du graphique.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure matplotlib.
    """
    fig, axes = plt.subplots(len(columns), 1, figsize=figsize, sharex=True)
    
    if len(columns) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, columns):
        ax.plot(df[date_column], df[col], linewidth=1.5)
        ax.set_ylabel(col, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('')
    
    axes[-1].set_xlabel('Date', fontsize=11)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    return fig


def plot_sales_vs_media(
    df: pd.DataFrame,
    target_column: str = 'sales',
    media_columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualise la relation entre ventes et dépenses media.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset MMM.
    target_column : str, default='sales'
        Colonne des ventes.
    media_columns : list of str, optional
        Colonnes media. Si None, détection automatique.
    figsize : tuple, default=(14, 10)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure avec scatter plots.
    """
    if media_columns is None:
        media_keywords = ['spend', 'cost', 'impression']
        media_columns = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in media_keywords)
        ]
    
    n_media = len(media_columns)
    n_cols = min(3, n_media)
    n_rows = (n_media + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_media > 1 else [axes]
    
    for idx, col in enumerate(media_columns):
        ax = axes[idx]
        ax.scatter(df[col], df[target_column], alpha=0.5, s=30)
        
        # Régression linéaire simple pour la tendance
        z = np.polyfit(df[col], df[target_column], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)
        
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel(target_column if idx % n_cols == 0 else '', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Correlation
        corr = df[[col, target_column]].corr().iloc[0, 1]
        ax.set_title(f'Corr: {corr:.3f}', fontsize=10)
    
    # Cacher les axes inutilisés
    for idx in range(n_media, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{target_column} vs Media Spend', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'coolwarm'
) -> plt.Figure:
    """
    Visualise la matrice de corrélation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    columns : list of str, optional
        Colonnes à inclure. Si None, toutes les colonnes numériques.
    figsize : tuple, default=(10, 8)
        Taille de la figure.
    cmap : str, default='coolwarm'
        Colormap.
    
    Returns
    -------
    matplotlib.figure.Figure
        Heatmap de corrélation.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Matrice de corrélation', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_distributions(
    df: pd.DataFrame,
    columns: List[str],
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualise les distributions des variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    columns : list of str
        Colonnes à visualiser.
    figsize : tuple, default=(14, 10)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Histogrammes et boxplots.
    """
    n_cols = len(columns)
    n_plot_cols = min(3, n_cols)
    n_rows = (n_cols + n_plot_cols - 1) // n_plot_cols
    
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=figsize)
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Histogramme avec KDE
        ax.hist(df[col], bins=30, alpha=0.6, edgecolor='black', density=True)
        
        # KDE
        df[col].plot.kde(ax=ax, linewidth=2, color='red')
        
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Densité', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Stats dans le titre
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.set_title(f'μ={mean_val:.1f}, σ={std_val:.1f}', fontsize=10)
    
    # Cacher les axes inutilisés
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Distributions des variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_seasonality_decomposition(
    df: pd.DataFrame,
    column: str,
    date_column: str = 'date',
    period: int = 52,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Décompose et visualise tendance et saisonnalité.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset avec série temporelle.
    column : str
        Colonne à décomposer.
    date_column : str, default='date'
        Colonne de dates.
    period : int, default=52
        Période pour la décomposition (52 semaines par défaut).
    figsize : tuple, default=(14, 10)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Décomposition en 4 graphiques.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Préparer la série avec index temporel
    ts = df.set_index(date_column)[column]
    
    # Décomposition
    decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')
    
    # Visualisation
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    # Série originale
    axes[0].plot(ts.index, ts.values, linewidth=1.5)
    axes[0].set_ylabel('Original', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Tendance
    axes[1].plot(ts.index, decomposition.trend, linewidth=1.5, color='orange')
    axes[1].set_ylabel('Tendance', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Saisonnalité
    axes[2].plot(ts.index, decomposition.seasonal, linewidth=1.5, color='green')
    axes[2].set_ylabel('Saisonnalité', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    # Résidus
    axes[3].plot(ts.index, decomposition.resid, linewidth=1, color='red', alpha=0.7)
    axes[3].set_ylabel('Résidus', fontsize=11)
    axes[3].set_xlabel('Date', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    
    fig.suptitle(f'Décomposition : {column}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_spending_patterns(
    df: pd.DataFrame,
    media_columns: List[str],
    date_column: str = 'date',
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Visualise les patterns de dépenses media (stacked area).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset MMM.
    media_columns : list of str
        Colonnes de dépenses media.
    date_column : str, default='date'
        Colonne de dates.
    figsize : tuple, default=(14, 6)
        Taille de la figure.
    
    Returns
    -------
    matplotlib.figure.Figure
        Stacked area chart.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Stacked area
    ax.stackplot(
        df[date_column],
        *[df[col] for col in media_columns],
        labels=media_columns,
        alpha=0.7
    )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Dépenses media', fontsize=12)
    ax.set_title('Évolution des dépenses media par canal', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_eda_report(
    df: pd.DataFrame,
    target_column: str = 'sales',
    media_columns: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Génère un rapport EDA complet avec tous les graphiques.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset MMM.
    target_column : str, default='sales'
        Colonne cible.
    media_columns : list of str, optional
        Colonnes media.
    save_path : str, optional
        Chemin pour sauvegarder les figures.
    
    Returns
    -------
    dict
        Dictionnaire {nom: figure} de tous les graphiques.
    """
    if media_columns is None:
        media_keywords = ['spend', 'cost', 'impression']
        media_columns = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in media_keywords)
        ]
    
    figures = {}
    
    # 1. Séries temporelles
    all_cols = [target_column] + media_columns
    figures['time_series'] = plot_time_series(df, all_cols, title='Évolution temporelle')
    
    # 2. Sales vs Media
    figures['sales_vs_media'] = plot_sales_vs_media(df, target_column, media_columns)
    
    # 3. Corrélations
    figures['correlation'] = plot_correlation_matrix(df, all_cols)
    
    # 4. Distributions
    figures['distributions'] = plot_distributions(df, all_cols)
    
    # 5. Décomposition saisonnalité (ventes)
    if 'date' in df.columns and len(df) >= 52:
        figures['seasonality'] = plot_seasonality_decomposition(df, target_column)
    
    # 6. Spending patterns
    if len(media_columns) > 0:
        figures['spending_patterns'] = plot_spending_patterns(df, media_columns)
    
    # Sauvegarder si demandé
    if save_path:
        for name, fig in figures.items():
            fig.savefig(f'{save_path}/{name}.png', dpi=300, bbox_inches='tight')
    
    return figures
