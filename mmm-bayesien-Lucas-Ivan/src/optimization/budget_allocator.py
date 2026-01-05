"""
Budget Allocation Optimizer

Optimisation de l'allocation budgétaire marketing.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Optional, Dict, List, Tuple, Callable
import warnings


def calculate_marginal_roi(
    spend: np.ndarray,
    coefficients: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    k: Optional[np.ndarray] = None,
    s: Optional[np.ndarray] = None,
    delta: float = 1.0
) -> np.ndarray:
    """
    Calcule le ROI marginal pour chaque canal.
    
    ROI marginal = (Sales(spend + Δ) - Sales(spend)) / Δ
    
    Parameters
    ----------
    spend : np.ndarray
        Dépenses actuelles par canal.
    coefficients : np.ndarray
        Coefficients beta du modèle.
    alpha : np.ndarray, optional
        Paramètres adstock.
    k : np.ndarray, optional
        Paramètres half-saturation.
    s : np.ndarray, optional
        Paramètres slope.
    delta : float, default=1.0
        Incrément pour calculer la dérivée.
    
    Returns
    -------
    np.ndarray
        ROI marginal par canal.
    """
    from models.transformations import geometric_adstock, hill_saturation
    
    n_channels = len(spend)
    marginal_rois = np.zeros(n_channels)
    
    for i in range(n_channels):
        # Dépenses actuelles
        spend_current = spend.copy()
        
        # Dépenses avec incrément
        spend_increased = spend.copy()
        spend_increased[i] += delta
        
        # Appliquer transformations
        def transform(x):
            x_transformed = x.copy()
            
            if alpha is not None:
                x_transformed = geometric_adstock(
                    x_transformed.reshape(-1, 1),
                    alpha=alpha[i:i+1],
                    l_max=8
                ).flatten()
            
            if k is not None and s is not None:
                x_transformed = hill_saturation(
                    x_transformed.reshape(-1, 1),
                    half_saturation=k[i:i+1],
                    slope=s[i:i+1]
                ).flatten()
            
            return x_transformed
        
        # Calculer les ventes
        sales_current = transform(spend_current[i:i+1]) * coefficients[i]
        sales_increased = transform(spend_increased[i:i+1]) * coefficients[i]
        
        # ROI marginal
        marginal_rois[i] = (sales_increased - sales_current) / delta
    
    return marginal_rois


def optimize_budget_allocation(
    total_budget: float,
    coefficients: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    k: Optional[np.ndarray] = None,
    s: Optional[np.ndarray] = None,
    min_spend: Optional[np.ndarray] = None,
    max_spend: Optional[np.ndarray] = None,
    current_spend: Optional[np.ndarray] = None
) -> Dict:
    """
    Trouve l'allocation budgétaire optimale.
    
    Parameters
    ----------
    total_budget : float
        Budget total à allouer.
    coefficients : np.ndarray
        Coefficients beta du modèle.
    alpha : np.ndarray, optional
        Paramètres adstock.
    k : np.ndarray, optional
        Paramètres half-saturation.
    s : np.ndarray, optional
        Paramètres slope.
    min_spend : np.ndarray, optional
        Dépenses minimum par canal.
    max_spend : np.ndarray, optional
        Dépenses maximum par canal.
    current_spend : np.ndarray, optional
        Dépenses actuelles (pour comparaison).
    
    Returns
    -------
    dict
        Résultats de l'optimisation avec allocation optimale.
    """
    from models.transformations import geometric_adstock, hill_saturation
    
    n_channels = len(coefficients)
    
    # Bornes par défaut
    if min_spend is None:
        min_spend = np.zeros(n_channels)
    if max_spend is None:
        max_spend = np.full(n_channels, total_budget)
    
    # Fonction objectif (à maximiser → minimiser le négatif)
    def objective(spend):
        # spend est un vecteur (n_channels,), on le reshape en (1, n_channels)
        # pour correspondre à la convention: (n_samples, n_channels)
        spend = spend.reshape(1, -1)
        
        # Appliquer transformations
        spend_transformed = spend.copy()
        
        if alpha is not None:
            spend_transformed = geometric_adstock(
                spend_transformed,
                alpha=alpha,
                l_max=8
            )
        
        if k is not None and s is not None:
            spend_transformed = hill_saturation(
                spend_transformed,
                half_saturation=k,
                slope=s
            )
        
        # Calculer les ventes
        sales = np.sum(spend_transformed.flatten() * coefficients)
        
        return -sales  # Négatif pour minimiser
    
    # Contraintes
    # 1. Budget total
    constraints = [
        LinearConstraint(np.ones(n_channels), total_budget, total_budget)
    ]
    
    # 2. Bornes par canal
    bounds = Bounds(min_spend, max_spend)
    
    # Point de départ (répartition égale ou current_spend)
    if current_spend is not None:
        x0 = current_spend
    else:
        x0 = np.full(n_channels, total_budget / n_channels)
    
    # Optimisation
    result = minimize(
        objective,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        warnings.warn(f"Optimisation non convergée : {result.message}")
    
    # Calculer les ventes optimales
    optimal_spend = result.x
    optimal_sales = -result.fun
    
    # Ventes actuelles (si comparaison possible)
    current_sales = None
    if current_spend is not None:
        current_sales = -objective(current_spend)
    
    # ROI marginal à l'optimum
    marginal_roi = calculate_marginal_roi(
        optimal_spend, coefficients, alpha, k, s
    )
    
    return {
        'optimal_spend': optimal_spend,
        'optimal_sales': optimal_sales,
        'current_spend': current_spend,
        'current_sales': current_sales,
        'improvement': (optimal_sales - current_sales) / current_sales * 100 if current_sales else None,
        'marginal_roi': marginal_roi,
        'success': result.success,
        'message': result.message
    }


def what_if_scenario(
    scenario_budget: float,
    coefficients: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    k: Optional[np.ndarray] = None,
    s: Optional[np.ndarray] = None,
    min_spend: Optional[np.ndarray] = None,
    max_spend: Optional[np.ndarray] = None
) -> Dict:
    """
    Simule un scénario avec un budget différent.
    
    Parameters
    ----------
    scenario_budget : float
        Budget hypothétique.
    coefficients : np.ndarray
        Coefficients du modèle.
    alpha, k, s : np.ndarray, optional
        Paramètres de transformation.
    min_spend, max_spend : np.ndarray, optional
        Contraintes par canal.
    
    Returns
    -------
    dict
        Résultats du scénario.
    """
    return optimize_budget_allocation(
        total_budget=scenario_budget,
        coefficients=coefficients,
        alpha=alpha,
        k=k,
        s=s,
        min_spend=min_spend,
        max_spend=max_spend
    )


def compare_scenarios(
    budgets: List[float],
    coefficients: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    k: Optional[np.ndarray] = None,
    s: Optional[np.ndarray] = None,
    channel_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare plusieurs scénarios de budget.
    
    Parameters
    ----------
    budgets : list of float
        Liste de budgets à comparer.
    coefficients : np.ndarray
        Coefficients du modèle.
    alpha, k, s : np.ndarray, optional
        Paramètres de transformation.
    channel_names : list of str, optional
        Noms des canaux.
    
    Returns
    -------
    pd.DataFrame
        Comparaison des scénarios.
    """
    results = []
    
    for budget in budgets:
        result = what_if_scenario(
            scenario_budget=budget,
            coefficients=coefficients,
            alpha=alpha,
            k=k,
            s=s
        )
        
        results.append({
            'budget_total': budget,
            'sales_predicted': result['optimal_sales'],
            'roi_overall': result['optimal_sales'] / budget if budget > 0 else 0
        })
    
    return pd.DataFrame(results)


def calculate_channel_roi(
    spend: np.ndarray,
    sales_contribution: np.ndarray
) -> np.ndarray:
    """
    Calcule le ROI par canal.
    
    ROI = (Contribution / Dépense) - 1
    
    Parameters
    ----------
    spend : np.ndarray
        Dépenses par canal.
    sales_contribution : np.ndarray
        Contribution aux ventes par canal.
    
    Returns
    -------
    np.ndarray
        ROI par canal.
    """
    roi = np.zeros_like(spend)
    
    for i in range(len(spend)):
        if spend[i] > 0:
            roi[i] = (sales_contribution[i] / spend[i]) - 1
        else:
            roi[i] = 0
    
    return roi


def get_optimal_increments(
    current_spend: np.ndarray,
    total_budget_increase: float,
    coefficients: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    k: Optional[np.ndarray] = None,
    s: Optional[np.ndarray] = None,
    n_steps: int = 10
) -> pd.DataFrame:
    """
    Calcule les incréments optimaux par canal pour un budget additionnel.
    
    Parameters
    ----------
    current_spend : np.ndarray
        Dépenses actuelles.
    total_budget_increase : float
        Budget additionnel à allouer.
    coefficients : np.ndarray
        Coefficients du modèle.
    alpha, k, s : np.ndarray, optional
        Paramètres de transformation.
    n_steps : int, default=10
        Nombre d'étapes d'allocation.
    
    Returns
    -------
    pd.DataFrame
        Historique des incréments optimaux.
    """
    n_channels = len(current_spend)
    spend = current_spend.copy()
    budget_remaining = total_budget_increase
    
    increments = []
    step_size = total_budget_increase / n_steps
    
    for step in range(n_steps):
        # Calculer ROI marginal pour chaque canal
        marginal_roi = calculate_marginal_roi(
            spend, coefficients, alpha, k, s, delta=step_size
        )
        
        # Allouer au canal avec le meilleur ROI marginal
        best_channel = np.argmax(marginal_roi)
        spend[best_channel] += step_size
        
        increments.append({
            'step': step + 1,
            'channel': best_channel,
            'increment': step_size,
            'total_spend': spend.copy(),
            'marginal_roi': marginal_roi[best_channel]
        })
    
    return pd.DataFrame(increments)