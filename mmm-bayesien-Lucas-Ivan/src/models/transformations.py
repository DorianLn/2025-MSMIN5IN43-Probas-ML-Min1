"""
Marketing Mix Modeling - Transformations

Ce module implémente les transformations clés pour le MMM bayésien :
1. Adstock géométrique : modélise la persistance temporelle de l'effet publicitaire
2. Saturation de Hill : modélise les rendements décroissants des investissements

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
École : EPF Engineering School
"""

import numpy as np
from typing import Union, Optional
import warnings


def geometric_adstock(
    x: np.ndarray,
    alpha: Union[float, np.ndarray],
    l_max: int = 8,
    normalize: bool = True,
    axis: int = 0
) -> np.ndarray:
    """
    Applique une transformation d'adstock géométrique (Koyck transformation).
    
    L'adstock modélise l'effet de persistance de la publicité : l'impact d'une 
    dépense publicitaire ne se limite pas à la période où elle est effectuée, 
    mais se prolonge sur plusieurs périodes suivantes avec un effet décroissant.
    
    Formule mathématique :
    $$
    y_t = x_t + \\alpha \\cdot x_{t-1} + \\alpha^2 \\cdot x_{t-2} + ... + \\alpha^{l_{max}} \\cdot x_{t-l_{max}}
    $$
    
    Où :
    - x_t : dépense publicitaire à la période t
    - α (alpha) : taux de rétention (0 < α < 1)
    - l_max : longueur maximale de l'effet de persistance
    
    Interprétation :
    - α = 0 : effet immédiat uniquement (pas de persistance)
    - α = 0.5 : 50% de l'effet persiste à la période suivante
    - α = 0.9 : forte persistance (ex: campagnes de branding)
    
    Parameters
    ----------
    x : np.ndarray
        Séries temporelles des dépenses publicitaires.
        Shape: (n_time_periods,) ou (n_time_periods, n_channels)
    alpha : float or np.ndarray
        Taux de rétention de l'adstock. Doit être dans [0, 1).
        Si array, shape doit correspondre au nombre de canaux.
    l_max : int, default=8
        Nombre maximum de périodes de persistance (lag maximum).
        Typiquement 4-12 semaines pour des données hebdomadaires.
    normalize : bool, default=True
        Si True, normalise les poids pour que leur somme = 1.
        Permet de conserver l'échelle des dépenses d'origine.
    axis : int, default=0
        Axe temporel le long duquel appliquer l'adstock.
    
    Returns
    -------
    np.ndarray
        Séries temporelles transformées avec effet d'adstock.
        Même shape que l'input x.
    
    Examples
    --------
    >>> # Exemple 1 : Canal unique avec faible persistance
    >>> spend = np.array([100, 0, 0, 0, 0])
    >>> adstocked = geometric_adstock(spend, alpha=0.3, l_max=4)
    >>> print(adstocked)  # [100, 30, 9, 2.7, 0.81]
    
    >>> # Exemple 2 : Deux canaux avec persistances différentes
    >>> spend = np.array([[100, 50], [0, 0], [0, 0]])
    >>> alpha = np.array([0.5, 0.8])  # TV plus persistant que digital
    >>> adstocked = geometric_adstock(spend, alpha=alpha, l_max=2)
    
    References
    ----------
    - Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017).
      Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.
    - https://www.pymc.io/projects/marketing/en/stable/
    
    Raises
    ------
    ValueError
        Si alpha n'est pas dans l'intervalle [0, 1) ou si les dimensions ne correspondent pas.
    """
    # Validation des inputs
    x = np.asarray(x)
    alpha = np.asarray(alpha)
    
    # Vérifier que alpha est dans [0, 1)
    if np.any(alpha < 0) or np.any(alpha >= 1):
        raise ValueError(
            f"alpha doit être dans [0, 1). Reçu: {alpha}"
        )
    
    if l_max < 1:
        raise ValueError(f"l_max doit être >= 1. Reçu: {l_max}")
    
    # Gérer le cas 1D vs 2D
    is_1d = x.ndim == 1
    if is_1d:
        x = x.reshape(-1, 1)
        alpha = np.atleast_1d(alpha)
    
    n_time, n_channels = x.shape
    
    # Vérifier la compatibilité des dimensions
    if alpha.size not in [1, n_channels]:
        raise ValueError(
            f"alpha doit avoir taille 1 ou {n_channels}. Reçu: {alpha.size}"
        )
    
    # Broadcast alpha si nécessaire
    if alpha.size == 1:
        alpha = np.full(n_channels, alpha.item())
    
    # Construire les poids de convolution pour chaque canal
    # weights[lag, channel] = alpha[channel]^lag
    lags = np.arange(l_max + 1)  # [0, 1, 2, ..., l_max]
    weights = alpha[np.newaxis, :] ** lags[:, np.newaxis]  # shape: (l_max+1, n_channels)
    
    # Normalisation optionnelle
    if normalize:
        # Somme géométrique : sum(alpha^i for i=0 to inf) = 1/(1-alpha)
        # Pour l_max fini : approximation par la somme des premiers termes
        weight_sum = weights.sum(axis=0, keepdims=True)
        weights = weights / weight_sum
    
    # Appliquer la convolution pour chaque canal
    # Implémentation manuelle pour éviter les problèmes de taille avec np.convolve
    result = np.zeros_like(x, dtype=float)
    
    for channel in range(n_channels):
        for t in range(n_time):
            # Pour chaque période t, sommer les contributions des périodes passées
            # y_t = sum_{l=0}^{min(t, l_max)} w_l * x_{t-l}
            for lag in range(min(t + 1, l_max + 1)):
                result[t, channel] += weights[lag, channel] * x[t - lag, channel]
    
    # Retourner au format 1D si input était 1D
    if is_1d:
        result = result[:, 0]  # Garde la dimension temporelle
    
    return result


def hill_saturation(
    x: np.ndarray,
    half_saturation: Union[float, np.ndarray],
    slope: Union[float, np.ndarray] = 1.0
) -> np.ndarray:
    """
    Applique une transformation de saturation de Hill (courbe sigmoïde).
    
    La saturation modélise le principe des rendements décroissants : augmenter
    les dépenses publicitaires finit par avoir un impact de moins en moins important
    sur les ventes (loi des rendements marginaux décroissants).
    
    Formule mathématique (Hill equation) :
    $$
    y = \\frac{x^s}{k^s + x^s}
    $$
    
    Où :
    - x : dépense publicitaire (après adstock)
    - k (half_saturation) : point de demi-saturation (x où y = 0.5)
    - s (slope) : pente de la courbe (contrôle la "douceur" de la transition)
    
    Interprétation :
    - k petit : saturation rapide (ex: marché de niche)
    - k grand : saturation lente (ex: marché de masse)
    - s < 1 : courbe très douce, saturation progressive
    - s = 1 : courbe standard (Michaelis-Menten)
    - s > 1 : courbe abrupte, effet de seuil
    
    La sortie est normalisée dans [0, 1], représentant la "saturation de l'effet".
    
    Parameters
    ----------
    x : np.ndarray
        Dépenses publicitaires (généralement après transformation adstock).
        Shape: (n_time_periods,) ou (n_time_periods, n_channels)
    half_saturation : float or np.ndarray
        Point de demi-saturation (K dans la formule de Hill).
        Doit être > 0. Si array, un paramètre par canal.
    slope : float or np.ndarray, default=1.0
        Paramètre de forme (S dans la formule de Hill).
        Doit être > 0. Si array, un paramètre par canal.
    
    Returns
    -------
    np.ndarray
        Valeurs saturées dans [0, 1]. Même shape que x.
    
    Examples
    --------
    >>> # Exemple 1 : Saturation standard (slope=1)
    >>> spend = np.array([0, 50, 100, 200, 500])
    >>> saturated = hill_saturation(spend, half_saturation=100, slope=1)
    >>> # À spend=100 → output≈0.5 (demi-saturation)
    
    >>> # Exemple 2 : Comparaison de différentes pentes
    >>> spend = np.linspace(0, 500, 100)
    >>> sat_douce = hill_saturation(spend, 100, slope=0.5)  # saturation douce
    >>> sat_abrupte = hill_saturation(spend, 100, slope=2.0)  # saturation rapide
    
    >>> # Exemple 3 : Plusieurs canaux avec paramètres différents
    >>> spend = np.array([[100, 200], [50, 100]])
    >>> k = np.array([80, 150])  # TV sature plus vite que digital
    >>> saturated = hill_saturation(spend, half_saturation=k)
    
    References
    ----------
    - Hill, A. V. (1910). The possible effects of the aggregation of the molecules
      of haemoglobin on its dissociation curves.
    - Chan, D., & Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling.
    
    Raises
    ------
    ValueError
        Si half_saturation ou slope ne sont pas strictement positifs.
    
    Notes
    -----
    La fonction de Hill est également utilisée en biologie (courbe dose-réponse)
    et en économie (fonction de production avec rendements décroissants).
    """
    # Validation des inputs
    x = np.asarray(x, dtype=float)
    half_saturation = np.asarray(half_saturation, dtype=float)
    slope = np.asarray(slope, dtype=float)
    
    # Vérifications de validité
    if np.any(half_saturation <= 0):
        raise ValueError(
            f"half_saturation doit être > 0. Reçu: {half_saturation}"
        )
    
    if np.any(slope <= 0):
        raise ValueError(
            f"slope doit être > 0. Reçu: {slope}"
        )
    
    # Vérifier que x ne contient pas de valeurs négatives
    if np.any(x < 0):
        warnings.warn(
            "Les valeurs négatives dans x seront traitées comme 0. "
            "Les dépenses publicitaires doivent être non-négatives.",
            UserWarning
        )
        x = np.maximum(x, 0)
    
    # Gérer le broadcasting pour multi-canaux
    # Formule de Hill : x^s / (k^s + x^s)
    
    # Éviter les overflow avec grandes valeurs
    # On utilise une forme numériquement stable
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        # x^s
        x_powered = np.power(x, slope)
        # k^s
        k_powered = np.power(half_saturation, slope)
        
        # Calcul de la saturation
        result = x_powered / (k_powered + x_powered)
        
        # Gérer les cas limites
        # Quand x=0, result devrait être 0
        result = np.where(x == 0, 0, result)
        # Quand x >> k, result devrait tendre vers 1
        result = np.where(np.isinf(x_powered), 1, result)
        # Remplacer les NaN par 0 (cas x=0)
        result = np.nan_to_num(result, nan=0.0, posinf=1.0)
    
    return result


def adstock_and_saturation(
    x: np.ndarray,
    alpha: Union[float, np.ndarray],
    half_saturation: Union[float, np.ndarray],
    l_max: int = 8,
    slope: Union[float, np.ndarray] = 1.0,
    normalize_adstock: bool = True
) -> np.ndarray:
    """
    Pipeline complet : applique successivement adstock puis saturation.
    
    C'est la transformation standard en MMM :
    1. Adstock : modélise la persistance temporelle
    2. Saturation : modélise les rendements décroissants
    
    Cette séquence capture deux phénomènes marketing clés :
    - L'effet publicitaire dure dans le temps (adstock)
    - Doubler le budget ne double pas l'impact (saturation)
    
    Parameters
    ----------
    x : np.ndarray
        Dépenses publicitaires brutes.
    alpha : float or np.ndarray
        Paramètre d'adstock (taux de rétention).
    half_saturation : float or np.ndarray
        Paramètre de saturation (point de demi-saturation).
    l_max : int, default=8
        Longueur maximale de l'effet d'adstock.
    slope : float or np.ndarray, default=1.0
        Paramètre de pente de la saturation.
    normalize_adstock : bool, default=True
        Si True, normalise les poids d'adstock.
    
    Returns
    -------
    np.ndarray
        Dépenses transformées (adstockées puis saturées).
    
    Examples
    --------
    >>> spend = np.array([100, 50, 0, 0, 0])
    >>> transformed = adstock_and_saturation(
    ...     spend, 
    ...     alpha=0.5, 
    ...     half_saturation=80
    ... )
    
    Notes
    -----
    L'ordre des transformations est important :
    - Adstock d'abord : capture la dynamique temporelle
    - Saturation ensuite : capture les rendements décroissants du total cumulé
    """
    # Étape 1 : Appliquer l'adstock
    x_adstocked = geometric_adstock(
        x=x,
        alpha=alpha,
        l_max=l_max,
        normalize=normalize_adstock
    )
    
    # Étape 2 : Appliquer la saturation sur les valeurs adstockées
    x_saturated = hill_saturation(
        x=x_adstocked,
        half_saturation=half_saturation,
        slope=slope
    )
    
    return x_saturated


# Fonctions utilitaires pour l'analyse

def get_effective_reach_curve(
    spend_range: np.ndarray,
    half_saturation: float,
    slope: float = 1.0
) -> np.ndarray:
    """
    Génère une courbe de portée effective pour visualiser la saturation.
    
    Utile pour visualiser l'effet de saturation et trouver le point optimal
    de dépenses (compromis coût/efficacité).
    
    Parameters
    ----------
    spend_range : np.ndarray
        Plage de dépenses à évaluer.
    half_saturation : float
        Point de demi-saturation.
    slope : float, default=1.0
        Pente de la courbe.
    
    Returns
    -------
    np.ndarray
        Valeurs de saturation correspondantes.
    """
    return hill_saturation(spend_range, half_saturation, slope)


def get_adstock_decay_weights(
    alpha: float,
    l_max: int,
    normalize: bool = True
) -> np.ndarray:
    """
    Calcule les poids de décroissance de l'adstock.
    
    Utile pour visualiser l'effet de persistance et comprendre
    combien de temps l'effet publicitaire dure.
    
    Parameters
    ----------
    alpha : float
        Taux de rétention de l'adstock.
    l_max : int
        Nombre de périodes.
    normalize : bool, default=True
        Si True, normalise les poids.
    
    Returns
    -------
    np.ndarray
        Poids de décroissance pour chaque lag.
    
    Examples
    --------
    >>> weights = get_adstock_decay_weights(alpha=0.5, l_max=5)
    >>> # Affiche comment l'effet décroît : [1.0, 0.5, 0.25, 0.125, ...]
    """
    lags = np.arange(l_max + 1)
    weights = alpha ** lags
    
    if normalize:
        weights = weights / weights.sum()
    
    return weights
