"""
Base Marketing Mix Model (MMM) - Bayesian Approach

Modèle MMM bayésien avec PyMC.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import pymc as pm
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import arviz as az

from models.priors import get_default_priors, create_prior, validate_priors
from models.transformations import geometric_adstock, hill_saturation


class BayesianMMM:
    """
    Modèle de Marketing Mix Modeling bayésien.
    
    Ce modèle estime l'impact des dépenses media sur les ventes en utilisant
    l'inférence bayésienne avec PyMC.
    
    Version simple : régression linéaire bayésienne
    Version avancée : avec adstock et saturation
    
    Attributes
    ----------
    model : pm.Model
        Modèle PyMC.
    trace : az.InferenceData
        Résultats de l'inférence MCMC.
    """
    
    def __init__(
        self,
        prior_config: Optional[Dict] = None,
        use_adstock: bool = False,
        use_saturation: bool = False
    ):
        """
        Initialise le modèle MMM.
        
        Parameters
        ----------
        prior_config : dict, optional
            Configuration des priors. Si None, utilise les defaults.
        use_adstock : bool, default=False
            Activer la transformation adstock.
        use_saturation : bool, default=False
            Activer la transformation saturation.
        """
        self.prior_config = prior_config
        self.use_adstock = use_adstock
        self.use_saturation = use_saturation
        self.model = None
        self.trace = None
        self.data = {}
        self.metadata = {}
    
    def build_model(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        X_control: Optional[np.ndarray] = None,
        adstock_params: Optional[Dict] = None,
        saturation_params: Optional[Dict] = None
    ) -> pm.Model:
        """
        Construit le modèle MMM bayésien.
        
        Équation du modèle :
        y ~ Normal(μ, σ)
        μ = β₀ + Σᵢ βᵢ · f(X_media_i) + Σⱼ γⱼ · X_control_j
        
        Où f(·) peut inclure adstock et/ou saturation (appliqués en preprocessing).
        
        Parameters
        ----------
        X_media : np.ndarray
            Matrice des dépenses media, shape (n_samples, n_media_channels).
        y : np.ndarray
            Vecteur cible (ventes), shape (n_samples,).
        X_control : np.ndarray, optional
            Matrice des variables de contrôle, shape (n_samples, n_controls).
        adstock_params : dict, optional
            Paramètres adstock si transformation déjà appliquée.
        saturation_params : dict, optional
            Paramètres saturation si transformation déjà appliquée.
        
        Returns
        -------
        pm.Model
            Modèle PyMC construit.
        """
        n_samples, n_media = X_media.shape
        n_control = X_control.shape[1] if X_control is not None else 0
        
        # Sauvegarder les données
        self.data = {
            'X_media': X_media,
            'y': y,
            'X_control': X_control,
            'n_samples': n_samples,
            'n_media': n_media,
            'n_control': n_control
        }
        
        # Configuration des priors
        if self.prior_config is None:
            self.prior_config = get_default_priors(n_media)
        
        validate_priors(self.prior_config, n_media)
        
        # Construire le modèle PyMC
        with pm.Model() as model:
            # === PRIORS ===
            
            # Intercept (baseline)
            intercept = create_prior(
                self.prior_config['intercept'],
                'intercept'
            )
            
            # Coefficients media (toujours positifs pour MMM)
            beta_media = create_prior(
                self.prior_config['media_coefficients'],
                'beta_media'
            )
            
            # Coefficients contrôle (si présents)
            if n_control > 0:
                control_config = self.prior_config.get(
                    'control_coefficients',
                    {'dist': 'Normal', 'mu': 0, 'sigma': 1.0, 'shape': n_control}
                )
                beta_control = create_prior(control_config, 'beta_control')
            
            # Bruit (écart-type)
            sigma = create_prior(
                self.prior_config['sigma'],
                'sigma'
            )
            
            # === TRANSFORMATIONS MEDIA ===
            # Note: Les transformations adstock/saturation sont appliquées
            # en preprocessing. Ici on ne fait que la régression linéaire.
            X_transformed = X_media
            
            # Les paramètres de transformation sont stockés dans metadata,
            # pas dans le modèle PyMC
            
            # === LIKELIHOOD ===
            
            # Contribution media
            media_contribution = pm.math.dot(X_transformed, beta_media)
            
            # Contribution contrôle
            if n_control > 0:
                control_contribution = pm.math.dot(X_control, beta_control)
                mu = intercept + media_contribution + control_contribution
            else:
                mu = intercept + media_contribution
            
            # Distribution des observations
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # === VARIABLES DÉTERMINISTES (pour analyse) ===
            
            # Contribution individuelle de chaque canal
            pm.Deterministic('media_contribution_total', media_contribution)
            
            for i in range(n_media):
                pm.Deterministic(
                    f'media_contribution_{i}',
                    X_transformed[:, i] * beta_media[i]
                )
        
        self.model = model
        return model
    
    def apply_transformations(
        self,
        X_media: np.ndarray,
        alpha: Optional[np.ndarray] = None,
        k: Optional[np.ndarray] = None,
        s: Optional[np.ndarray] = None,
        l_max: int = 8
    ) -> np.ndarray:
        """
        Applique les transformations adstock et saturation aux données media.
        
        Parameters
        ----------
        X_media : np.ndarray
            Dépenses media brutes.
        alpha : np.ndarray, optional
            Paramètres adstock par canal. Si None, utilise 0.5 par défaut.
        k : np.ndarray, optional
            Paramètres half_saturation par canal.
        s : np.ndarray, optional
            Paramètres slope par canal.
        l_max : int, default=8
            Lag maximum pour adstock.
        
        Returns
        -------
        np.ndarray
            Données transformées.
        """
        from models.transformations import geometric_adstock, hill_saturation
        
        X_transformed = X_media.copy()
        n_media = X_media.shape[1]
        
        # Adstock
        if self.use_adstock:
            if alpha is None:
                alpha = np.array([0.5] * n_media)
            
            X_transformed = geometric_adstock(
                X_transformed,
                alpha=alpha,
                l_max=l_max,
                normalize=True
            )
        
        # Saturation
        if self.use_saturation:
            if k is None:
                # Valeurs par défaut positives (ne pas utiliser mean sur données normalisées)
                k = np.array([1.0] * n_media)
            if s is None:
                s = np.array([1.0] * n_media)
            
            X_transformed = hill_saturation(
                X_transformed,
                half_saturation=k,
                slope=s
            )
        
        return X_transformed
    
    def fit(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        X_control: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        k: Optional[np.ndarray] = None,
        s: Optional[np.ndarray] = None,
        l_max: int = 8,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.95,
        **kwargs
    ) -> az.InferenceData:
        """
        Entraîne le modèle avec MCMC (NUTS sampler).
        
        Parameters
        ----------
        X_media : np.ndarray
            Dépenses media.
        y : np.ndarray
            Ventes observées.
        X_control : np.ndarray, optional
            Variables de contrôle.
        alpha : np.ndarray, optional
            Paramètres adstock par canal (si None, utilise 0.5).
        k : np.ndarray, optional
            Paramètres half_saturation par canal.
        s : np.ndarray, optional
            Paramètres slope par canal.
        l_max : int, default=8
            Lag maximum pour adstock.
        draws : int, default=2000
            Nombre d'échantillons MCMC.
        tune : int, default=1000
            Nombre d'échantillons de warm-up.
        chains : int, default=4
            Nombre de chaînes MCMC parallèles.
        target_accept : float, default=0.95
            Taux d'acceptation cible pour NUTS.
        **kwargs
            Arguments supplémentaires pour pm.sample.
        
        Returns
        -------
        az.InferenceData
            Trace MCMC avec résultats de l'inférence.
        """
        # Appliquer les transformations si activées
        X_transformed = self.apply_transformations(X_media, alpha, k, s, l_max)
        
        # Sauvegarder les paramètres de transformation
        adstock_params = {'alpha': alpha} if alpha is not None else None
        saturation_params = {'k': k, 's': s} if k is not None else None
        
        # Construire le modèle
        self.build_model(X_transformed, y, X_control, adstock_params, saturation_params)
        
        # Sampling MCMC
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
                **kwargs
            )
        
        # Sauvegarder les métadonnées
        self.metadata['draws'] = draws
        self.metadata['tune'] = tune
        self.metadata['chains'] = chains
        self.metadata['adstock_params'] = adstock_params
        self.metadata['saturation_params'] = saturation_params
        
        return self.trace
    
    def predict(
        self,
        X_media: np.ndarray,
        X_control: Optional[np.ndarray] = None,
        use_posterior_mean: bool = True
    ) -> np.ndarray:
        """
        Prédit les ventes pour de nouvelles dépenses media.
        
        Parameters
        ----------
        X_media : np.ndarray
            Nouvelles dépenses media.
        X_control : np.ndarray, optional
            Nouvelles variables de contrôle.
        use_posterior_mean : bool, default=True
            Si True, utilise la moyenne a posteriori. Sinon, échantillonne.
        
        Returns
        -------
        np.ndarray
            Prédictions de ventes.
        """
        if self.trace is None:
            raise ValueError("Modèle non entraîné. Appeler .fit() d'abord.")
        
        # Appliquer les mêmes transformations que lors du fit
        alpha = self.metadata.get('adstock_params', {}).get('alpha')
        k = self.metadata.get('saturation_params', {}).get('k')
        s = self.metadata.get('saturation_params', {}).get('s')
        
        X_transformed = self.apply_transformations(X_media, alpha, k, s)
        
        # Extraire les paramètres
        if use_posterior_mean:
            intercept = self.trace.posterior['intercept'].mean(dim=['chain', 'draw']).values
            beta_media = self.trace.posterior['beta_media'].mean(dim=['chain', 'draw']).values
            
            if X_control is not None:
                beta_control = self.trace.posterior['beta_control'].mean(dim=['chain', 'draw']).values
        
        # Calcul de la prédiction
        predictions = intercept + np.dot(X_transformed, beta_media)
        
        if X_control is not None and use_posterior_mean:
            predictions += np.dot(X_control, beta_control)
        
        return predictions
    
    def get_transformation_params(self) -> Dict:
        """
        Retourne les paramètres des transformations appliquées.
        
        Returns
        -------
        dict
            Paramètres adstock et saturation utilisés.
        """
        return {
            'adstock': self.metadata.get('adstock_params'),
            'saturation': self.metadata.get('saturation_params')
        }
    
    def get_channel_contributions(self) -> pd.DataFrame:
        """
        Calcule les contributions de chaque canal aux ventes.
        
        Returns
        -------
        pd.DataFrame
            Contributions moyennes par canal.
        """
        if self.trace is None:
            raise ValueError("Modèle non entraîné.")
        
        n_media = self.data['n_media']
        contributions = []
        
        for i in range(n_media):
            var_name = f'media_contribution_{i}'
            contrib = self.trace.posterior[var_name].mean(dim=['chain', 'draw']).values
            contributions.append({
                'channel': f'media_{i}',
                'total_contribution': contrib.sum(),
                'mean_contribution': contrib.mean()
            })
        
        return pd.DataFrame(contributions)
    
    def summary(self) -> pd.DataFrame:
        """
        Résumé des paramètres estimés (statistiques a posteriori).
        
        Returns
        -------
        pd.DataFrame
            Résumé avec moyennes, écarts-types, HDI, etc.
        """
        if self.trace is None:
            raise ValueError("Modèle non entraîné.")
        
        return az.summary(self.trace, var_names=['intercept', 'beta_media', 'sigma'])
