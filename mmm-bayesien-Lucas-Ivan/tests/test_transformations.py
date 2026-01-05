"""
Tests unitaires pour les transformations MMM (adstock et saturation).

Ce module teste la robustesse et la justesse des transformations :
- Validation des cas limites
- Vérification des propriétés mathématiques
- Tests de forme des sorties
- Gestion des erreurs

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

import sys
import os

# Ajouter le chemin src au PYTHONPATH pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.transformations import (
    geometric_adstock,
    hill_saturation,
    adstock_and_saturation,
    get_adstock_decay_weights,
    get_effective_reach_curve
)


class TestGeometricAdstock:
    """Tests pour la transformation d'adstock géométrique."""
    
    def test_adstock_zero_alpha_no_carryover(self):
        """Avec alpha=0, pas d'effet de persistance (sortie = entrée)."""
        spend = np.array([100, 50, 30, 20, 10])
        result = geometric_adstock(spend, alpha=0.0, l_max=5)
        
        # Avec alpha=0, pas de carry-over
        assert_array_almost_equal(result, spend)
    
    def test_adstock_high_alpha_strong_carryover(self):
        """Avec alpha élevé, fort effet de persistance."""
        spend = np.array([100, 0, 0, 0, 0])
        result = geometric_adstock(spend, alpha=0.5, l_max=4, normalize=False)
        
        # Attendu : [100, 50, 25, 12.5, 6.25] (décroissance géométrique)
        expected = np.array([100, 50, 25, 12.5, 6.25])
        assert_array_almost_equal(result, expected, decimal=2)
    
    def test_adstock_normalization(self):
        """La normalisation préserve l'échelle totale des dépenses."""
        spend = np.array([100, 0, 0, 0, 0])
        
        # Sans normalisation
        result_no_norm = geometric_adstock(spend, alpha=0.5, l_max=10, normalize=False)
        
        # Avec normalisation
        result_norm = geometric_adstock(spend, alpha=0.5, l_max=10, normalize=True)
        
        # La somme normalisée devrait être proche de la somme originale
        assert np.sum(spend) == pytest.approx(np.sum(result_norm), rel=0.1)
        
        # La forme de décroissance devrait être similaire (proportionnelle)
        assert result_norm[0] > result_norm[1] > result_norm[2]
    
    def test_adstock_output_shape(self):
        """La sortie doit avoir la même forme que l'entrée."""
        # Test 1D
        spend_1d = np.array([10, 20, 30, 40])
        result_1d = geometric_adstock(spend_1d, alpha=0.3, l_max=2)
        assert result_1d.shape == spend_1d.shape
        
        # Test 2D (multi-canaux)
        spend_2d = np.array([[10, 20], [30, 40], [50, 60]])
        result_2d = geometric_adstock(spend_2d, alpha=0.3, l_max=2)
        assert result_2d.shape == spend_2d.shape
    
    def test_adstock_multi_channel_different_alphas(self):
        """Chaque canal peut avoir son propre taux de rétention."""
        spend = np.array([[100, 100], [0, 0], [0, 0]])
        alpha = np.array([0.2, 0.8])  # Canal 1 faible persistance, Canal 2 forte
        
        result = geometric_adstock(spend, alpha=alpha, l_max=2, normalize=False)
        
        # Canal 1 (alpha=0.2) devrait décroître plus vite que Canal 2 (alpha=0.8)
        assert result[1, 0] < result[1, 1]  # À t=1, canal 2 a plus d'effet résiduel
    
    def test_adstock_invalid_alpha(self):
        """Alpha doit être dans [0, 1)."""
        spend = np.array([10, 20, 30])
        
        # Alpha négatif
        with pytest.raises(ValueError, match="alpha doit être dans"):
            geometric_adstock(spend, alpha=-0.1, l_max=2)
        
        # Alpha >= 1
        with pytest.raises(ValueError, match="alpha doit être dans"):
            geometric_adstock(spend, alpha=1.0, l_max=2)
        
        with pytest.raises(ValueError, match="alpha doit être dans"):
            geometric_adstock(spend, alpha=1.5, l_max=2)
    
    def test_adstock_invalid_l_max(self):
        """l_max doit être >= 1."""
        spend = np.array([10, 20, 30])
        
        with pytest.raises(ValueError, match="l_max doit être"):
            geometric_adstock(spend, alpha=0.5, l_max=0)
    
    def test_adstock_decay_weights_sum(self):
        """Les poids normalisés doivent sommer à 1."""
        weights = get_adstock_decay_weights(alpha=0.7, l_max=10, normalize=True)
        assert np.sum(weights) == pytest.approx(1.0)
    
    def test_adstock_decay_weights_no_normalization(self):
        """Sans normalisation, le premier poids devrait être 1."""
        weights = get_adstock_decay_weights(alpha=0.5, l_max=5, normalize=False)
        assert weights[0] == 1.0
        assert weights[1] == 0.5
        assert weights[2] == 0.25


class TestHillSaturation:
    """Tests pour la transformation de saturation de Hill."""
    
    def test_saturation_at_zero(self):
        """À dépense nulle, saturation = 0."""
        spend = np.array([0, 10, 100])
        result = hill_saturation(spend, half_saturation=50, slope=1.0)
        
        assert result[0] == 0.0
    
    def test_saturation_at_half_saturation_point(self):
        """Au point de demi-saturation, output ≈ 0.5."""
        spend = np.array([50])
        result = hill_saturation(spend, half_saturation=50, slope=1.0)
        
        assert result[0] == pytest.approx(0.5, rel=0.01)
    
    def test_saturation_bounded_between_0_and_1(self):
        """La sortie doit être dans [0, 1]."""
        spend = np.linspace(0, 1000, 100)
        result = hill_saturation(spend, half_saturation=100, slope=1.0)
        
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_saturation_monotonic_increasing(self):
        """La saturation doit être croissante (plus de dépense = plus d'effet)."""
        spend = np.array([10, 50, 100, 200, 500])
        result = hill_saturation(spend, half_saturation=100, slope=1.0)
        
        # Vérifier que chaque valeur est >= à la précédente
        assert np.all(np.diff(result) >= 0)
    
    def test_saturation_slope_effect(self):
        """Une pente plus élevée doit créer une courbe plus abrupte."""
        spend = np.linspace(0, 200, 50)
        
        result_gentle = hill_saturation(spend, half_saturation=100, slope=0.5)
        result_steep = hill_saturation(spend, half_saturation=100, slope=2.0)
        
        # À faible dépense, pente douce donne plus d'effet
        assert result_gentle[10] > result_steep[10]
        
        # À forte dépense, pente abrupte rattrape
        assert result_steep[-1] >= result_gentle[-1]
    
    def test_saturation_multi_channel(self):
        """Chaque canal peut avoir son propre paramètre de saturation."""
        spend = np.array([[100, 100], [200, 200]])
        k = np.array([50, 150])  # Canal 1 sature plus vite
        
        result = hill_saturation(spend, half_saturation=k, slope=1.0)
        
        # Canal 1 devrait être plus saturé que Canal 2
        assert result[0, 0] > result[0, 1]
    
    def test_saturation_invalid_half_saturation(self):
        """half_saturation doit être > 0."""
        spend = np.array([10, 20, 30])
        
        with pytest.raises(ValueError, match="half_saturation doit être"):
            hill_saturation(spend, half_saturation=0)
        
        with pytest.raises(ValueError, match="half_saturation doit être"):
            hill_saturation(spend, half_saturation=-10)
    
    def test_saturation_invalid_slope(self):
        """slope doit être > 0."""
        spend = np.array([10, 20, 30])
        
        with pytest.raises(ValueError, match="slope doit être"):
            hill_saturation(spend, half_saturation=50, slope=0)
        
        with pytest.raises(ValueError, match="slope doit être"):
            hill_saturation(spend, half_saturation=50, slope=-1)
    
    def test_saturation_negative_spend_warning(self):
        """Les dépenses négatives doivent déclencher un avertissement."""
        spend = np.array([-10, 20, 30])
        
        with pytest.warns(UserWarning, match="valeurs négatives"):
            result = hill_saturation(spend, half_saturation=50, slope=1.0)
        
        # Les valeurs négatives doivent être traitées comme 0
        assert result[0] == 0.0
    
    def test_effective_reach_curve(self):
        """La fonction utilitaire doit générer une courbe valide."""
        spend_range = np.linspace(0, 500, 100)
        curve = get_effective_reach_curve(spend_range, half_saturation=100, slope=1.0)
        
        assert curve.shape == spend_range.shape
        assert np.all(curve >= 0)
        assert np.all(curve <= 1)


class TestAdstockAndSaturation:
    """Tests pour le pipeline complet (adstock + saturation)."""
    
    def test_pipeline_output_shape(self):
        """Le pipeline doit préserver la forme de l'entrée."""
        spend = np.array([100, 50, 30, 20, 10])
        result = adstock_and_saturation(
            spend,
            alpha=0.5,
            half_saturation=80,
            l_max=4,
            slope=1.0
        )
        
        assert result.shape == spend.shape
    
    def test_pipeline_bounded_output(self):
        """La sortie finale doit être dans [0, 1] (grâce à la saturation)."""
        spend = np.linspace(0, 1000, 100)
        result = adstock_and_saturation(
            spend,
            alpha=0.3,
            half_saturation=200,
            l_max=5
        )
        
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_pipeline_multi_channel(self):
        """Le pipeline doit fonctionner avec plusieurs canaux."""
        spend = np.array([[100, 50], [80, 40], [60, 30]])
        alpha = np.array([0.4, 0.6])
        k = np.array([70, 90])
        
        result = adstock_and_saturation(
            spend,
            alpha=alpha,
            half_saturation=k,
            l_max=3
        )
        
        assert result.shape == spend.shape
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_pipeline_with_zero_spending(self):
        """Dépenses nulles doivent donner sortie nulle."""
        spend = np.zeros(10)
        result = adstock_and_saturation(
            spend,
            alpha=0.5,
            half_saturation=100,
            l_max=5
        )
        
        assert_array_almost_equal(result, np.zeros(10))
    
    def test_pipeline_order_matters(self):
        """L'ordre adstock → saturation est différent de saturation → adstock."""
        spend = np.array([100, 50, 25, 12, 6])
        
        # Ordre correct : adstock puis saturation
        correct_result = adstock_and_saturation(
            spend,
            alpha=0.5,
            half_saturation=80,
            l_max=3
        )
        
        # Ordre inversé (pour comparaison conceptuelle)
        wrong_saturated = hill_saturation(spend, half_saturation=80, slope=1.0)
        wrong_result = geometric_adstock(wrong_saturated, alpha=0.5, l_max=3)
        
        # Les résultats doivent être différents
        assert not np.allclose(correct_result, wrong_result)


class TestEdgeCases:
    """Tests pour les cas limites et validations."""
    
    def test_large_values_no_overflow(self):
        """Les grandes valeurs ne doivent pas causer d'overflow."""
        spend = np.array([1e6, 1e7, 1e8])
        
        # Adstock
        result_adstock = geometric_adstock(spend, alpha=0.9, l_max=10)
        assert np.all(np.isfinite(result_adstock))
        
        # Saturation
        result_saturation = hill_saturation(spend, half_saturation=1e5, slope=2.0)
        assert np.all(np.isfinite(result_saturation))
    
    def test_very_small_values(self):
        """Les très petites valeurs doivent être gérées correctement."""
        spend = np.array([1e-10, 1e-8, 1e-6])
        
        result = adstock_and_saturation(
            spend,
            alpha=0.5,
            half_saturation=1e-5,
            l_max=3
        )
        
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
    
    def test_single_value_input(self):
        """Les inputs de taille 1 doivent fonctionner."""
        spend = np.array([100])
        
        result_adstock = geometric_adstock(spend, alpha=0.5, l_max=2)
        assert result_adstock.shape == (1,)
        
        result_saturation = hill_saturation(spend, half_saturation=50, slope=1.0)
        assert result_saturation.shape == (1,)
    
    def test_consistent_dtypes(self):
        """Les types de données doivent être préservés."""
        spend_float32 = np.array([10, 20, 30], dtype=np.float32)
        
        result = geometric_adstock(spend_float32, alpha=0.5, l_max=2)
        # Le résultat peut être promu à float64, c'est normal
        assert result.dtype in [np.float32, np.float64]


if __name__ == "__main__":
    # Permet d'exécuter les tests directement avec python
    pytest.main([__file__, "-v", "--tb=short"])
