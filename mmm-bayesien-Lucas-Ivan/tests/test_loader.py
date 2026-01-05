"""
Tests unitaires pour le data loader MMM.

Auteur : Ivan
Projet : MMM Bayésien - MSMIN5IN43
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.loader import (
    load_csv_data,
    validate_mmm_data,
    create_sample_data,
    split_train_test,
    get_dataset_summary
)


class TestLoadCSVData:
    """Tests pour le chargement de CSV."""
    
    def test_load_valid_csv(self, tmp_path):
        """Charge un CSV valide avec succès."""
        # Créer un CSV temporaire
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='W'),
            'sales': np.random.uniform(100, 200, 10),
            'tv_spend': np.random.uniform(50, 100, 10),
            'facebook_spend': np.random.uniform(30, 80, 10)
        })
        df.to_csv(csv_path, index=False)
        
        # Charger
        loaded_df = load_csv_data(csv_path)
        
        assert len(loaded_df) == 10
        assert 'sales' in loaded_df.columns
        assert 'tv_spend' in loaded_df.columns
        assert pd.api.types.is_datetime64_any_dtype(loaded_df['date'])
    
    def test_load_missing_file(self):
        """Erreur si le fichier n'existe pas."""
        with pytest.raises(FileNotFoundError):
            load_csv_data("non_existent_file.csv")
    
    def test_load_missing_target_column(self, tmp_path):
        """Erreur si la colonne cible est manquante."""
        csv_path = tmp_path / "missing_target.csv"
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='W'),
            'tv_spend': np.random.uniform(50, 100, 5)
        })
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Colonne cible"):
            load_csv_data(csv_path, target_column='sales')
    
    def test_auto_detect_media_columns(self, tmp_path):
        """Détecte automatiquement les colonnes media."""
        csv_path = tmp_path / "auto_detect.csv"
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='W'),
            'sales': np.random.uniform(100, 200, 5),
            'tv_spend': np.random.uniform(50, 100, 5),
            'google_cost': np.random.uniform(30, 80, 5),
            'impressions': np.random.randint(1000, 5000, 5)
        })
        df.to_csv(csv_path, index=False)
        
        loaded_df = load_csv_data(csv_path)
        
        # Doit détecter tv_spend, google_cost, impressions
        assert len(loaded_df.columns) >= 4
    
    def test_no_date_parsing(self, tmp_path):
        """Option pour ne pas parser les dates."""
        csv_path = tmp_path / "no_parse.csv"
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-08', '2023-01-15'],
            'sales': [100, 150, 200],
            'tv_spend': [50, 60, 70]
        })
        df.to_csv(csv_path, index=False)
        
        loaded_df = load_csv_data(csv_path, parse_dates=False)
        
        # Date doit rester en string
        assert loaded_df['date'].dtype == object


class TestValidateMMM:
    """Tests pour la validation de datasets."""
    
    def test_valid_dataset(self):
        """Dataset valide passe la validation."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=52, freq='W'),
            'sales': np.random.uniform(100, 200, 52),
            'tv_spend': np.random.uniform(50, 100, 52)
        })
        
        report = validate_mmm_data(df, media_columns=['tv_spend'])
        
        assert report['valid'] is True
        assert report['n_periods'] == 52
        assert report['n_media_channels'] == 1
    
    def test_too_few_periods(self):
        """Warning si moins de 52 périodes."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30, freq='W'),
            'sales': np.random.uniform(100, 200, 30),
            'tv_spend': np.random.uniform(50, 100, 30)
        })
        
        report = validate_mmm_data(df, media_columns=['tv_spend'], min_periods=52)
        
        assert len(report['warnings']) > 0
        assert 'périodes' in report['warnings'][0].lower()
    
    def test_missing_values_in_target(self):
        """Erreur si valeurs manquantes dans la cible."""
        df = pd.DataFrame({
            'sales': [100, np.nan, 200, 150],
            'tv_spend': [50, 60, 70, 80]
        })
        
        report = validate_mmm_data(df, media_columns=['tv_spend'], min_periods=4)
        
        assert report['valid'] is False
        assert any('manquantes' in e for e in report['errors'])
    
    def test_negative_sales_warning(self):
        """Warning si ventes négatives."""
        df = pd.DataFrame({
            'sales': [100, -50, 200, 150],
            'tv_spend': [50, 60, 70, 80]
        })
        
        report = validate_mmm_data(df, media_columns=['tv_spend'], min_periods=4)
        
        assert any('négatives' in w.lower() for w in report['warnings'])
    
    def test_missing_media_column(self):
        """Erreur si colonne media manquante."""
        df = pd.DataFrame({
            'sales': [100, 150, 200],
            'tv_spend': [50, 60, 70]
        })
        
        report = validate_mmm_data(df, media_columns=['tv_spend', 'missing_column'])
        
        assert report['valid'] is False
        assert any('missing_column' in e for e in report['errors'])


class TestCreateSampleData:
    """Tests pour la génération de données synthétiques."""
    
    def test_default_sample_data(self):
        """Génère un dataset avec paramètres par défaut."""
        df = create_sample_data()
        
        assert len(df) == 104  # 2 ans hebdomadaires
        assert 'date' in df.columns
        assert 'sales' in df.columns
        assert len([c for c in df.columns if 'media' in c]) == 3
    
    def test_custom_parameters(self):
        """Génère un dataset avec paramètres personnalisés."""
        df = create_sample_data(n_periods=52, n_media_channels=5, seed=123)
        
        assert len(df) == 52
        assert len([c for c in df.columns if 'media' in c]) == 5
    
    def test_no_negative_sales(self):
        """Les ventes générées ne sont jamais négatives."""
        df = create_sample_data(n_periods=200, seed=999)
        
        assert (df['sales'] >= 0).all()
    
    def test_dates_are_sequential(self):
        """Les dates sont séquentielles."""
        df = create_sample_data(n_periods=10)
        
        assert df['date'].is_monotonic_increasing
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
    
    def test_reproducibility(self):
        """Même seed produit mêmes données."""
        df1 = create_sample_data(seed=42)
        df2 = create_sample_data(seed=42)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_different_seeds_different_data(self):
        """Seeds différents produisent données différentes."""
        df1 = create_sample_data(seed=42)
        df2 = create_sample_data(seed=99)
        
        assert not df1['sales'].equals(df2['sales'])


class TestSplitTrainTest:
    """Tests pour la division train/test."""
    
    def test_default_split_ratio(self):
        """Split 80/20 par défaut."""
        df = create_sample_data(n_periods=100)
        train, test = split_train_test(df)
        
        assert len(train) == 80
        assert len(test) == 20
    
    def test_custom_split_ratio(self):
        """Split personnalisé."""
        df = create_sample_data(n_periods=100)
        train, test = split_train_test(df, train_ratio=0.7)
        
        assert len(train) == 70
        assert len(test) == 30
    
    def test_chronological_split(self):
        """Le split respecte l'ordre chronologique."""
        df = create_sample_data(n_periods=50)
        train, test = split_train_test(df)
        
        # Train doit avoir les dates les plus anciennes
        assert train['date'].max() < test['date'].min()
    
    def test_no_overlap(self):
        """Pas de chevauchement entre train et test."""
        df = create_sample_data(n_periods=100)
        df['id'] = range(len(df))
        
        train, test = split_train_test(df)
        
        train_ids = set(train['id'])
        test_ids = set(test['id'])
        
        assert len(train_ids.intersection(test_ids)) == 0


class TestGetDatasetSummary:
    """Tests pour le résumé de dataset."""
    
    def test_summary_structure(self):
        """Le résumé contient les clés attendues."""
        df = create_sample_data(n_periods=52)
        summary = get_dataset_summary(df)
        
        assert 'n_periods' in summary
        assert 'n_columns' in summary
        assert 'date_range' in summary
        assert 'numeric_stats' in summary
    
    def test_numeric_stats(self):
        """Les stats numériques sont calculées correctement."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='W'),
            'sales': [100] * 10
        })
        
        summary = get_dataset_summary(df)
        
        assert summary['numeric_stats']['sales']['mean'] == 100
        assert summary['numeric_stats']['sales']['std'] == 0
        assert summary['numeric_stats']['sales']['min'] == 100
        assert summary['numeric_stats']['sales']['max'] == 100
    
    def test_date_range(self):
        """La plage de dates est extraite."""
        start = pd.Timestamp('2023-01-01')
        end = pd.Timestamp('2023-12-31')
        
        df = pd.DataFrame({
            'date': pd.date_range(start, end, freq='W'),
            'sales': range(53)
        })
        
        summary = get_dataset_summary(df)
        
        assert summary['date_range'][0] == start
        assert summary['date_range'][1] == end
    
    def test_missing_values_counted(self):
        """Les valeurs manquantes sont comptées."""
        df = pd.DataFrame({
            'sales': [100, np.nan, 200, np.nan, 300]
        })
        
        summary = get_dataset_summary(df)
        
        assert summary['numeric_stats']['sales']['missing'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
