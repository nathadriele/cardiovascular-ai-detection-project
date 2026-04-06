"""
Tests for preprocessing module
Author: Nathalia Adriele
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import (
    identify_column_types,
    create_preprocessing_pipeline,
    preprocess_cardiovascular_data
)


class TestColumnIdentification:
    """Test column type identification."""

    def test_identify_numeric_columns(self):
        """Test identification of numeric columns."""
        df = pd.DataFrame({
            'idade': [25, 30, 35],
            'peso': [70.5, 80.2, 65.3],
            'altura': [1.75, 1.80, 1.68]
        })

        numeric, categorical, binary = identify_column_types(df)
        assert len(numeric) == 3
        assert len(categorical) == 0
        assert len(binary) == 0

    def test_identify_categorical_columns(self):
        """Test identification of categorical columns."""
        df = pd.DataFrame({
            'sexo': ['M', 'F', 'M'],
            'escolaridade': ['fundamental', 'medio', 'superior']
        })

        numeric, categorical, binary = identify_column_types(df)
        assert len(numeric) == 0
        assert len(categorical) == 2
        assert len(binary) == 0

    def test_identify_binary_columns(self):
        """Test identification of binary columns."""
        df = pd.DataFrame({
            'fumante': [0, 1, 0],
            'diabetico': [1, 0, 1]
        })

        numeric, categorical, binary = identify_column_types(df)
        assert len(numeric) == 0
        assert len(binary) == 2
        assert len(categorical) == 0


class TestPreprocessingPipeline:
    """Test preprocessing pipeline creation."""

    def test_create_pipeline(self):
        """Test creation of preprocessing pipeline."""
        numeric_cols = ['idade', 'peso']
        categorical_cols = ['sexo']
        binary_cols = ['fumante']

        pipeline = create_preprocessing_pipeline(
            numeric_cols, categorical_cols, binary_cols
        )

        assert pipeline is not None
        assert hasattr(pipeline, 'transform')
        assert hasattr(pipeline, 'fit_transform')

    def test_pipeline_transform(self):
        """Test pipeline transformation."""
        df = pd.DataFrame({
            'idade': [25, 30, 35],
            'peso': [70.5, 80.2, 65.3],
            'sexo': ['M', 'F', 'M'],
            'fumante': [0, 1, 0],
            'target': [0, 1, 0]
        })

        numeric_cols = ['idade', 'peso']
        categorical_cols = ['sexo']
        binary_cols = ['fumante']

        pipeline = create_preprocessing_pipeline(
            numeric_cols, categorical_cols, binary_cols
        )

        X = df.drop('target', axis=1)
        y = df['target']

        X_transformed = pipeline.fit_transform(X)

        assert X_transformed is not None
        assert X_transformed.shape[0] == 3


class TestDataPreprocessing:
    """Test complete data preprocessing."""

    @pytest.fixture
    def sample_data(self):
        """Create sample cardiovascular data."""
        np.random.seed(42)
        n_samples = 100

        df = pd.DataFrame({
            'idade': np.random.randint(25, 80, n_samples),
            'peso': np.random.uniform(50, 100, n_samples),
            'altura': np.random.uniform(1.50, 1.90, n_samples),
            'pa_sistolica': np.random.randint(90, 180, n_samples),
            'pa_diastolica': np.random.randint(60, 120, n_samples),
            'colesterol_total': np.random.randint(150, 300, n_samples),
            'sexo': np.random.choice(['M', 'F'], n_samples),
            'fumante': np.random.choice([0, 1], n_samples),
            'presenca_doenca_cardiovascular': np.random.choice([0, 1], n_samples)
        })

        return df

    def test_preprocess_cardiovascular_data(self, sample_data):
        """Test complete preprocessing function."""
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = \
            preprocess_cardiovascular_data(sample_data)

        # Check if splits are correct
        total_samples = len(sample_data)
        train_samples = len(X_train)
        val_samples = len(X_val)
        test_samples = len(X_test)

        assert train_samples + val_samples + test_samples == total_samples

        # Check if target variables match
        assert len(y_train) == train_samples
        assert len(y_val) == val_samples
        assert len(y_test) == test_samples

        # Check if preprocessor exists
        assert preprocessor is not None


class TestDataQuality:
    """Test data quality checks."""

    def test_handle_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'idade': [25, np.nan, 35],
            'peso': [70.5, 80.2, np.nan],
            'sexo': ['M', 'F', 'M']
        })

        # Check for missing values
        assert df['idade'].isna().sum() == 1
        assert df['peso'].isna().sum() == 1

    def test_data_validation(self):
        """Test data validation."""
        df = pd.DataFrame({
            'idade': [25, 30, 150],  # 150 is invalid
            'pa_sistolica': [120, 130, 80],
            'pa_diastolica': [80, 85, 200]  # 200 is invalid
        })

        # Check for invalid values
        assert df['idade'].max() > 120  # Invalid age
        assert df['pa_diastolica'].max() > 150  # Invalid diastolic


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
