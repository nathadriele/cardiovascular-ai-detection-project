"""
Tests for feature engineering module
Author: Nathalia Adriele
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_engineering import (
    calculate_bmi,
    categorize_bmi,
    categorize_age,
    calculate_non_hdl_cholesterol,
    calculate_ldl_hdl_ratio,
    calculate_triglycerides_hdl_ratio
)


class TestBMICalculations:
    """Test BMI calculations and categorization."""

    def test_calculate_bmi(self):
        """Test BMI calculation."""
        df = pd.DataFrame({
            'peso': [70, 80, 60],
            'altura': [1.75, 1.80, 1.65]
        })

        bmi = calculate_bmi(df)

        assert len(bmi) == 3
        assert abs(bmi[0] - 22.86) < 0.1  # 70 / (1.75^2)
        assert abs(bmi[1] - 24.69) < 0.1  # 80 / (1.80^2)
        assert abs(bmi[2] - 22.03) < 0.1  # 60 / (1.65^2)

    def test_categorize_bmi_underweight(self):
        """Test BMI categorization for underweight."""
        bmi = pd.Series([18.4])
        category = categorize_bmi(bmi)
        assert category.iloc[0] == 'abaixo_peso'

    def test_categorize_bmi_normal(self):
        """Test BMI categorization for normal weight."""
        bmi = pd.Series([22.5])
        category = categorize_bmi(bmi)
        assert category.iloc[0] == 'normal'

    def test_categorize_bmi_overweight(self):
        """Test BMI categorization for overweight."""
        bmi = pd.Series([27.5])
        category = categorize_bmi(bmi)
        assert category.iloc[0] == 'sobrepeso'

    def test_categorize_bmi_obesity(self):
        """Test BMI categorization for obesity."""
        bmi = pd.Series([32.5])
        category = categorize_bmi(bmi)
        assert category.iloc[0] == 'obesidade'


class TestAgeCategorization:
    """Test age categorization."""

    def test_categorize_age_young(self):
        """Test age categorization for young adults."""
        idade = pd.Series([30])
        category = categorize_age(idade)
        assert category.iloc[0] == '25-34'

    def test_categorize_age_middle(self):
        """Test age categorization for middle-aged adults."""
        idade = pd.Series([50])
        category = categorize_age(idade)
        assert category.iloc[0] == '45-54'

    def test_categorize_age_elderly(self):
        """Test age categorization for elderly."""
        idade = pd.Series([70])
        category = categorize_age(idade)
        assert category.iloc[0] == '65-74'


class TestCholesterolCalculations:
    """Test cholesterol calculations."""

    def test_calculate_non_hdl_cholesterol(self):
        """Test non-HDL cholesterol calculation."""
        df = pd.DataFrame({
            'colesterol_total': [200, 220, 180],
            'colesterol_hdl': [50, 45, 55]
        })

        non_hdl = calculate_non_hdl_cholesterol(df)

        assert len(non_hdl) == 3
        assert non_hdl.iloc[0] == 150  # 200 - 50
        assert non_hdl.iloc[1] == 175  # 220 - 45
        assert non_hdl.iloc[2] == 125  # 180 - 55

    def test_calculate_ldl_hdl_ratio(self):
        """Test LDL/HDL ratio calculation."""
        df = pd.DataFrame({
            'colesterol_ldl': [130, 140, 120],
            'colesterol_hdl': [50, 45, 55]
        })

        ratio = calculate_ldl_hdl_ratio(df)

        assert len(ratio) == 3
        assert abs(ratio.iloc[0] - 2.6) < 0.1  # 130 / 50
        assert abs(ratio.iloc[1] - 3.11) < 0.1  # 140 / 45
        assert abs(ratio.iloc[2] - 2.18) < 0.1  # 120 / 55

    def test_calculate_triglycerides_hdl_ratio(self):
        """Test triglycerides/HDL ratio calculation."""
        df = pd.DataFrame({
            'triglicerides': [150, 200, 100],
            'colesterol_hdl': [50, 45, 55]
        })

        ratio = calculate_triglycerides_hdl_ratio(df)

        assert len(ratio) == 3
        assert abs(ratio.iloc[0] - 3.0) < 0.1  # 150 / 50
        assert abs(ratio.iloc[1] - 4.44) < 0.1  # 200 / 45
        assert abs(ratio.iloc[2] - 1.82) < 0.1  # 100 / 55


class TestFeatureEngineering:
    """Test complete feature engineering pipeline."""

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
            'colesterol_hdl': np.random.randint(30, 80, n_samples),
            'colesterol_ldl': np.random.randint(80, 180, n_samples),
            'triglicerides': np.random.randint(50, 300, n_samples),
            'sexo': np.random.choice(['M', 'F'], n_samples),
            'fumante': np.random.choice([0, 1], n_samples)
        })

        return df

    def test_complete_feature_engineering(self, sample_data):
        """Test complete feature engineering pipeline."""
        # This test will verify that all features can be created
        # without errors

        # BMI calculation
        sample_data['imc'] = calculate_bmi(sample_data)
        assert 'imc' in sample_data.columns
        assert sample_data['imc'].notna().all()

        # BMI categorization
        sample_data['imc_categoria'] = categorize_bmi(sample_data['imc'])
        assert 'imc_categoria' in sample_data.columns
        assert sample_data['imc_categoria'].notna().all()

        # Age categorization
        sample_data['faixa_etaria'] = categorize_age(sample_data['idade'])
        assert 'faixa_etaria' in sample_data.columns
        assert sample_data['faixa_etaria'].notna().all()

        # Cholesterol calculations
        sample_data['colesterol_nao_hdl'] = calculate_non_hdl_cholesterol(sample_data)
        assert 'colesterol_nao_hdl' in sample_data.columns
        assert sample_data['colesterol_nao_hdl'].notna().all()

        sample_data['ratio_ldl_hdl'] = calculate_ldl_hdl_ratio(sample_data)
        assert 'ratio_ldl_hdl' in sample_data.columns
        assert sample_data['ratio_ldl_hdl'].notna().all()


class TestDataValidation:
    """Test data validation in feature engineering."""

    def test_handle_missing_values_in_features(self):
        """Test handling of missing values in feature engineering."""
        df = pd.DataFrame({
            'peso': [70, np.nan, 60],
            'altura': [1.75, 1.80, np.nan]
        })

        # Should handle missing values gracefully
        bmi = calculate_bmi(df)

        # Check that function returns something even with missing values
        assert bmi is not None
        assert len(bmi) == 3

    def test_handle_edge_cases(self):
        """Test handling of edge cases."""
        df = pd.DataFrame({
            'colesterol_total': [0, 300, 150],  # 0 is edge case
            'colesterol_hdl': [50, 0, 45]  # 0 is edge case (division)
        })

        non_hdl = calculate_non_hdl_cholesterol(df)

        # Should handle edge cases
        assert non_hdl is not None
        assert len(non_hdl) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
