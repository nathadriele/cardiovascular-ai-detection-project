"""
Tests for model training and evaluation
Author: Nathalia Adriele
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train import create_models, train_model
from src.models.evaluate import ModelEvaluator


class TestModelCreation:
    """Test model creation."""

    def test_create_models(self):
        """Test creation of all models."""
        models = create_models()

        assert models is not None
        assert isinstance(models, dict)
        assert len(models) == 7

        # Check if all expected models are present
        expected_models = [
            'logistic_regression',
            'decision_tree',
            'random_forest',
            'xgboost',
            'gradient_boosting',
            'knn',
            'svm'
        ]

        for model_name in expected_models:
            assert model_name in models
            assert models[model_name] is not None

    def test_model_types(self):
        """Test if models are of correct type."""
        models = create_models()

        assert isinstance(models['logistic_regression'], LogisticRegression)
        assert isinstance(models['decision_tree'], DecisionTreeClassifier)
        assert isinstance(models['random_forest'], RandomForestClassifier)
        assert isinstance(models['gradient_boosting'], GradientBoostingClassifier)
        assert isinstance(models['knn'], KNeighborsClassifier)
        assert isinstance(models['svm'], SVC)


class TestModelTraining:
    """Test model training."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples)
        })

        y = np.random.choice([0, 1], n_samples)

        return X, y

    def test_train_logistic_regression(self, sample_data):
        """Test training logistic regression."""
        X, y = sample_data
        model = LogisticRegression(random_state=42)

        trained_model = train_model(model, X, y)

        assert trained_model is not None
        assert hasattr(trained_model, 'predict')
        assert hasattr(trained_model, 'predict_proba')

    def test_train_random_forest(self, sample_data):
        """Test training random forest."""
        X, y = sample_data
        model = RandomForestClassifier(random_state=42)

        trained_model = train_model(model, X, y)

        assert trained_model is not None
        assert hasattr(trained_model, 'predict')
        assert hasattr(trained_model, 'feature_importances_')

    def test_model_predictions(self, sample_data):
        """Test model predictions."""
        X, y = sample_data
        model = LogisticRegression(random_state=42)

        trained_model = train_model(model, X, y)
        predictions = trained_model.predict(X)

        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert all(p in [0, 1] for p in predictions)


class TestModelEvaluation:
    """Test model evaluation."""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.choice([0, 1], n_samples)
        y_pred = np.random.choice([0, 1], n_samples)
        y_pred_proba = np.random.uniform(0, 1, n_samples)

        return y_true, y_pred, y_pred_proba

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()

        assert evaluator is not None

    def test_calculate_metrics(self, sample_predictions):
        """Test metrics calculation."""
        y_true, y_pred, y_pred_proba = sample_predictions
        evaluator = ModelEvaluator()

        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)

        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics

        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1

    def test_confusion_matrix(self, sample_predictions):
        """Test confusion matrix calculation."""
        y_true, y_pred, _ = sample_predictions
        evaluator = ModelEvaluator()

        cm = evaluator.calculate_confusion_matrix(y_true, y_pred)

        assert cm is not None
        assert cm.shape == (2, 2)

        # Check that all values are non-negative
        assert np.all(cm >= 0)


class TestModelComparison:
    """Test model comparison."""

    @pytest.fixture
    def multiple_models(self):
        """Create multiple models for comparison."""
        return {
            'model_1': LogisticRegression(random_state=42),
            'model_2': RandomForestClassifier(random_state=42),
            'model_3': GradientBoostingClassifier(random_state=42)
        }

    @pytest.fixture
    def training_data(self):
        """Create training data."""
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples)
        })

        y = np.random.choice([0, 1], n_samples)

        return X, y

    def test_compare_models(self, multiple_models, training_data):
        """Test model comparison."""
        X, y = training_data
        evaluator = ModelEvaluator()

        results = {}

        for name, model in multiple_models.items():
            trained_model = train_model(model, X, y)
            y_pred = trained_model.predict(X)
            y_pred_proba = trained_model.predict_proba(X)[:, 1]

            metrics = evaluator.calculate_metrics(y, y_pred, y_pred_proba)
            results[name] = metrics

        # Check that all models have metrics
        assert len(results) == 3

        # Check that metrics are comparable
        for name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'roc_auc' in metrics


class TestModelSerialization:
    """Test model serialization."""

    def test_model_can_be_pickled(self):
        """Test that model can be serialized."""
        import pickle

        model = LogisticRegression(random_state=42)

        # Create simple training data
        X = np.random.randn(100, 3)
        y = np.random.choice([0, 1], 100)

        model.fit(X, y)

        # Try to pickle
        pickled = pickle.dumps(model)

        assert pickled is not None
        assert len(pickled) > 0

        # Try to unpickle
        unpickled_model = pickle.loads(pickled)

        # Check that unpickled model works
        predictions = unpickled_model.predict(X)
        assert len(predictions) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
