"""
Tests for LightGBM-based models.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from randommachine.lgbm_models import (
    RLGBM,
    RandomLGBMRegressor,
    RandomLGBMClassifier,
)
from randommachine.losses import MeanSquaredError, LogisticLoss


class TestRandomLGBMRegressor:
    """Test cases for RandomLGBMRegressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression dataset."""
        X, y = make_regression(n_samples=200, n_features=10, noise=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_initialization(self):
        """Test model initialization."""
        model = RandomLGBMRegressor(
            loss=MeanSquaredError,
            num_iterations=5,
            learning_rate=0.3,
            min_max_depth=3,
            max_max_depth=5,
            random_state=42,
        )

        assert model.num_iterations_ == 5
        assert model.learning_rate_ == 0.3
        assert len(model.base_learners_) == 3  # depths 3, 4, 5
        assert len(model.ensemble_) == 0  # Empty before training

    def test_fit_predict(self, regression_data):
        """Test fitting and prediction."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomLGBMRegressor(
            loss=MeanSquaredError,
            num_iterations=5,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            early_stopping_rounds=10,
            tree_iterations=10,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # Check ensemble is not empty after training
        assert len(model.ensemble_) > 0

        # Make predictions
        predictions = model.predict(X_test)

        # Check prediction shape
        assert predictions.shape == y_test.shape

        # Check predictions are reasonable
        mse = mean_squared_error(y_test, predictions)
        assert mse < np.var(y_test)  # Model should be better than mean

    def test_fit_with_eval(self, regression_data):
        """Test fitting with evaluation data."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomLGBMRegressor(
            num_iterations=5,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            early_stopping_rounds=10,
            tree_iterations=10,
            random_state=42,
        )

        model.fit(X_train, y_train, X_eval=X_test, y_eval=y_test)

        # Should have trained successfully
        assert len(model.ensemble_) > 0


class TestRandomLGBMClassifier:
    """Test cases for RandomLGBMClassifier."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_initialization(self):
        """Test model initialization."""
        model = RandomLGBMClassifier(
            loss=LogisticLoss,
            num_iterations=5,
            learning_rate=0.3,
            min_max_depth=3,
            max_max_depth=5,
            random_state=42,
        )

        assert model.num_iterations_ == 5
        assert model.learning_rate_ == 0.3
        assert len(model.base_learners_) == 3  # depths 3, 4, 5

    def test_fit_predict(self, classification_data):
        """Test fitting and prediction."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomLGBMClassifier(
            loss=LogisticLoss,
            num_iterations=5,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            early_stopping_rounds=10,
            tree_iterations=10,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Check prediction shape
        assert predictions.shape == y_test.shape

        # Check predictions are binary
        assert set(np.unique(predictions)).issubset({0, 1})

        # Check accuracy is reasonable
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.4  # Should be somewhat better than random

    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomLGBMClassifier(
            num_iterations=5,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            tree_iterations=10,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # Get probabilities
        probas = model.predict_proba(X_test)

        # Check shape
        assert probas.shape == y_test.shape

        # Check probabilities are in [0, 1]
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)
