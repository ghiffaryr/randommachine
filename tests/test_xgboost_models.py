"""
Tests for XGBoost-based models.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from randommachine.xgboost_models import (
    RXGBM,
    RandomXGBRegressor,
    RandomXGBClassifier,
)
from randommachine.losses import MeanSquaredError, LogisticLoss


class TestRandomXGBRegressor:
    """Test cases for RandomXGBRegressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression dataset."""
        X, y = make_regression(
            n_samples=200, n_features=10, noise=5, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_initialization(self):
        """Test model initialization."""
        model = RandomXGBRegressor(
            loss=MeanSquaredError,
            num_iterations=5,
            learning_rate=0.3,
            min_max_depth=3,
            max_max_depth=5,
            random_state=42
        )

        assert model.num_iterations_ == 5
        assert model.learning_rate_ == 0.3
        assert len(model.base_learners_) == 3  # depths 3, 4, 5
        assert len(model.ensemble_) == 0  # Empty before training

    def test_uniform_probabilities(self):
        """Probabilities should be uniform across depth range."""
        model = RandomXGBRegressor(min_max_depth=3, max_max_depth=5, random_state=42)
        n = len(model.base_learners_)
        for p in model.probabilities_:
            assert abs(p - 1 / n) < 1e-9

    def test_fit_predict(self, regression_data):
        """Test fitting and prediction."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomXGBRegressor(
            loss=MeanSquaredError,
            num_iterations=5,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            early_stopping_rounds=10,
            tree_iterations=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        assert len(model.ensemble_) > 0

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape

        mse = mean_squared_error(y_test, predictions)
        assert mse < np.var(y_test) * 2

    def test_fit_with_eval(self, regression_data):
        """Test fitting with evaluation data (early stopping path)."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomXGBRegressor(
            num_iterations=10,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            early_stopping_rounds=5,
            tree_iterations=10,
            random_state=42
        )

        model.fit(X_train, y_train, X_eval=X_test, y_eval=y_test)
        assert len(model.ensemble_) > 0

    def test_predict_returns_continuous_values(self, regression_data):
        """Regressor should return real-valued predictions."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomXGBRegressor(
            num_iterations=5,
            min_max_depth=3,
            max_max_depth=4,
            tree_iterations=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Should not be binary
        assert len(np.unique(predictions)) > 2


class TestRandomXGBClassifier:
    """Test cases for RandomXGBClassifier."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification dataset."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=8,
            n_redundant=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_initialization(self):
        """Test model initialization."""
        model = RandomXGBClassifier(
            loss=LogisticLoss,
            num_iterations=5,
            learning_rate=0.3,
            min_max_depth=3,
            max_max_depth=5,
            random_state=42
        )

        assert model.num_iterations_ == 5
        assert model.learning_rate_ == 0.3
        assert len(model.base_learners_) == 3  # depths 3, 4, 5

    def test_fit_predict(self, classification_data):
        """Test fitting and prediction."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomXGBClassifier(
            loss=LogisticLoss,
            num_iterations=5,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            early_stopping_rounds=10,
            tree_iterations=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape
        assert set(np.unique(predictions)).issubset({0, 1})

        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.4

    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomXGBClassifier(
            num_iterations=5,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            tree_iterations=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape == y_test.shape
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_fit_with_eval(self, classification_data):
        """Test fitting with evaluation data."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomXGBClassifier(
            num_iterations=10,
            learning_rate=0.5,
            min_max_depth=3,
            max_max_depth=4,
            early_stopping_rounds=5,
            tree_iterations=10,
            random_state=42
        )

        model.fit(X_train, y_train, X_eval=X_test, y_eval=y_test)
        assert len(model.ensemble_) > 0

    def test_predict_binary_labels(self, classification_data):
        """Classifier labels must be 0 or 1."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomXGBClassifier(
            num_iterations=5,
            min_max_depth=3,
            max_max_depth=4,
            tree_iterations=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert set(np.unique(predictions)).issubset({0, 1})
