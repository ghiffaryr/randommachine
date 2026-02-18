"""
Tests for generic random models (user-supplied base learners).
"""

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from randommachine.random_models import (
    RM,
    RandomRegressor,
    RandomClassifier,
)
from randommachine.losses import MeanSquaredError, LogisticLoss


def _reg_learners():
    return [
        LGBMRegressor(max_depth=d, n_estimators=10, learning_rate=0.3, verbose=-1)
        for d in range(3, 6)
    ]


def _clf_learners():
    return [
        LGBMRegressor(max_depth=d, n_estimators=10, learning_rate=0.3, verbose=-1)
        for d in range(3, 5)
    ] + [
        CatBoostRegressor(depth=d, iterations=10, learning_rate=0.3, verbose=False)
        for d in range(3, 5)
    ]


class TestRandomRegressor:
    """Test cases for RandomRegressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression dataset."""
        X, y = make_regression(n_samples=200, n_features=10, noise=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_initialization(self):
        """Test model initialization with explicit base_learners."""
        learners = _reg_learners()
        model = RandomRegressor(
            base_learners=learners,
            loss=MeanSquaredError,
            num_iterations=5,
            learning_rate=0.3,
            random_state=42,
        )

        assert model.num_iterations_ == 5
        assert model.learning_rate_ == 0.3
        assert len(model.base_learners_) == len(learners)
        assert len(model.ensemble_) == 0  # Empty before training

    def test_default_uniform_probabilities(self):
        """Probabilities default to uniform when not supplied."""
        learners = _reg_learners()
        model = RandomRegressor(base_learners=learners, random_state=42)
        expected_prob = 1 / len(learners)
        for p in model.probabilities_:
            assert abs(p - expected_prob) < 1e-9

    def test_fit_predict(self, regression_data):
        """Test fitting and prediction."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomRegressor(
            base_learners=_reg_learners(),
            loss=MeanSquaredError,
            num_iterations=5,
            learning_rate=0.5,
            early_stopping_rounds=10,
            random_state=42,
        )
        model.fit(X_train, y_train)

        assert len(model.ensemble_) > 0

        predictions = model.predict(X_test)
        assert predictions.shape == y_test.shape

        mse = mean_squared_error(y_test, predictions)
        assert mse < np.var(y_test) * 2

    def test_fit_with_eval(self, regression_data):
        """Test fitting with evaluation data."""
        X_train, X_test, y_train, y_test = regression_data

        model = RandomRegressor(
            base_learners=_reg_learners(),
            num_iterations=5,
            learning_rate=0.5,
            early_stopping_rounds=10,
            random_state=42,
        )
        model.fit(X_train, y_train, X_eval=X_test, y_eval=y_test)
        assert len(model.ensemble_) > 0

    def test_mixed_learners(self, regression_data):
        """Test that model accepts a heterogeneous pool of learners."""
        X_train, X_test, y_train, y_test = regression_data

        learners = [
            LGBMRegressor(max_depth=d, n_estimators=10, verbose=-1) for d in range(3, 5)
        ] + [
            XGBRegressor(max_depth=d, n_estimators=10, verbosity=0) for d in range(3, 5)
        ]
        model = RandomRegressor(
            base_learners=learners, num_iterations=3, learning_rate=0.5, random_state=42
        )
        model.fit(X_train, y_train)
        assert len(model.base_learners_) == len(learners)
        assert len(model.ensemble_) > 0


class TestRandomClassifier:
    """Test cases for RandomClassifier."""

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
        """Test model initialization with explicit base_learners."""
        learners = _clf_learners()
        model = RandomClassifier(
            base_learners=learners,
            loss=LogisticLoss,
            num_iterations=5,
            learning_rate=0.3,
            random_state=42,
        )

        assert model.num_iterations_ == 5
        assert model.learning_rate_ == 0.3
        assert len(model.base_learners_) == len(learners)

    def test_fit_predict(self, classification_data):
        """Test fitting and prediction."""
        X_train, X_test, y_train, y_test = classification_data

        model = RandomClassifier(
            base_learners=_clf_learners(),
            loss=LogisticLoss,
            num_iterations=5,
            learning_rate=0.5,
            early_stopping_rounds=10,
            random_state=42,
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

        model = RandomClassifier(
            base_learners=_clf_learners(),
            num_iterations=5,
            learning_rate=0.5,
            early_stopping_rounds=10,
            random_state=42,
        )
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape == y_test.shape
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_mixed_learners(self, classification_data):
        """Test that model accepts a heterogeneous pool of learners."""
        X_train, X_test, y_train, y_test = classification_data

        learners = [
            LGBMRegressor(max_depth=d, n_estimators=10, verbose=-1) for d in range(3, 5)
        ] + [
            XGBRegressor(max_depth=d, n_estimators=10, verbosity=0) for d in range(3, 5)
        ]
        model = RandomClassifier(
            base_learners=learners, num_iterations=3, learning_rate=0.5, random_state=42
        )
        model.fit(X_train, y_train)
        assert len(model.ensemble_) > 0
