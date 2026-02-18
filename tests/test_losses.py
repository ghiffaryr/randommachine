"""
Tests for loss functions.
"""

import numpy as np
import pytest
from randommachine.losses import MeanSquaredError, LogisticLoss


class TestMeanSquaredError:
    """Test cases for MeanSquaredError loss function."""

    def test_compute_derivatives(self):
        """Test gradient and hessian computation."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([1.5, 2.5, 2.5, 4.5, 4.5])

        g, h = MeanSquaredError.compute_derivatives(y, f)

        # Check gradient
        expected_g = 2 * (f - y)
        np.testing.assert_array_almost_equal(g, expected_g)

        # Check hessian
        expected_h = 2.0 * np.ones(y.shape[0])
        np.testing.assert_array_almost_equal(h, expected_h)

    def test_loss(self):
        """Test loss computation."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        loss = MeanSquaredError.loss(y, f)

        # Perfect predictions should have zero loss
        assert loss == 0.0

        # Test with non-zero error
        f = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        loss = MeanSquaredError.loss(y, f)
        expected_loss = np.mean((y - f) ** 2)

        assert abs(loss - expected_loss) < 1e-6


class TestLogisticLoss:
    """Test cases for LogisticLoss loss function."""

    def test_compute_derivatives(self):
        """Test gradient and hessian computation."""
        y = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        f = np.array([0.5, -0.5, 1.0, -1.0, 0.0])

        g, h = LogisticLoss.compute_derivatives(y, f)

        # Check shapes
        assert g.shape == y.shape
        assert h.shape == y.shape

        # Check all values are finite
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(h))

    def test_loss(self):
        """Test loss computation."""
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        f = np.array([2.0, -2.0, 1.0, -1.0, 0.0])

        loss = LogisticLoss.loss(y, f)

        # Loss should be non-negative
        assert loss >= 0.0

        # Loss should be finite
        assert np.isfinite(loss)

    def test_loss_binary_labels(self):
        """Test loss with binary labels (0, 1)."""
        y = np.array([1.0, 0.0, 1.0, 0.0])
        f = np.array([1.0, -1.0, 2.0, -2.0])

        loss = LogisticLoss.loss(y, f)

        # Loss should be positive and finite
        assert loss > 0.0
        assert np.isfinite(loss)
