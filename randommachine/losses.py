"""
Loss functions for RandomMachine models.
"""

import numpy as np


class MeanSquaredError:
    """Mean squared error loss function."""

    @staticmethod
    def compute_derivatives(y, f):
        """
        Compute first and second derivatives of the loss function.

        Args:
            y (np.ndarray): True labels
            f (np.ndarray): Predictions

        Returns:
            tuple: (gradient, hessian)
        """
        g = 2 * (f - y)
        h = 2.0 * np.ones(y.shape[0])
        return g, h

    @staticmethod
    def loss(y, f):
        """
        Compute the loss value.

        Args:
            y (np.ndarray): True labels
            f (np.ndarray): Predictions

        Returns:
            float: Loss value
        """
        return np.square(np.subtract(y, f)).mean()


class LogisticLoss:
    """Logistic loss function for binary classification."""

    @staticmethod
    def compute_derivatives(y, f):
        """
        Compute first and second derivatives of the loss function.

        Args:
            y (np.ndarray): True labels (0 or 1)
            f (np.ndarray): Raw predictions (logits)

        Returns:
            tuple: (gradient, hessian)
        """
        # Compute sigmoid probability: p = 1 / (1 + exp(-f))
        p = 1.0 / (1.0 + np.exp(-f))
        # Gradient of binary cross-entropy: p - y
        g = p - y
        # Hessian: p * (1 - p)
        h = p * (1.0 - p)
        # Clip hessian to avoid numerical issues
        h = np.maximum(h, 1e-16)
        return g, h

    @staticmethod
    def loss(y, f):
        """
        Compute the loss value.

        Args:
            y (np.ndarray): True labels
            f (np.ndarray): Predictions

        Returns:
            float: Loss value
        """
        losses = []
        for yt, yp in zip(y, f):
            yp_proba = 1.0 / (1.0 + np.exp(-yp))
            first_part = yt * np.log(yp_proba)
            second_part = (1 - yt) * np.log(1 - yp_proba)
            merge = -1 * (first_part + second_part)
            if merge == np.nan:
                merge = -1
            losses.append(merge)
        return np.mean(losses)
