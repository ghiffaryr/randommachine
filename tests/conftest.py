"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    np.random.seed(42)
    yield
    np.random.seed(None)
