"""
RandomMachine - A Python library for random ensemble learning models
"""

from .losses import MeanSquaredError, LogisticLoss
from .lgbm_models import RLGBM, RandomLGBMRegressor, RandomLGBMClassifier
from .catboost_models import RCBM, RandomCatBoostRegressor, RandomCatBoostClassifier
from .hetero_models import HNBM, HeteroBoostRegressor, HeteroBoostClassifier
from .random_models import RM, RandomRegressor, RandomClassifier
from .xgboost_models import RXGBM, RandomXGBRegressor, RandomXGBClassifier

__version__ = "0.1.0"

__all__ = [
    # Loss functions
    "MeanSquaredError",
    "LogisticLoss",
    # LGBM models
    "RLGBM",
    "RandomLGBMRegressor",
    "RandomLGBMClassifier",
    # CatBoost models
    "RCBM",
    "RandomCatBoostRegressor",
    "RandomCatBoostClassifier",
    # Heterogeneous models
    "HNBM",
    "HeteroBoostRegressor",
    "HeteroBoostClassifier",
    # Random models
    "RM",
    "RandomRegressor",
    "RandomClassifier",
    # XGBoost models
    "RXGBM",
    "RandomXGBRegressor",
    "RandomXGBClassifier",
]
