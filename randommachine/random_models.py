"""
Random machine models: generic base-learner-passing models and CatBoost+LightGBM mixed models.
"""

import os
import random
import numpy as np
import catboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone

from .losses import MeanSquaredError, LogisticLoss


class RM:
    """
    A random Machine that randomly selects between different base learners.
    
    Args:
        loss (class): Loss function
        num_iterations (int): Number of boosting iterations
        learning_rate (float): Learning rate
        base_learners (list): List of base learners
        probabilities (list): List of sampling probabilities
        early_stopping_rounds (int): Early stopping rounds
        
    Attributes:
        ensemble_ (list): Ensemble after training
    """
    
    def __init__(self, loss, num_iterations, learning_rate, base_learners, 
                 probabilities, early_stopping_rounds):
        self.loss_ = loss
        self.num_iterations_ = num_iterations
        self.learning_rate_ = learning_rate
        self.base_learners_ = base_learners
        self.probabilities_ = probabilities
        self.early_stopping_rounds = early_stopping_rounds
        self.ensemble_ = []

    def fit(self, X, y, X_eval=None, y_eval=None, model_directory_path: str = "resources"):
        """
        Train the model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            X_eval (np.ndarray, optional): Evaluation feature matrix
            y_eval (np.ndarray, optional): Evaluation labels
            model_directory_path (str): Directory path for model artifacts
        """
        # Create model directory if it doesn't exist
        os.makedirs(model_directory_path, exist_ok=True)
        
        z = np.zeros(X.shape[0])
        self.ensemble_ = []
        lowest_error = None
        lowest_error_i = 0
        
        if X_eval is not None and y_eval is not None:
            eval_preds = np.zeros(X_eval.shape[0])
            lowest_eval_error = None
            lowest_eval_error_i = 0
            
        for i in range(0, self.num_iterations_):
            try:
                g, h = self.loss_.compute_derivatives(y, z)
                error = self.loss_.loss(y, z)
            except:
                g, h = self.loss_.compute_derivatives(np.array(y)[:, 0], z)
                error = self.loss_.loss(np.array(y)[:, 0], z)
                
            if lowest_error == None or error < lowest_error:
                lowest_error = error
                lowest_error_i = i
                
            if X_eval is not None and y_eval is not None:
                for learner in self.ensemble_:
                    eval_preds += self.learning_rate_ * learner.predict(X_eval)
                try:
                    eval_error = self.loss_.loss(y_eval, eval_preds)
                except:
                    eval_error = self.loss_.loss(np.array(y_eval)[:, 0], eval_preds)
                    
                if lowest_eval_error == None or eval_error < lowest_eval_error:
                    lowest_eval_error = eval_error
                    lowest_eval_error_i = i
                    
                print('Iteration:', i, 'Train loss:', error, 'Lowest loss:', lowest_error, 
                      'at Iteration:', lowest_error_i, 'Eval loss:', eval_error, 
                      'Lowest eval loss:', lowest_eval_error, 'at Iteration:', lowest_eval_error_i)
                      
                if eval_error > lowest_eval_error and i % self.early_stopping_rounds == 0:
                    print('Eval loss did not decrease anymore!', 'Early stopping...')
                    break
            else:
                print('Iteration:', i, 'Train loss:', error, 'Lowest loss:', lowest_error, 
                      'at Iteration:', lowest_error_i)
                      
                if error > lowest_error and i % self.early_stopping_rounds == 0:
                    print('Loss did not decrease anymore!', 'Early stopping...')
                    break
                    
            base_learner = clone(np.random.choice(self.base_learners_, p=self.probabilities_))
            print('Learner chosen:', base_learner)
            
            if isinstance(base_learner, catboost.core.CatBoostRegressor):
                info_path = os.path.join(model_directory_path, 'catboost_info')
                os.makedirs(info_path, exist_ok=True)
                base_learner.set_params(train_dir=info_path)
                
            base_learner.fit(X, -np.divide(g, h), sample_weight=h)
            z += base_learner.predict(X) * self.learning_rate_
            self.ensemble_.append(base_learner)
            print('\n')

    def predict_raw(self, X):
        """
        Predict using the model (raw scores).
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Raw predictions
        """
        preds = np.zeros(X.shape[0])
        for learner in self.ensemble_:
            preds += self.learning_rate_ * learner.predict(X)
        return preds


# ---------------------------------------------------------------------------
# Generic models: base learners are passed directly by the user
# ---------------------------------------------------------------------------

class RandomRegressor(RM):
    """
    A generic random regressor with user-supplied base learners.

    Instead of auto-generating a CatBoost+LightGBM pool, this class accepts
    an explicit list of scikit-learn-compatible regression estimators that form
    the candidate pool from which one learner is sampled at each boosting step.

    Args:
        base_learners (list): List of sklearn-compatible regression estimators
            to sample from. Each is cloned before fitting at every iteration.
        probabilities (list, optional): Sampling probabilities for each
            base learner. Defaults to uniform.
        loss (class): Loss function (default: MeanSquaredError)
        num_iterations (int): Number of boosting iterations (default: 20)
        learning_rate (float): Learning rate (default: 0.5)
        early_stopping_rounds (int): Early stopping rounds (default: 3)
        random_state (int): Random seed (default: 0)

    Example::

        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        from randommachine import RandomRegressor, MeanSquaredError

        learners = [
            LGBMRegressor(max_depth=3, n_estimators=100, verbose=-1),
            LGBMRegressor(max_depth=5, n_estimators=100, verbose=-1),
            CatBoostRegressor(depth=4, iterations=100, verbose=False),
        ]
        model = RandomRegressor(base_learners=learners, num_iterations=20)
        model.fit(X_train, y_train)
    """

    def __init__(self, base_learners, probabilities=None, loss=MeanSquaredError,
                 num_iterations=20, learning_rate=0.5, early_stopping_rounds=3,
                 random_state=0):
        np.random.seed(random_state)
        if probabilities is None:
            probabilities = [1 / len(base_learners)] * len(base_learners)
        super().__init__(loss, num_iterations, learning_rate, base_learners,
                         probabilities, early_stopping_rounds)

    def predict(self, X):
        """
        Predict using the model.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        return self.predict_raw(X)


class RandomClassifier(RM):
    """
    A generic random classifier with user-supplied base learners.

    Accepts an explicit list of scikit-learn-compatible regression estimators
    (continuous outputs are passed through a sigmoid to obtain class
    probabilities, following the Newton-Raphson gradient boosting scheme with
    LogisticLoss).

    Args:
        base_learners (list): List of sklearn-compatible regression estimators.
        probabilities (list, optional): Sampling probabilities. Defaults to uniform.
        loss (class): Loss function (default: LogisticLoss)
        num_iterations (int): Number of boosting iterations (default: 20)
        learning_rate (float): Learning rate (default: 0.5)
        early_stopping_rounds (int): Early stopping rounds (default: 3)
        random_state (int): Random seed (default: 0)

    Example::

        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from randommachine import RandomClassifier, LogisticLoss

        learners = [
            LGBMRegressor(max_depth=3, n_estimators=100, verbose=-1),
            XGBRegressor(max_depth=4, n_estimators=100, verbosity=0),
        ]
        clf = RandomClassifier(base_learners=learners, num_iterations=30)
        clf.fit(X_train, y_train)
    """

    def __init__(self, base_learners, probabilities=None, loss=LogisticLoss,
                 num_iterations=20, learning_rate=0.5, early_stopping_rounds=3,
                 random_state=0):
        np.random.seed(random_state)
        if probabilities is None:
            probabilities = [1 / len(base_learners)] * len(base_learners)
        super().__init__(loss, num_iterations, learning_rate, base_learners,
                         probabilities, early_stopping_rounds)

    def predict_proba(self, X):
        """
        Predict class probability using the model.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Probability predictions (P(y=1))
        """
        return 1.0 / (1.0 + np.exp(-super().predict_raw(X)))

    def predict(self, X):
        """
        Predict class labels using the model.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Class predictions (0 or 1)
        """
        return np.where(self.predict_proba(X) > 0.5, 1, 0)
