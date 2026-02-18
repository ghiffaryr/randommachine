"""
Heterogeneous Newton Boosting Machine models.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import clone

from .losses import MeanSquaredError, LogisticLoss


class HNBM:
    """
    A generic Heterogeneous Newton Boosting Machine.

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

    def __init__(
        self,
        loss,
        num_iterations,
        learning_rate,
        base_learners,
        probabilities,
        early_stopping_rounds,
    ):
        self.loss_ = loss
        self.num_iterations_ = num_iterations
        self.learning_rate_ = learning_rate
        self.base_learners_ = base_learners
        self.probabilities_ = probabilities
        self.early_stopping_rounds = early_stopping_rounds
        self.ensemble_ = []

    def fit(self, X, y, X_eval=None, y_eval=None):
        """
        Train the model.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            X_eval (np.ndarray, optional): Evaluation feature matrix
            y_eval (np.ndarray, optional): Evaluation labels
        """
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

                print(
                    "Iteration:",
                    i,
                    "Train loss:",
                    error,
                    "Lowest loss:",
                    lowest_error,
                    "at Iteration:",
                    lowest_error_i,
                    "Eval loss:",
                    eval_error,
                    "Lowest eval loss:",
                    lowest_eval_error,
                    "at Iteration:",
                    lowest_eval_error_i,
                )

                if (
                    eval_error > lowest_eval_error
                    and i % self.early_stopping_rounds == 0
                ):
                    print("Eval loss did not decrease anymore!", "Early stopping...")
                    break
            else:
                print(
                    "Iteration:",
                    i,
                    "Train loss:",
                    error,
                    "Lowest loss:",
                    lowest_error,
                    "at Iteration:",
                    lowest_error_i,
                )

                if error > lowest_error and i % self.early_stopping_rounds == 0:
                    print("Loss did not decrease anymore!", "Early stopping...")
                    break

            base_learner = clone(
                np.random.choice(self.base_learners_, p=self.probabilities_)
            )
            print("Learner chosen:", base_learner)
            base_learner.fit(X, -np.divide(g, h), sample_weight=h)
            z += base_learner.predict(X) * self.learning_rate_
            self.ensemble_.append(base_learner)
            print("\n")

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


class HeteroBoostRegressor(HNBM):
    """
    A heterogeneous boosting regressor using decision trees and kernel ridge regressors.

    Args:
        loss (class): Loss function (default: MeanSquaredError)
        num_iterations (int): Number of boosting iterations
        learning_rate (float): Learning rate
        p_tree (float): Probability of selecting a tree at each iteration
        min_max_depth (int): Minimum maximum depth of trees
        max_max_depth (int): Maximum maximum depth of trees
        alpha (float): L2-regularization penalty in the ridge regression
        gamma (float): RBF-kernel parameter
        early_stopping_rounds (int): Early stopping rounds
        random_state (int): Random state for reproducibility
    """

    def __init__(
        self,
        loss=MeanSquaredError,
        num_iterations=1000,
        learning_rate=0.1,
        p_tree=0.8,
        min_max_depth=4,
        max_max_depth=8,
        alpha=1.0,
        gamma=1.0,
        early_stopping_rounds=100,
        random_state=0,
    ):

        np.random.seed(random_state)

        base_learners = []
        probabilities = []

        # Insert decision tree base learners
        depth_range = range(min_max_depth, 1 + max_max_depth)
        for d in depth_range:
            base_learners.append(
                DecisionTreeRegressor(max_depth=d, random_state=random_state)
            )
            probabilities.append(p_tree / len(depth_range))

        # Insert kernel ridge base learner
        base_learners.append(KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma))
        probabilities.append(1.0 - p_tree)

        super().__init__(
            loss,
            num_iterations,
            learning_rate,
            base_learners,
            probabilities,
            early_stopping_rounds,
        )

    def predict(self, X):
        """
        Predict using the model.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        return super().predict_raw(X)


class HeteroBoostClassifier(HNBM):
    """
    A heterogeneous boosting classifier using decision trees and kernel ridge regressors.

    Args:
        loss (class): Loss function (default: LogisticLoss)
        num_iterations (int): Number of boosting iterations
        learning_rate (float): Learning rate
        p_tree (float): Probability of selecting a tree at each iteration
        min_max_depth (int): Minimum maximum depth of trees
        max_max_depth (int): Maximum maximum depth of trees
        alpha (float): L2-regularization penalty in the ridge regression
        gamma (float): RBF-kernel parameter
        early_stopping_rounds (int): Early stopping rounds
        random_state (int): Random state for reproducibility
    """

    def __init__(
        self,
        loss=LogisticLoss,
        num_iterations=1000,
        learning_rate=0.1,
        p_tree=0.8,
        min_max_depth=4,
        max_max_depth=8,
        alpha=1.0,
        gamma=1.0,
        early_stopping_rounds=100,
        random_state=0,
    ):

        np.random.seed(random_state)

        base_learners = []
        probabilities = []

        # Insert decision tree base learners
        depth_range = range(min_max_depth, 1 + max_max_depth)
        for d in depth_range:
            base_learners.append(
                DecisionTreeRegressor(max_depth=d, random_state=random_state)
            )
            probabilities.append(p_tree / len(depth_range))

        # Insert kernel ridge base learner
        base_learners.append(KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma))
        probabilities.append(1.0 - p_tree)

        super().__init__(
            loss,
            num_iterations,
            learning_rate,
            base_learners,
            probabilities,
            early_stopping_rounds,
        )

    def predict_proba(self, X):
        """
        Predict probability using the model.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Probability predictions
        """
        return 1.0 / (1.0 + np.exp(-super().predict_raw(X)))

    def predict(self, X):
        """
        Predict using the model.

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Class predictions
        """
        return np.where(self.predict_proba(X) > 0.5, 1, 0)
