"""
Model Optimization Module for Injury Prediction System.

This module provides functionality for optimizing machine learning models through
hyperparameter tuning. It compares baseline models with optimized versions and
calculates performance improvements.
"""

import xgboost as xgb
import numpy as np
from typing import Dict
from sklearn.model_selection import RandomizedSearchCV
from utils_model.metrics import ModelMetrics

# Importing global variables for training and testing data, if needed.
# from data_preprocessing import (g_x_train, g_y_train, g_x_test, g_y_test)

class ModelOptimizer:
    """
    Optimizes machine learning models through hyperparameter tuning.

    This class provides methods to train baseline models, optimize hyperparameters
    using randomized search, and compare performance between baseline and optimized models.
    It focuses on XGBoost regression models for injury prediction.
    """

    # Fixed random state for reproducibility
    RANDOM_STATE = 42

    # Hyperparameter search space for RandomizedSearchCV
    PARAM_GRID = {
        'n_estimators': [50, 100, 200, 300],       # Number of trees in the ensemble
        'learning_rate': [0.01, 0.05, 0.1, 0.2],   # Step size shrinkage
        'max_depth': [3, 5, 7, 9],                 # Maximum depth of trees
        'min_child_weight': [1, 3, 5],             # Minimum sum of instance weight in a child
        'subsample': [0.6, 0.8, 1.0],              # Fraction of samples used for fitting trees
        'colsample_bytree': [0.6, 0.8, 1.0],       # Fraction of features used for each tree
        'gamma': [0, 0.1, 0.2]                     # Minimum loss reduction for partition
    }

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        """
        Initialize the ModelOptimizer with training and testing data.

        Args:
            x_train (np.ndarray): Feature matrix for training
            y_train (np.ndarray): Target values for training
            x_test (np.ndarray): Feature matrix for testing
            y_test (np.ndarray): Target values for testing
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # Train baseline and optimized models
        self.baseline_model = self._train_baseline_model()
        self.optimized_model = self._optimize_model()
        # Evaluate both models
        self.baseline_metrics = ModelMetrics(self.baseline_model, self.x_test, self.y_test, "Baseline Model").evaluate()
        self.optimized_metrics = ModelMetrics(self.optimized_model, self.x_test, self.y_test, "Optimized Model").evaluate()

    def _train_baseline_model(self) -> xgb.XGBRegressor:
        """
        Train a baseline XGBoost regression model with default hyperparameters.

        This method creates and trains a model with standard hyperparameters to serve
        as a baseline for comparison with the optimized model.

        Returns:
            xgb.XGBRegressor: Trained baseline model
        """
        # Create model with standard hyperparameters
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=self.RANDOM_STATE
        )
        # Train the model
        base_model.fit(self.x_train, self.y_train)

        return base_model

    def _optimize_model(self) -> xgb.XGBRegressor:
        """
        Optimize model hyperparameters using RandomizedSearchCV.

        This method performs a randomized search over the hyperparameter space defined
        in PARAM_GRID to find the best combination of parameters for the XGBoost model.

        Returns:
            xgb.XGBRegressor: Trained model with optimized hyperparameters
        """
        # Set up randomized search with cross-validation
        random_search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(objective='reg:squarederror',
                                       random_state=self.RANDOM_STATE),
            param_distributions=self.PARAM_GRID,
            n_iter=25,                           # Number of parameter settings sampled
            scoring='neg_mean_squared_error',    # Optimization metric
            cv=5,                                # 5-fold cross-validation
            verbose=0,                           # No output during fitting
            random_state=self.RANDOM_STATE,
            n_jobs=-1                            # Use all available processors
        )
        # Perform the search
        random_search.fit(self.x_train, self.y_train)

        # Create a new model with the best parameters found
        opt_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.RANDOM_STATE,
            **random_search.best_params_
        )
        # Train the optimized model
        opt_model.fit(self.x_train, self.y_train)

        return opt_model

    def get_improvements(self) -> Dict[str, float]:
        """
        Get the improvements calculated by the optimizer.

        Returns:
            Dict[str, float]: Dictionary of metric improvements
        """
        return {
            'mse': self.optimized_metrics['mse'] - self.baseline_metrics['mse'] / abs(self.baseline_metrics['mse']) * 100,
            'mae': self.optimized_metrics['mae'] - self.baseline_metrics['mae'] / abs(self.baseline_metrics['mse']) * 100,
            'r2': self.optimized_metrics['r2'] - self.baseline_metrics['r2'] / abs(self.baseline_metrics['r2']) * 100
        }

    def print_improvements(self) -> None:
        """
        Display the performance improvements between baseline and optimized models.

        This method prints the percentage improvements in MSE, MAE, and R²
        between the baseline and optimized models.
        """
        print("Performance Improvements:")
        for metric, improvement in self.get_improvements().items():
            print(f"{metric.upper()}: {improvement:.2f}%")

    def print_hyperparameters(self):
        """
        Print the hyperparameters of both baseline and optimized models.

        This function helps visualize the differences in hyperparameters between
        the default model and the optimized model.
        """
        baseline_params = self.baseline_model.get_params()
        optimized_params = self.optimized_model.get_params()
        diff = {k: (baseline_params[k], optimized_params[k]) for k in baseline_params if baseline_params[k] != optimized_params.get(k) and k != 'missing'}
        print("\nHyperparameter Differences")
        print("The first value is the baseline value and the second is the optimized value:")
        for key, value in diff.items():
            print(f"{key}: {value}")

# ============================================================================
# Global Model Optimization
# ============================================================================
# The code below initializes and runs the model optimization process when the module
# is imported. It creates global variables that can be used by other modules.

# Initialize global variables for training and testing data
# optimizer = ModelOptimizer(g_x_train, g_y_train, g_x_test, g_y_test)
# optimizer.print_improvements()
# optimizer.print_hyperparameters()