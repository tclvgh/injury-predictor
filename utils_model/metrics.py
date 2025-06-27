"""
Model Evaluation Metrics Module.

This module provides functionality for evaluating machine learning model performance
using standard regression metrics such as MSE, MAE, and R-squared.
"""

from typing import Dict, Any, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class ModelMetrics:
    """
    Evaluates machine learning model performance using standard regression metrics.

    This class provides methods to calculate performance metrics for regression models,
    evaluate models on test data, and display formatted results.
    """

    def __init__(self, model: Any, g_x_test: np.ndarray, g_y_test: np.ndarray, model_name: str = "Model"):
        """
        Initialize a ModelEvaluator with a model name.

        Args:
            model_name (str): Name of the model being evaluated. Defaults to "Model".
        """
        self.model_name = model_name
        self.model = model

        # Ensure the model is trained before evaluation
        self.y_pred = self.model.predict(g_x_test)
        self.g_y_test = g_y_test
        self.g_x_test = g_x_test

    def evaluate(self) -> Dict[str, Union[str, float, np.ndarray]]:
        """
        Evaluate a model on test data and calculate performance metrics.

        Uses the model to make predictions on input data, then calculates
        performance metrics by comparing predictions to ground truth values.

        Returns:
            Dict[str, Union[str, float, np.ndarray]]: Dictionary containing model name,
                                                     performance metrics, and predictions
        """

        return {
            'model_name': self.model_name,
            'mse': mean_squared_error(self.g_y_test, self.y_pred),
            'mae': mean_absolute_error(self.g_y_test, self.y_pred),
            'r2': r2_score(self.g_y_test, self.y_pred),
            'y_pred': self.y_pred
        }

    def print_metrics(self) -> None:
        """
        Display formatted model evaluation metrics.

        Prints the model name and performance metrics, including a linear regression graph in a human-readable format.

        """
        print(f"\n{self.model_name} Performance Metrics:")
        print(f"Mean Squared Error: {self.evaluate()['mse']:.2f}")
        print(f"Mean Absolute Error: {self.evaluate()['mae']:.2f}")
        print(f"R2 Score: {self.evaluate()['r2']:.2f}")

