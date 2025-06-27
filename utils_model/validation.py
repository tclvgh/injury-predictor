from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
import numpy as np

class ModelValidation:
    """
    Validates the performance of a machine learning model using cross-validation.

    This class provides methods to evaluate the model's performance on training data
    using repeated k-fold cross-validation and calculates the mean absolute error (MAE).
    """

    def __init__(self, model, g_x_test: np.ndarray, g_y_test: np.ndarray):
        """
        Initialize the ModelValidation with a trained model.

        Args:
            model: The trained machine learning model to be validated.
        """
        self.model = model
        self.g_x_test = g_x_test
        self.g_y_test = g_y_test

    def rkf_validate(self):
        """
        Perform repeated k-fold cross-validation on the model.

        This method uses RepeatedKFold to split the training data into multiple folds,
        trains the model on each fold, and evaluates its performance using mean absolute error (MAE).

        Returns:
            scores: Array of MAE scores from cross-validation.
        """
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        scores = cross_val_score(self.model, self.g_x_test, self.g_y_test, cv=rkf, scoring=mae_scorer)
        return -scores

    def print_validation_results(self):
        """
        Print the results of the validation.

        This method performs the validation and prints the mean and standard deviation of the MAE scores.
        """
        scores = self.rkf_validate()
        print("Cross-Validation MAE Scores:", scores)
        print("Mean MAE:", -np.mean(scores))
        print("Standard Deviation:", np.std(scores))
