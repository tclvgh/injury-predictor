"""
Injury Prediction Module for Baseball Pitchers.

This module implements the core functionality for predicting the time to a pitcher's next injury
based on their pitching patterns and injury history. It uses an XGBoost regression model
trained on historical data to make predictions.
"""

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
from data_preprocessing import (
    g_features,
    g_original_injury_data,
    g_original_pitch_data,
)

# XGBoost model configuration parameters
# These parameters were selected based on model optimization experiments
MODEL_CONFIG = {
    'objective': 'reg:squarederror',  # Regression task with squared error loss
    'random_state': 42,               # For reproducibility
    'subsample': 1.0,                 # Use all data for each tree
    'n_estimators': 200,              # Number of trees in the ensemble
    'min_child_weight': 5,            # Minimum sum of instance weight in a child
    'max_depth': 3,                   # Maximum depth of trees
    'learning_rate': 0.01,            # Step size shrinkage to prevent overfitting
    'gamma': 0.2,                     # Minimum loss reduction for further partition
    'colsample_bytree': 0.6           # Fraction of features to use for each tree
}

@dataclass
class PredictionResult:
    """
    Data class to store and structure injury prediction results.

    This class encapsulates all relevant information about a pitcher's injury prediction,
    including their recent injury history, predicted time to next injury, and current
    pitching statistics.

    Attributes:
        pitcher_name (str): Name of the pitcher
        most_recent_injury (datetime): Date of the pitcher's most recent injury
        predicted_days_to_next_injury (float): Predicted number of days until next injury
        predicted_next_injury_date (datetime): Calculated date of the predicted next injury
        current_pitch_count (int): Number of pitches thrown since most recent injury
        most_common_pitch_type (str): Most frequently used pitch type since most recent injury
        avg_effective_speed (float): Average effective speed of pitches since most recent injury
        avg_release_spin_rate (float): Average spin rate of pitches since most recent injury
    """
    pitcher_name: str
    most_recent_injury: datetime
    predicted_days_to_next_injury: float
    predicted_next_injury_date: datetime
    current_pitch_count: int
    most_common_pitch_type: str
    avg_effective_speed: float
    avg_release_spin_rate: float

class InjuryPredictor:
    """
    Predicts the time to a pitcher's next injury based on pitching patterns and injury history.

    This class encapsulates the injury prediction model and provides methods to retrieve
    pitcher data, calculate features, and make predictions about future injuries.
    """

    def __init__(self, g_x_train: pd.DataFrame, g_y_train: pd.DataFrame):
        """
        Initialize the InjuryPredictor with a trained XGBoost model.

        Args:
            g_x_train (pd.DataFrame): Training feature data for the model
            g_y_train (pd.DataFrame): Training target data (injury days) for the model

        The model is initialized only if feature data is available.
        """
        # Store training data
        self.g_x_train = g_x_train
        self.g_y_train = g_y_train

        # Initialize the model if features are provided, otherwise set to None
        self.model = self._initialize_model()

    def _initialize_model(self) -> xgb.XGBRegressor:
        """
        Initialize and train the XGBoost regression model.

        Returns:
            xgb.XGBRegressor: Trained XGBoost model for injury prediction
        """
        model = xgb.XGBRegressor(**MODEL_CONFIG)
        model.fit(self.g_x_train, self.g_y_train)
        return model

    @staticmethod
    def _encode_features(pitch_data: pd.DataFrame, pitcher_name: str, pitch_type: str) -> tuple:
        """
        Encode the features for a pitcher and their most common pitch type.

        This method encodes the pitcher's name and the most common pitch type into numerical values
        for model prediction.

        Args:
            pitch_data (pd.DataFrame): DataFrame containing pitch data
            pitcher_name (str): Name of the pitcher
            pitch_type (str): Most common pitch type

        Returns:
            np.array: Encoded features as a numpy array
        """
        # Encode the pitcher's name
        name_encoder = LabelEncoder()
        name_encoder.fit(pitch_data['player_name'].values)
        name_encoded = name_encoder.transform([pitcher_name])[0] if pitcher_name in name_encoder.classes_ else -1

        # Encode the most common pitch type
        pitch_type_encoder = LabelEncoder()
        pitch_type_encoder.fit(g_features['most_common_pitch_type'])
        pitch_type_encoded = pitch_type_encoder.transform([pitch_type])[0] if pitch_type in pitch_type_encoder.classes_ else -1

        return name_encoded, pitch_type_encoded

    @staticmethod
    def _most_recent_injury(pitcher_name: str) -> datetime | str:
        """
        Get the most recent injury date for a specific pitcher.

        Args:
            pitcher_name (str): Name of the pitcher to check

        Returns:
            datetime: Most recent injury date
        """
        pitcher_injuries = g_original_injury_data[
            g_original_injury_data['name'] == pitcher_name
            ].sort_values('injury_date')

        if pitcher_injuries.empty:
            return "No injury data found for this pitcher."

        # Get the last (most recent) injury date in the sorted DataFrame
        return pitcher_injuries.iloc[-1]['injury_date']

    @staticmethod
    def _most_recent_pitch_data(pitcher_name: str, injury_date: datetime) -> pd.DataFrame | str:
        """
        Return all pitch data for a given pitcher after a specific injury date.

        Args:
            pitcher_name (str): Name of the pitcher.
            injury_date (pd.Timestamp): The injury date to filter from.

        Returns:
            pd.DataFrame: Subset of g_original_pitch_data with matching pitcher and game_date > injury_date.
        """
        # Get the pitch data for the specified pitcher after the injury date
        filtered_data = g_original_pitch_data.loc[
            (g_original_pitch_data['player_name'] == pitcher_name) &
            (g_original_pitch_data['game_date'] > injury_date)
            ]

        if filtered_data.empty:
            return "No pitch data found for this pitcher since the specified injury date."

        # Check for required columns in the pitch data
        if 'pitch_type' not in filtered_data or filtered_data['pitch_type'].empty:
            return "No pitch type data available for this pitcher."
        if 'effective_speed' not in filtered_data or filtered_data['effective_speed'].empty:
            return "No effective speed data available for this pitcher."
        if 'release_spin_rate' not in filtered_data or filtered_data['release_spin_rate'].empty:
            return "No release spin rate data available for this pitcher."

        return filtered_data

    @staticmethod
    def _most_recent_statistics(pitch_data: pd.DataFrame) -> tuple:
        """
        Calculate the most common pitch type, number of pitches, and average statistics.

        Args:
            pitch_data (pd.DataFrame): DataFrame containing pitch data

        Returns:
            tuple: Most common pitch type, number of pitches, average effective speed, average release spin rate
        """
        # Calculate the most common pitch type
        most_common_pitch_type = pitch_data['pitch_type'].mode()[0] if not pitch_data['pitch_type'].empty else 'Unknown'

        # Calculate the number of pitches
        num_pitches = len(pitch_data)

        # Calculate average effective speed and release spin rate
        avg_effective_speed = pitch_data['effective_speed'].mean() if 'effective_speed' in pitch_data else 0.0
        avg_release_spin_rate = pitch_data['release_spin_rate'].mean() if 'release_spin_rate' in pitch_data else 0.0

        return most_common_pitch_type, num_pitches, avg_effective_speed, avg_release_spin_rate

    def _most_recent_features(self, pitch_data: pd.DataFrame, pitcher_name: str) -> np.array:
        """
        Prepare features for prediction from pitch data.

        This method encodes categorical features, calculates aggregate statistics,
        and returns a DataFrame ready for model prediction.

        Args:
            pitch_data (pd.DataFrame): DataFrame containing pitch data
            pitcher_name (string): Name of the pitcher for encoding

        Returns:
            pd.DataFrame: DataFrame with encoded and aggregated features
        """
        # Calculate aggregate statistics from the pitch data
        most_common_pitch_type, num_pitches, avg_effective_speed, avg_release_spin_rate = (
            self._most_recent_statistics(pitch_data))

        # Check for valid encoded values and encode
        encoded = self._encode_features(pitch_data, pitcher_name, most_common_pitch_type)
        if -1 in encoded:
            return "Invalid pitcher name or pitch type for encoding."
        name_encoded, pitch_type_encoded = encoded

        # Prepare features as a numpy array
        return np.array([
            [
                name_encoded,
                pitch_type_encoded,
                num_pitches,
                avg_effective_speed,
                avg_release_spin_rate
            ]
        ])

    def predict_next_injury(self, pitcher_name: str) -> PredictionResult | str:
        """
        Predict when a pitcher is likely to experience their next injury.

        This is the main prediction method that:
        1. Retrieves the pitcher's most recent injury
        2. Gets their pitching data since that injury
        3. Calculates features from the pitch data
        4. Makes a prediction using the trained model
        5. Returns a result with prediction details

        Args:
            pitcher_name (str): Name of the pitcher to make predictions for

        Returns:
            PredictionResult: Either an object containing the prediction details, or an error message string
                  if prediction cannot be made
        """
        # Retrieve the most recent injury date for the pitcher
        pitcher_most_recent_injury = self._most_recent_injury(pitcher_name)

        # Check if the most recent injury date is a string (error message)
        if isinstance(pitcher_most_recent_injury, str):
            return pitcher_most_recent_injury

        # Retrieve the features for the pitcher since the most recent injury
        pitch_data = self._most_recent_pitch_data(pitcher_name, pitcher_most_recent_injury)

        # Check if the pitch data is a string (error message)
        if isinstance(pitch_data, str):
            return pitch_data

        # Get the most common pitch type and number of pitches
        pitch_type_counts = pitch_data['pitch_type'].value_counts()
        most_common_pitch_type = pitch_type_counts.idxmax() if not pitch_type_counts.empty else 'Unknown'

        # Retrieve the most common pitch type, number of pitches, and average statistics, then encode (name, pitch_type)
        features = self._most_recent_features(pitch_data, pitcher_name)

        if isinstance(features, str):
            return features

        # Make prediction and convert to integer days
        days_to_next_injury = self.model.predict(features)[0]
        days_to_next_injury = int(days_to_next_injury)
        predicted_injury_date = pitcher_most_recent_injury + pd.DateOffset(days=days_to_next_injury)

        # Create a prediction result
        return PredictionResult(
            pitcher_name=pitcher_name,
            predicted_days_to_next_injury=days_to_next_injury,
            predicted_next_injury_date=predicted_injury_date,
            most_recent_injury=pitcher_most_recent_injury,
            most_common_pitch_type=most_common_pitch_type,
            current_pitch_count = int(features[0][2]),
            avg_effective_speed = float(features[0][3]),
            avg_release_spin_rate = float(features[0][4])
        )
