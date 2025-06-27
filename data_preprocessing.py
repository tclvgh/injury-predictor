"""
Data Preprocessing Module for Injury Prediction System.

This module handles loading, cleaning, and preprocessing of baseball player injury and pitch data.
It provides functionality for feature engineering and data preparation for the machine learning model.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import TypeAlias, Tuple, List
from sqlalchemy import create_engine
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import glob
import os

# Type alias for functions returning both processed and original data
DataFramePair: TypeAlias = Tuple[pd.DataFrame, pd.DataFrame]

# Constants for data splitting and reproducibility
TRAIN_TEST_SPLIT_RATIO = 0.2  # 80% training, 20% testing
RANDOM_SEED = 42  # Fixed random seed for reproducibility

# Load environment database url and credentials from .env file
load_dotenv()
db_url = os.getenv("DATABASE_URL")

class DataProcessor:
    """
    Processes raw baseball data for machine learning model input.

    This class handles loading, cleaning, encoding categorical variables, and scaling numeric features
    for both injury and pitch data. It maintains the original data for reference while creating
    processed versions suitable for model training.

    Args:
    data_source (str): Source of the data, either a PostgreSQL connection string or a directory path for CSV files.
                       If using PostgreSQL, it should be in the format 'postgresql://user:password@host:port/dbname'.
                       If using CSV files, it should be the directory containing the CSV files.
    """
    # Columns to be encoded as categorical variables
    INJURY_CATEGORICAL_COLS = ['name']
    PITCH_CATEGORICAL_COLS = ['pitch_type', 'player_name', 'pitch_name']

    # Columns to be scaled as numeric features
    PITCH_NUMERIC_COLS = [
        'release_speed', 'effective_speed', 'release_spin_rate',
        'pitch_number', 'spin_axis', 'age_pit', 'pitcher_days_since_prev_game'
    ]

    def __init__(self, data_source: str):
        self.original_pitch_data = None
        self.encoded_pitch_data = None
        self.original_injury_data = None
        self.encoded_injury_data = None
        self.features_data = None
        self.encoded_features_data = None

        # Load raw data from the specified source (PostgreSQL or CSV)
        if data_source.startswith('postgresql:'):
            self.raw_injury_data = self._load_injuries_postgresql(data_source)
            self.raw_pitch_data = self._load_pitches_postgresql(data_source)
        else:
            self.raw_injury_data = self._load_injuries_csv(data_source)
            self.raw_pitch_data = self._load_pitches_csv(data_source)

    @staticmethod
    def _load_injuries_postgresql(data_source: str) -> pd.DataFrame:
        """
        Load injury data from a Heroku Postgres database using the DATABASE_URL in .env.

        Returns:
            pd.DataFrame: DataFrame containing injury data or empty DataFrame if file doesn't exist.
        """
        engine = create_engine(data_source)
        if not db_url:
            return pd.DataFrame()
        try:
            df = pd.read_sql("SELECT * FROM injuries", engine)
        except Exception:
            df = pd.DataFrame()
        return df

    @staticmethod
    def _load_pitches_postgresql(data_source: str) -> pd.DataFrame:
        """
        Load pitch log data from a Heroku Postgres database using the DATABASE_URL in .env.

        Returns:
            pd.DataFrame: DataFrame containing pitch data or empty DataFrame if file doesn't exist.
        """
        engine = create_engine(data_source)
        if not db_url:
            return pd.DataFrame()
        try:
            df = pd.read_sql("SELECT * FROM pitches", engine)
        except Exception:
            df = pd.DataFrame()
        return df

    @staticmethod
    def _load_injuries_csv(data_source: str = 'data') -> pd.DataFrame:
        """
        Load injury data from CSV file.

        Args:
            data_source (str): Path to the injury data CSV file. Defaults to 'data'.  Uses '2020to2024injuries.csv' in that directory.

        Returns:
            pd.DataFrame: DataFrame containing injury data or empty DataFrame if file doesn't exist.
        """
        # Always resolve path relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_file_csv = os.path.join(base_dir, data_source, '2020to2024injuries.csv')
        if not os.path.exists(data_file_csv):
            return pd.DataFrame()
        return pd.read_csv(data_file_csv)

    @staticmethod
    def _load_pitches_csv(data_source: str = 'data') -> pd.DataFrame:
        """
        Load and combine pitch log data from multiple CSV files.

        Searches for files matching '*pitchlog.csv' pattern in the specified directory
        and concatenates them into a single DataFrame.

        Args:
            data_source (str): Directory containing pitch log CSV files. Defaults to 'data'.

        Returns:
            pd.DataFrame: Combined DataFrame with all pitch log data or empty DataFrame if no files found.
        """
        # Always resolve path relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pattern = os.path.join(base_dir, data_source, '*pitchlog.csv')
        files = glob.glob(pattern)
        if not files:
            return pd.DataFrame()
        df_list = [pd.read_csv(f) for f in files]
        return pd.concat(df_list, ignore_index=True)

    @staticmethod
    def _encode_categorical_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Convert categorical columns to numeric using label encoding.

        Args:
            data (pd.DataFrame): Input DataFrame with categorical columns
            columns (List[str]): List of column names to encode

        Returns:
            pd.DataFrame: DataFrame with encoded categorical columns
        """
        processed_data = data.copy()
        for col in columns:
            le = LabelEncoder()
            processed_data[col] = le.fit_transform(processed_data[col])
        return processed_data

    @staticmethod
    def _scale_numeric_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Scale numeric columns using StandardScaler.

        Args:
            data (pd.DataFrame): Input DataFrame with numeric columns
            columns (List[str]): List of column names to scale

        Returns:
            pd.DataFrame: DataFrame with scaled numeric columns
        """
        processed_data = data.copy()
        scaler = StandardScaler()
        processed_data[columns] = scaler.fit_transform(processed_data[columns])
        return processed_data

    @staticmethod
    def _clean_base_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing null values and duplicates.

        Args:
            data (pd.DataFrame): Input DataFrame to clean

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        return data.dropna().drop_duplicates()

    @staticmethod
    def _encode_features(features_data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features and scale numeric features for model training.

        This function encodes categorical variables using LabelEncoder and scales numeric features
        using StandardScaler. It prepares the DataFrame for machine learning model input.

        Args:
            features_data (pd.DataFrame): DataFrame containing engineered pitcher features

        Returns:
            pd.DataFrame: DataFrame with encoded and scaled features ready for model training
        """
        # Encode categorical features
        features_data['name_encoded'] = LabelEncoder().fit_transform(features_data['name'].values)
        features_data['most_common_pitch_type'] = features_data['pitch_type'].fillna('Unknown')
        features_data['pitch_type_encoded'] = LabelEncoder().fit_transform(
            features_data['most_common_pitch_type'].values)

        return features_data

    def process_injury_data(self) -> DataFramePair:
        """
        Process injury data for model input.

        Loads raw injury data, converts dates to datetime format, cleans the data,
        and encodes categorical variables.

        Returns:
            DataFramePair: Tuple containing (processed_data, original_data)
        """
        self.raw_injury_data['injury_date'] = pd.to_datetime(self.raw_injury_data['injury_date'])
        self.original_injury_data = self._clean_base_data(self.raw_injury_data)
        self.encoded_injury_data = self._encode_categorical_columns(
            self.original_injury_data,
            self.INJURY_CATEGORICAL_COLS
        )

        return self.encoded_injury_data, self.original_injury_data

    def process_pitch_data(self) -> DataFramePair:
        """
        Process pitch data for model input.

        Loads raw pitch data, converts dates to datetime format, cleans the data,
        encodes categorical variables, and scales numeric features.

        Returns:
            DataFramePair: Tuple containing (processed_data, original_data)
        """
        self.raw_pitch_data['game_date'] = pd.to_datetime(self.raw_pitch_data['game_date'])
        self.original_pitch_data = self._clean_base_data(self.raw_pitch_data)
        self.encoded_pitch_data = self._encode_categorical_columns(
            self.original_pitch_data,
            self.PITCH_CATEGORICAL_COLS
        )
        self.encoded_pitch_data = self._scale_numeric_columns(
            self.encoded_pitch_data,
            self.PITCH_NUMERIC_COLS
        )
        return self.encoded_pitch_data, self.original_pitch_data

    def create_pitcher_features(self, pitcher_name: str) -> pd.DataFrame:
        """
        Create features for a specific pitcher based on their injury and pitch history.

        This function analyzes the relationship between injuries and pitching patterns
        for a specific pitcher, creating features that capture the time between injuries
        and the pitching characteristics during those periods.

        Args:
            pitcher_name (str): Name of the pitcher to analyze

        Returns:
            pd.DataFrame: DataFrame with engineered features for the pitcher,
                                   or None if insufficient data is available
        """
        # Filter and sort injury data for the specified pitcher
        pitcher_injuries = self.original_injury_data.loc[
            self.original_injury_data['name'] == pitcher_name
        ].sort_values('injury_date').reset_index(drop=True)

        # Filter and sort pitch data for the specified pitcher
        pitcher_pitches = self.original_pitch_data.loc[
            self.original_pitch_data['player_name'] == pitcher_name
        ].sort_values('game_date').reset_index(drop=True)

        # Need at least 2 injuries to calculate time between them
        if len(pitcher_injuries) < 2 or pitcher_pitches.empty:
            return pd.DataFrame()

        features_list = []
        injury_dates = pitcher_injuries['injury_date'].values

        # Analyze each period between consecutive injuries
        for i in range(len(injury_dates) - 1):
            start_date = injury_dates[i]
            end_date = injury_dates[i + 1]
            # Find pitches thrown between these two injury dates
            mask = (pitcher_pitches['game_date'] > start_date) & (pitcher_pitches['game_date'] <= end_date)
            pitches_between = pitcher_pitches.loc[mask]

            if pitches_between.empty:
                continue

            # Calculate the most common pitch type during this period
            most_common_pitch_type = pitches_between['pitch_type'].mode()

            # Create a feature record for this injury period
            features_list.append({
                'name': pitcher_name,
                'current_injury_date': start_date,
                'next_injury_date': end_date,
                'days_to_next_injury': (end_date - start_date) / np.timedelta64(1, 'D'),
                'num_pitches': len(pitches_between),
                'pitch_type': most_common_pitch_type.iloc[0] if not most_common_pitch_type.empty else None,
                'avg_effective_speed': pitches_between['effective_speed'].mean(),
                'avg_release_spin_rate': pitches_between['release_spin_rate'].mean(),
            })

        return pd.DataFrame(features_list)

    def create_combined_features(self) -> pd.DataFrame:
        """
        Create features for all pitchers in the injury data.

        This function iterates through all unique pitchers in the injury data,
        creating features based on their injury and pitch history.

        Returns:
            pd.DataFrame: DataFrame containing engineered features for all pitchers
        """
        all_pitchers = self.original_injury_data['name'].unique()
        all_features = []

        for pitcher in all_pitchers:
            features = self.create_pitcher_features(pitcher)
            if features is not None:
                all_features.append(features)

        return pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()

    def create_model_datasets(self, features_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare datasets for model training and testing.

        This function splits the engineered pitcher features into training and testing sets.

        Args:
            features_data (pd.DataFrame): DataFrame containing engineered pitcher features

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple containing:
                - X_train: Feature matrix for training
                - y_train: Target values for training
                - X_test: Feature matrix for testing
                - y_test: Target values for testing
        """

        features_data_encoded = self._encode_features(features_data)

        # Split data into training (80%) and testing (20%) sets
        train_size = int(0.8 * len(features_data_encoded))
        train_df = features_data_encoded.iloc[:train_size]
        test_df = features_data_encoded.iloc[train_size:]

        # Define feature columns for the model
        feature_cols = ['name_encoded', 'pitch_type_encoded', 'num_pitches', 'avg_effective_speed',
                        'avg_release_spin_rate']

        # Return features (X) and target (y) for both training and testing sets
        return (
            train_df[feature_cols],
            train_df['days_to_next_injury'],
            test_df[feature_cols],
            test_df['days_to_next_injury']
        )

# ============================================================================
# Global Data Processing
# ============================================================================
# The code below is executed when the module is imported, creating global variables
# that can be used by other modules for model training and prediction.

# Initialize the data processor
# data_processor = DataProcessor(db_url) # Use database URL from .env file
data_processor = DataProcessor(db_url)  # Use 'data' directory for CSV files

# Process injury and pitch data, keeping both processed and original versions
g_injury_data_encoded, g_original_injury_data = data_processor.process_injury_data()
g_pitch_data_encoded, g_original_pitch_data = data_processor.process_pitch_data()

# Split processed data into training and testing sets
g_injury_train, g_injury_test = train_test_split(
    g_injury_data_encoded, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED
)
g_pitch_train, g_pitch_test = train_test_split(
    g_pitch_data_encoded, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_SEED
)

# Create features for all pitchers based on their injury and pitch history
g_features = data_processor.create_combined_features()

# Prepare the final feature sets for model training and testing
g_x_train, g_y_train, g_x_test, g_y_test = data_processor.create_model_datasets(g_features)
