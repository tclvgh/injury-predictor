import numpy as np
from injury_prediction import InjuryPredictor, PredictionResult
from data_preprocessing import g_original_pitch_data, g_x_train, g_y_train

# Function to get predictions for all pitchers
def pitchers_with_predictions(injury_predictor: InjuryPredictor) -> list:
    """Get predictions for all available pitchers and return as a sorted list"""
    # Get unique pitcher names from the original pitch data
    pitchers = np.sort(g_original_pitch_data['player_name'].unique())

    predictions = []

    # Get predictions for each pitcher
    for pitcher in pitchers:
        result = injury_predictor.predict_next_injury(pitcher)

        # Only include valid prediction results (not error messages)
        if isinstance(result, PredictionResult):
            predictions.append(result.pitcher_name)

    return predictions

def main():
    """Main function to get predictions and write to file"""
    # Initialize the InjuryPredictor with training data
    injury_predictor = InjuryPredictor(g_x_train, g_y_train)

    # Get predictions for all pitchers
    pitchers = pitchers_with_predictions(injury_predictor)

if __name__ == "__main__":
    main()