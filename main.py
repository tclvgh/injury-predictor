from typing import Optional
import utils_model.metrics as me
import utils_model.optimization as mo
import utils_model.visualizations as mv
from injury_prediction import InjuryPredictor, PredictionResult
from data_preprocessing import g_x_test, g_y_test, g_x_train, g_y_train
from data.data_pitchers_with_predictions import available_pitchers # List of available pitchers with predictions

# Maximum number of pitchers to display in the selection list
MAX_DISPLAYED_PITCHERS = 15

# Separator line for UI formatting (40 equal signs)
SEPARATOR_LINE = "=" * 40

# Separator line for prediction results (40 hyphens)
PREDICTION_SEPARATOR = "-" * 40

# Initialize the InjuryPredictor with training data
injury_predictor = InjuryPredictor(g_x_train, g_y_train)

def main() -> None:
    """
    Main entry point for the MLB Pitcher Injury Prediction application.

    Initializes the injury predictor, displays the application header,
    allows user to select a pitcher, and displays prediction results.
    """
    print("\nMLB Pitcher Injury Prediction Tool")
    print(SEPARATOR_LINE)
    option = select_option()
    if option == 1:
        print("\nPredict an injury:")
        print(SEPARATOR_LINE)
        selected_pitcher_prediction = select_pitcher()
        prediction_result = injury_predictor.predict_next_injury(selected_pitcher_prediction)
        display_prediction_results(prediction_result)
    elif option == 2:
        print("\nModel Metrics:")
        print(SEPARATOR_LINE)
        evaluator = me.ModelMetrics(injury_predictor.model, g_x_test, g_y_test, "Injury Predictor Model")
        evaluator.print_metrics()
    elif option == 3:
        print("\nModel Optimization:")
        print(SEPARATOR_LINE)
        optimizer = mo.ModelOptimizer(g_x_train, g_y_train, g_x_test, g_y_test)
        optimizer.print_improvements()
        optimizer.print_hyperparameters()
    elif option == 4:
        print("\nModel Visualizations:")
        print(SEPARATOR_LINE)
        choice = select_visualization()
        display_visualization(choice)
    elif option == 5:
        print("Exiting application.")
        return

    # Restart the main menu after displaying results
    main()

def select_option() -> int:
    """
    Prompts the user to select an option from a menu.

    Returns:
        int: The selected option number
    """
    print("\nSelect an option:")
    print("1. Select a pitcher and predict next injury")
    print("2. View model metrics")
    print("3. View model optimization")
    print("4. View model visualizations")
    print("5. Exit")

    try:
        choice = input("Enter option number: ")
        if not choice.strip():
            return 5 # Default to exit if input is empty

        selected_option = int(choice)
        if selected_option in [1, 2, 3, 4, 5]:
            return selected_option

        print("Invalid selection. Exiting.")
        return 5  # Default to exit on invalid input
    except ValueError:
        print("Invalid input. Exiting.")
        return 5  # Default to exit on error

def select_pitcher() -> Optional[str]:
    """
    Displays a list of available pitchers and prompts the user to select one.

    Returns:
        str: The selected pitcher's name, or None if an error occurs

    Note:
        - Only displays up to MAX_DISPLAYED_PITCHERS pitchers
        - Returns the first pitcher if input is empty or invalid
    """
    print("\nPitchers with available predictions:")
    for i, pitcher_name in enumerate(available_pitchers[:MAX_DISPLAYED_PITCHERS], 1):
        print(f"{i}. {pitcher_name}")

    try:
        choice = input("\nEnter pitcher number (or press Enter for first pitcher): ")
        if not choice.strip():
            return str(available_pitchers[0])

        selected_index = int(choice) - 1
        if 0 <= selected_index < len(available_pitchers):
            return str(available_pitchers[selected_index])

        print("Invalid selection. Using first pitcher.")
        return str(available_pitchers[0])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def select_visualization() -> int:
    """
    Prompts the user to select an option from a menu.

    Returns:
        int: The selected option number
    """
    print("\nSelect an option:")
    print("1. Timeline of Features and Injuries")
    print("2. Feature Importance")
    print("3. Linear Regression")
    print("4. RFK Score")
    print("5. Return to main menu")

    visualize_choice = input("Enter visualization option: ")
    visualize_choice = int(visualize_choice)

    return visualize_choice

def display_visualization(choice: int) -> None:
    """
    Displays the selected visualization based on user choice.

    Args:
        choice (int): The choice of visualization to display
    """
    visualization = mv.ModelVisualizations(injury_predictor, g_x_test, g_y_test, "Injury Predictor Model")
    if choice == 1:
        selected_pitcher_visualization = select_pitcher()
        visualization.features_injuries_timeline(selected_pitcher_visualization)
    elif choice == 2:
        visualization.feature_importance_bar_graph()
    elif choice == 3:
        visualization.linear_regression_plot()
    elif choice == 4:
        visualization.rkf_box_plot()
    elif choice == 5:
        print("Returning to main menu.")
        return
    else:
        print("Invalid selection. Exiting.")
        return

def display_prediction_results(result: PredictionResult | str) -> None:
    """
    Displays prediction results in a formatted manner based on the result type.

    Args:
        result (Any): The result to display, which can be one of:
                     - InjuryPredictor: Shows model status
                     - str: Displays the string message
                     - PredictionResult: Shows detailed prediction information
                     - Any other type: Shows an error message
    """
    print(PREDICTION_SEPARATOR)
    print("Prediction Results:")
    if isinstance(result, str):
        print("No pitch data found for for this pitcher since the most recent injury.")
    elif isinstance(result, PredictionResult):
        print(f"Pitcher: {result.pitcher_name}")
        print(f"Predicted days to next injury: {result.predicted_days_to_next_injury}")
        print(f"Predicted injury date: {result.predicted_next_injury_date}")
        print(f"Most recent injury: {result.most_recent_injury}")
        print(f"Most common pitch type: {result.most_common_pitch_type}")
        print(f"Current pitch count: {result.current_pitch_count}")
        print(f"Average effective speed: {result.avg_effective_speed}")
        print(f"Average release spin rate: {result.avg_release_spin_rate}")
    print(PREDICTION_SEPARATOR)

    return

if __name__ == "__main__":
    main()
