from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import utils_model.metrics as me
import utils_model.validation as mva
from injury_prediction import InjuryPredictor, PredictionResult
from data_preprocessing import g_original_pitch_data, g_original_injury_data

class ModelVisualizations:
    """
    Provides methods for visualizing model performance and predictions.

    This class includes methods to create plots that compare actual vs predicted values,
    visualize model performance metrics, and display linear regression trends.
    """

    def __init__(self, injury_predictor: InjuryPredictor, g_x_test: np.ndarray, g_y_test: np.ndarray, model_name: str = "Model"):
        # Get the model from the injury predictor
        self.model_name = model_name
        self.model = injury_predictor.model

        # Ensure the model is trained before evaluation
        self.y_pred = self.model.predict(g_x_test)
        self.g_y_test = g_y_test
        self.g_x_test = g_x_test

        # Store the injury predictor for later use in visualizations
        self.injury_predictor = injury_predictor

        # Initialize the evaluator with the model and test data
        self.evaluator = me.ModelMetrics(self.model, self.g_x_test, self.g_y_test, self.model_name)

        # Initialize the validation
        self.validation = mva.ModelValidation(self.model, self.g_x_test, self.g_y_test)

    def linear_regression_plot(self) -> None:
        """
        Create a linear regression plot comparing actual vs predicted values.
        """

        plt.figure(figsize=(10, 8))

        # y_pred: predicted values from the model on the test set
        y_pred = self.evaluator.evaluate()['y_pred']

        # metrics: evaluation metrics dictionary
        metrics = self.evaluator.evaluate()

        title = "Actual vs. Predicted Values"
        color = 'red'

        # Create scatter plot of actual vs predicted values
        plt.scatter(self.g_y_test, y_pred, alpha=0.6, color=color, label='Data Points')

        # Fit a linear regression line to show the trend
        g_y_test_array = np.array(self.g_y_test)
        lr_model = LinearRegression()
        lr_model.fit(g_y_test_array.reshape(-1, 1), y_pred)

        # Generate points for the regression line
        x_range = np.linspace(min(self.g_y_test), max(self.g_y_test), 100)
        y_range = lr_model.predict(x_range.reshape(-1, 1))

        # Plot the regression line
        plt.plot(x_range, y_range, color='green', linewidth=2, label='Linear Regression Line')

        # Add the perfect prediction line (y=x) for reference
        plt.plot([min(self.g_y_test), max(self.g_y_test)], [min(self.g_y_test), max(self.g_y_test)],
                 color='black', linestyle='--', label='Perfect Prediction (y=x)')

        # Add metrics to the plot
        plt.annotate(f"R² = {metrics['r2']:.4f}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.annotate(f"MSE = {metrics['mse']:.4f}",
                     xy=(0.05, 0.90), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.annotate(f"MAE = {metrics['mae']:.4f}",
                     xy=(0.05, 0.85), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Add labels and title
        plt.xlabel('Actual Days to Next Injury')
        plt.ylabel('Predicted Days to Next Injury')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    def feature_importance_bar_graph(self) -> None:
        """
        Create a bar plot showing feature importance from the model.
        """
        plt.figure(figsize=(10, 6))

        # Get feature importance from the model, remove 'name' if it exists, then sort by importance
        feature_importance = self.model.get_booster().get_score(importance_type='weight')
        feature_importance = {k: v for k, v in feature_importance.items() if k != 'name'}
        feature_importance = {k: v for k, v in feature_importance.items() if k != 'name_encoded'}
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        # Sort features by importance
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())

        # Create a mapping for more readable feature names
        feature_name_map = {
            'pitch_type_encoded': 'Pitch Type',
            'num_pitches': 'Number of Pitches',
            'avg_effective_speed': 'Average Effective Speed',
            'avg_release_spin_rate': 'Average Release Spin Rate'
        }

        # Replace feature names with more readable versions
        readable_features = [feature_name_map.get(f, f) for f in features]

        # Normalize importance values for percentage representation
        total_importance = sum(importance)
        normalized_importance = [100 * imp / total_importance for imp in importance]

        # Vertical bar chart
        bars = plt.bar(readable_features, normalized_importance, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Importance (%)')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom')

        plt.title('Feature Importance for Injury Prediction')
        plt.tight_layout()
        plt.show()
        return None

    def feature_importance_pie_chart(self) -> None:
        """
        Create a pie chart showing feature importance from the model.
        """
        # Get feature importance from the model, remove 'name' if it exists, then sort by importance
        feature_importance = self.model.get_booster().get_score(importance_type='weight')
        feature_importance = {k: v for k, v in feature_importance.items() if k != 'name'}
        feature_importance = {k: v for k, v in feature_importance.items() if k != 'name_encoded'}
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        features = list(feature_importance.keys())
        importance = list(feature_importance.values())

        # Create a mapping for more readable feature names
        feature_name_map = {
            'pitch_type_encoded': 'Pitch Type',
            'num_pitches': 'Number of Pitches',
            'avg_effective_speed': 'Average Effective Speed',
            'avg_release_spin_rate': 'Average Release Spin Rate'
        }

        # Replace feature names with more readable versions
        readable_features = [feature_name_map.get(f, f) for f in features]

        # Normalize importance values for percentage representation
        total_importance = sum(importance)
        normalized_importance = [100 * imp / total_importance for imp in importance]

        # Create pie chart
        fig = go.Figure(
            data=[go.Pie(
                labels=readable_features,
                values=normalized_importance,
                textinfo='label+percent',
                insidetextorientation='radial'
            )]
        )

        fig.update_layout(title_text='Feature Importance for Injury Prediction', width=600, height=600)
        fig.show()
        return None

    def features_injuries_timeline(self, pitcher_name: str) -> None | str:
        """
        Create a timeline plot showing features and injuries for a specific pitcher.

        Args:
            pitcher_name (str): Name of the pitcher to visualize.
        """
        # Add predicted next injury if available
        prediction_result = self.injury_predictor.predict_next_injury(pitcher_name)

        # Get pitch data for the specified pitcher
        pitcher_pitches = g_original_pitch_data[
            g_original_pitch_data['player_name'] == pitcher_name
            ].sort_values('game_date')

        # Get pitcher's injury data
        pitcher_injuries = g_original_injury_data[
            g_original_injury_data['name'] == pitcher_name
            ].sort_values('injury_date')

        # Group pitch data by date to calculate daily averages (if data exists)
        if len(pitcher_pitches) > 0:
            daily_data = pitcher_pitches.groupby('game_date').agg({
                'effective_speed': 'mean',
                'release_spin_rate': 'mean',
                'player_name': 'count'  # Count of pitches as workload (using player_name which is guaranteed to exist)
            }).reset_index()
        else:
            daily_data = pd.DataFrame()

        if len(pitcher_pitches) == 0:
            return f"No pitch data found since the most recent injury."

        if len(pitcher_injuries) == 0:
            return f"No injury data found since the most recent injury."

        # Create figure with subplots (with shared x-axes for better performance and alignment)
        fig = make_subplots(rows=3, cols=1,
                           vertical_spacing=0.1,
                           row_heights=[0.33, 0.33, 0.33])

        # Set the title for the entire figure
        fig.update_layout(title_text=f"Timeline of Features and Injuries for {pitcher_name}",
                          title_x=0.5)

        # Add speed data
        fig.add_trace(
            go.Scatter(x=daily_data['game_date'], y=daily_data['effective_speed'],
                      mode='lines+markers', name='Effective Speed',
                      line=dict(color='blue')),
            row=1, col=1
        )

        # Add spin rate data
        fig.add_trace(
            go.Scatter(x=daily_data['game_date'], y=daily_data['release_spin_rate'],
                      mode='lines+markers', name='Spin Rate',
                      line=dict(color='green')),
            row=2, col=1
        )

        # Add workload data with text labels for pitch counts
        fig.add_trace(
            go.Bar(x=daily_data['game_date'], y=daily_data['player_name'],
                  name='Pitch Count', marker_color='orange',
                  text=daily_data['player_name'],
                  textposition='outside'),
            row=3, col=1
        )

        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0,bargroupgap=0)

        # Add actual injuries as vertical lines
        for _, injury in pitcher_injuries.iterrows():
            injury_date = injury['injury_date'].timestamp() * 1000

            # Add vertical line for actual injury in all subplots
            fig.add_vline(
                x=injury_date,
                name='Injury',
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text="Injury",
                annotation_position="top right"
            )

        if isinstance(prediction_result, PredictionResult) and hasattr(prediction_result, 'predicted_next_injury_date'):
            # Format the date for display
            predicted_date_str = prediction_result.predicted_next_injury_date.strftime('%Y-%m-%d')
            fig.add_vline(
                x=prediction_result.predicted_next_injury_date.timestamp() * 1000,
                name='Predicted',
                line_width=2,
                line_dash="dot",
                line_color="purple",
                annotation_text=f"Predicted: {predicted_date_str}",
                annotation_position="top right"
            )

        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Speed (mph)", row=1, col=1)
        fig.update_yaxes(title_text="Spin Rate (rpm)", row=2, col=1)
        fig.update_yaxes(title_text="Workload", row=3, col=1)

        # Show the figure
        fig.show()
        return None

    def rkf_box_plot(self) -> None:
        """
        Create a box plot to visualize the distribution of MAE scores from repeated k-fold cross-validation.
        """
        # Get the validation scores using the rkf_validate method
        scores = self.validation.rkf_validate()

        # Create a box plot with all data points shown as individual dots
        plt.figure(figsize=(10, 6))
        plt.boxplot(scores, vert=False, showmeans=True)
        plt.scatter(scores, np.ones_like(scores), color='red', alpha=0.6, label='All Points')
        plt.title('Distribution of MAE Scores from Repeated K-Fold Cross-Validation')
        plt.xlabel('Mean Absolute Error (MAE)')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()