# Baseball Pitcher Injury Predictor

A machine learning system that predicts when baseball pitchers are likely to experience their next injury based on their pitching patterns and injury history.

## Project Overview

This project analyzes the relationship between pitching patterns and injuries in baseball pitchers. By examining historical pitch data (workloads, velocity, spin rate, pitch types) and injury records, the system builds a predictive model that can estimate when a pitcher might be at risk for their next injury.

Key features:
- Data preprocessing and feature engineering for baseball statistics
- Machine learning model using XGBoost regression
- Hyperparameter optimization for improved prediction accuracy
- Interactive web interface for making predictions

## Data Sources

The system uses two main types of data:
- **Pitch logs**: Detailed pitch-by-pitch data from Statcast (MLB's tracking system)
- **Injury records**: Historical injury data for baseball pitchers from Fangraphs

Scraped data files from 2020 to 2024 are included in the `data/` directory:
- `2020to2024injuries.csv`: Pitcher injury records
- `20XXpitchlog.csv`: Pitch tracking data from various seasons

This is not a real-time system; it uses historical data to make predictions about future injuries. Only data from the MLB 2020 to 2024 seasons is used for training and testing the model.

## Model Details

The prediction model uses XGBoost regression to predict the number of days until a pitcher's next injury. Features include:
- Pitch count since last injury
- Most common pitch type
- Average pitch velocity
- Average spin rate

The model is optimized using RandomizedSearchCV to find hyperparameters, resulting in improved prediction accuracy compared to the baseline model.

## Project Structure

- `data_preprocessing.py`: Data loading, cleaning, and feature engineering
- `injury_prediction.py`: Core prediction functionality using XGBoost
- `main.py`: Command-line interface for running predictions, viewing metrics, and optimizing the model
- `baseball.ipynb`: Jupyter notebook with interactive interface
- `Dockerfile`: Container configuration for deployment
- `requirements.txt`: Python dependencies
- `data/`: Scraped data files for injuries and pitch logs
- `docs/`: Documentation of functions in HTML, created by Sphinx
- `utils_data/`: Utility functions for data scraping, converting, and migrating
- `utils_model/`: Utility functions for model metrics, optimization, validation, and visualizations

## Installation

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)
- Git (optional, for cloning the repository)
- 3 GB+ of RAM (for running the model efficiently)
- 3 GB+ of disk space (for storing data files and model artifacts)
- Internet connection (for downloading dependencies and data)

### Option 1: Docker Installation

1. Clone the repository:
   ```
   git clone https://github.com/tclvgh/injury-predictor
   cd injury-predictor
   ```

2. Build the Docker image:
   ```
   docker build -t injury-predictor .
   ```

3. Run the container:
   ```
   docker run -p 8866:8866 injury-predictor
   ```

4. Access the application in your browser at `http://localhost:8866`
   (this may take up to 10 minutes to load the data and train the model)

### Option 2: Direct Installation

1. Clone the repository:
   ```
   git clone https://github.com/tclvgh/injury-predictor
   cd injury-predictor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   voila baseball.ipynb
   ```

4. Access the application in your browser at `http://localhost:8866`
   (this may take up to 10 minutes to load the data and train the model)

## Usage

### Data Preprocessing

The `DataProcessor` class in `data_preprocessing.py` is used to load and preprocess raw baseball data:

```python
from data_preprocessing import DataProcessor

# Initialize the data processor and load the data
processor = DataProcessor('data')

# Process the data
processed_injury_data = processor.process_injury_data()
processed_pitch_log_data = processor.process_pitch_data()
```

### Making Predictions

The system can predict when a pitcher might experience their next injury based on:
- Their most recent injury date
- Their pitching patterns since that injury in relation to their historical data

```python
from injury_prediction import InjuryPredictor
from data_preprocessing import g_x_train, g_y_train

# Initialize the predictor
predictor = InjuryPredictor(g_x_train, g_y_train)

# Make a prediction for a specific pitcher
result = predictor.predict_next_injury("Zack Wheeler")

# Display the prediction
print(f"Pitcher: {result.pitcher_name}")
print(f"Most recent injury: {result.most_recent_injury.strftime('%Y-%m-%d')}")
print(f"Predicted days to next injury: {result.predicted_days_to_next_injury}")
print(f"Predicted next injury date: {result.predicted_next_injury_date.strftime('%Y-%m-%d')}")
```

### Model Evaluation

The `ModelMetrics` class in `/utils_model/metrics.py` is to assess model performance:

```python
import utils_model.metrics as me
from data_preprocessing import g_x_test, g_y_test, g_x_train, g_y_train
from injury_prediction import InjuryPredictor

predictor = InjuryPredictor(g_x_train, g_y_train)

# Initialize the evaluator with true and predicted values
evaluator = me.ModelMetrics(predictor.model, g_x_test, g_y_test, "My Injury Prediction Model")

# Display the evaluation metrics, including MAE, MSE, and R² and a linear regression plot graph
evaluator.print_metrics()
```

### Model Optimization

The `ModelOptimizer` class in `/utils_model/optimization.py` is to optimize the model performance:

```python
import utils_model.optimization as mo
from data_preprocessing import g_x_train, g_y_train, g_x_test, g_y_test

# Initialize the optimizer
optimizer = mo.ModelOptimizer(g_x_train, g_y_train, g_x_test, g_y_test)

optimizer.print_improvements()
optimizer.print_hyperparameters()
```

### Model Visualizations

The `ModelVisualizations` class in `/utils_model/visualizations.py` is to graph and plot visualizations for the model's predictions:

```python
import utils_model.visualizations as mv
from data_preprocessing import g_x_train, g_y_train, g_x_test, g_y_test
from injury_prediction import InjuryPredictor

# Initialize the predictor
injury_predictor = InjuryPredictor(g_x_train, g_y_train)

# Initialize the visualizations with the predictor and test data
visualization = mv.ModelVisualizations(injury_predictor, g_x_test, g_y_test)

# Create visualizations for the model
visualization.features_injuries_timeline("Wheeler, Zack")
visualization.feature_importance_bar_graph()
visualization.linear_regression_plot()
```