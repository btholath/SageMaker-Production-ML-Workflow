"""
Imagine you’re teaching a robot to guess how much a house costs based on things like its size or number of bedrooms. The train.py script is like a teacher who trains the robot using a specific method called ElasticNet. It tries different settings to make the robot’s guesses as accurate as possible and saves all the results in a notebook (called MLflow) so you can check how well the robot did.

The script is like a teacher training a robot (ElasticNet) to guess house prices.
It tries different settings (like how strict the robot should be) from params.py.
It uses house data from data.py (like size, bedrooms) to teach the robot (X_train, y_train) and test its guesses (X_val, y_val).
It checks how good the guesses are using tools from utils.py (like how close the guesses are to real prices).
It saves everything—data, settings, scores, and the robot—in a notebook called MLflow so you can look back and pick the best robot.
"""

# Import tools to help the robot learn and save its progress
import mlflow  # Like a notebook to save the robot's learning progress
import numpy as np  # Helps with math calculations
from data import X_train, X_val, y_train, y_val  # The house data to learn from
from sklearn.linear_model import Ridge, ElasticNet  # The robot's learning methods
from xgboost import XGBRegressor  # Another learning method (not used here)
from sklearn.model_selection import ParameterGrid  # Helps try different settings
from sklearn.preprocessing import StandardScaler  # Makes numbers similar in size
from params import ridge_param_grid, elasticnet_param_grid, xgb_param_grid  # List of settings to try
from utils import eval_metrics  # Tools to check how well the robot guesses

# Make all numbers similar in size (like making all ingredients the same size for baking)
scaler = StandardScaler(with_mean=False)  # Use with_mean=False for sparse data
X_train = scaler.fit_transform(X_train)  # Learn scaling from training data
X_val = scaler.transform(X_val)  # Apply scaling to validation data

# Try different settings for the ElasticNet robot
for params in ParameterGrid(elasticnet_param_grid):  # Loop through different settings
    with mlflow.start_run():  # Start a new page in the MLflow notebook
        # Create and teach the ElasticNet robot with more steps for learning
        lr = ElasticNet(**params, max_iter=10000)  # Allow 10,000 steps to learn
        lr.fit(X_train, y_train)  # Teach the robot using training house data

        # Ask the robot to guess house prices for validation data
        y_pred = lr.predict(X_val)  # Get the robot's guesses

        # Check how good the guesses are
        metrics = eval_metrics(y_val, y_pred)  # Measure how close guesses are to real prices

        # Save the training house data in the notebook
        mlflow.log_input(
            mlflow.data.from_numpy(X_train.toarray()),  # Save house info used for training
            context='Training dataset'
        )

        # Save the validation house data in the notebook
        mlflow.log_input(
            mlflow.data.from_numpy(X_val.toarray()),  # Save house info used for checking
            context='Validation dataset'
        )

        # Save the settings used for this robot
        mlflow.log_params(params)  # Write down settings, like "used 0.1 sugar"

        # Save how well the robot did
        mlflow.log_metrics(metrics)  # Write down scores like "90% accurate"

        # Save the trained robot so we can use it later
        mlflow.sklearn.log_model(
            lr,  # The trained robot
            name="ElasticNet",  # Name it "ElasticNet"
            input_example=X_train[:1],  # Save an example house
            code_paths=['train.py', 'data.py', 'params.py', 'utils.py']  # Save instructions
        )