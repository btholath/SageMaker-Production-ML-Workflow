#
# ElasticNet Model Training and Tuning Script
#
# Purpose: To train and evaluate an ElasticNet regression model using a grid of
#          hyperparameters. All experiment details, including parameters, metrics,
#          and the trained model, are logged to MLflow.
# Steps:
#   1. Load preprocessed data.
#   2. Scale the feature matrices.
#   3. Iterate through a grid of ElasticNet hyperparameters.
#   4. For each parameter set, start an MLflow run.
#   5. Train the model, make predictions, and calculate evaluation metrics.
#   6. Log all inputs, parameters, metrics, and the final model to MLflow.
# Outcome: Multiple runs logged to MLflow, one for each hyperparameter combination,
#          allowing for easy comparison and selection of the best model.
#

import mlflow
import numpy as np
from data import X_train, X_val, y_train, y_val
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from params import ridge_param_grid, elasticnet_param_grid, xgb_param_grid
from utils import eval_metrics

# Step 1: Scale the feature data. StandardScaler is used to normalize the features,
# which is important for linear models like ElasticNet.
scaler = StandardScaler(with_mean=False)  # with_mean=False is suitable for sparse data.
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 2: Iterate over each combination of hyperparameters defined in `params.py`.
print("Starting hyperparameter tuning for ElasticNet...")
for params in ParameterGrid(elasticnet_param_grid):
    # Step 3: Start a new MLflow run for each hyperparameter set.
    with mlflow.start_run():
        print(f"Training with params: {params}")

        # Step 4: Initialize and train the ElasticNet model.
        # Increased `max_iter` to ensure convergence.
        lr = ElasticNet(**params, max_iter=10000)
        lr.fit(X_train, y_train)

        # Step 5: Make predictions on the validation set.
        y_pred = lr.predict(X_val)

        # Step 6: Evaluate the model's performance using custom metrics.
        metrics = eval_metrics(y_val, y_pred)

        # Step 7: Log all relevant information to MLflow for tracking.
        # Log input datasets for reproducibility.
        mlflow.log_input(mlflow.data.from_numpy(X_train.toarray()), context='Training dataset')
        mlflow.log_input(mlflow.data.from_numpy(X_val.toarray()), context='Validation dataset')

        # Log the hyperparameters used in this run.
        mlflow.log_params(params)

        # Log the performance metrics.
        mlflow.log_metrics(metrics)

        # Log the trained model, its name, an input example, and associated code.
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="ElasticNet",  # The subdirectory for the model in artifacts.
            input_example=X_train[:1],
            code_paths=['train.py', 'data.py', 'params.py', 'utils.py']
        )
print("Hyperparameter tuning complete.")