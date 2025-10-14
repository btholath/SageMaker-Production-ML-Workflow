import mlflow
import numpy as np
from data import X_train, X_val, y_train, y_val
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
from params import ridge_param_grid, elasticnet_param_grid, xgb_param_grid
from utils import eval_metrics

# Train ElasticNet models with hyperparameter tuning
for params in ParameterGrid(elasticnet_param_grid):
    with mlflow.start_run():
        # Initialize and train ElasticNet model
        lr = ElasticNet(**params)
        lr.fit(X_train, y_train)

        # Predict on validation set
        y_pred = lr.predict(X_val)

        # Calculate evaluation metrics
        metrics = eval_metrics(y_val, y_pred)

        # Log training dataset
        mlflow.log_input(
            mlflow.data.from_numpy(X_train.toarray()),
            context='Training dataset'
        )

        # Log validation dataset
        mlflow.log_input(
            mlflow.data.from_numpy(X_val.toarray()),
            context='Validation dataset'
        )

        # Log hyperparameters
        mlflow.log_params(params)

        # Log metrics (RMSE, MAPE, R2)
        mlflow.log_metrics(metrics)

        # Log the trained model with input example and code dependencies
        mlflow.sklearn.log_model(
            lr,
            name="ElasticNet",
            input_example=X_train[:1],
            code_paths=['train.py', 'data.py', 'params.py', 'utils.py']
        )