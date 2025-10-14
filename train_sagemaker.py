#
# SageMaker Training Script for XGBoost
#
# Purpose: To train an XGBoost model within an AWS SageMaker training job. This script
#          is designed to be run in the SageMaker environment, reading data from
#          input channels and saving the model to an output directory. It also logs
#          all experiment details to a remote MLflow server.
# Steps:
#   1. Parse command-line arguments provided by the SageMaker environment.
#   2. Set up MLflow tracking to point to a remote server.
#   3. Load training and validation data from specified SageMaker channels.
#   4. Define XGBoost hyperparameters.
#   5. Train the model, evaluate its performance, and log results to MLflow.
#   6. Save the trained model to the SageMaker model directory for deployment.
# Outcome: A trained XGBoost model saved in the specified S3 location and a
#          corresponding run logged in MLflow.
#
import mlflow
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import os
import argparse

# Helper function to calculate evaluation metrics.
def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAPE": mape, "R2": r2}

def main():
    # Step 1: Parse arguments passed by the SageMaker training job.
    # These environment variables point to data channels and output directories in S3.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()

    # Step 2: Configure MLflow to connect to the remote tracking server.
    # IMPORTANT: Replace <ec2-public-ip> with the actual public IP of your server.
    mlflow.set_tracking_uri("http://<ec2-public-ip>:5000")
    mlflow.set_experiment("SageMaker_Training")

    # Step 3: Load training and validation data from the specified input channels.
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    val_data = pd.read_csv(os.path.join(args.validation, 'validation.csv'))
    X_train, y_train = train_data.drop('SalePrice', axis=1), train_data['SalePrice']
    X_val, y_val = val_data.drop('SalePrice', axis=1), val_data['SalePrice']

    # Step 4: Define the hyperparameters for the XGBoost model.
    params = {
        'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 4,
        'min_child_weight': 1, 'subsample': 0.9, 'colsample_bytree': 0.9,
        'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0
    }

    # Step 5: Start an MLflow run to track the training process.
    with mlflow.start_run():
        # Train the XGBoost model.
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate the model on the validation set.
        y_pred = model.predict(X_val)
        metrics = eval_metrics(y_val, y_pred)

        # Log parameters, metrics, and the model to MLflow.
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "XGBRegressor", input_example=X_val.iloc[:1])

        # Step 6: Save the trained model to the directory specified by SageMaker.
        # This makes the model available for deployment.
        model.save_model(os.path.join(args.model_dir, 'model.xgb'))

if __name__ == "__main__":
    main()