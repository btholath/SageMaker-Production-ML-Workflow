#
# SageMaker Deployment Script
#
# Purpose: To deploy a trained MLflow model from a specified S3 URI to an
#          AWS SageMaker endpoint for real-time inference.
# Steps:
#   1. Define configuration parameters for the SageMaker endpoint.
#   2. Specify the S3 URI of the model to be deployed.
#   3. Initialize the MLflow SageMaker deployment client.
#   4. Create the deployment using the specified configuration.
# Outcome: An active SageMaker endpoint serving the specified model.
#

import mlflow.sagemaker
from mlflow.deployments import get_deploy_client

# --- Configuration ---
# Step 1: Define the name for the production endpoint and the S3 location of the model.
endpoint_name = "prod-endpoint"
model_uri = "s3://mlflow-project-artifacts/4/d2ad59e0241c4f6f9212ff7e22ca780a/artifacts/XGBRegressor"

# Step 2: Define the SageMaker deployment configuration dictionary.
config = {
    "execution_role_arn": "arn:aws:iam::816680701120:role/house-price-role",  # IAM role with SageMaker permissions.
    "bucket_name": "mlflow-project-artifacts",  # S3 bucket for deployment artifacts.
    "image_url": "816680701120.dkr.ecr.us-east-1.amazonaws.com/xgb:2.9.1",  # ECR Docker image for serving.
    "region_name": "us-east-1",  # AWS region for deployment.
    "archive": False,  # If True, creates an inactive endpoint. False for an active one.
    "instance_type": "ml.m5.xlarge",  # EC2 instance type for the endpoint.
    "instance_count": 1,  # Number of instances to host the endpoint.
    "synchronous": True  # If True, the script waits for deployment to complete.
}

# --- Deployment ---
# Step 3: Initialize the MLflow deployment client for the 'sagemaker' target.
client = get_deploy_client("sagemaker")

# Step 4: Create the deployment. This provisions the SageMaker endpoint.
print(f"Deploying model from {model_uri} to endpoint '{endpoint_name}'...")
client.create_deployment(
    name=endpoint_name,
    model_uri=model_uri,
    flavor="python_function",  # Specifies the MLflow model flavor to use for serving.
    config=config
)
print("Deployment to SageMaker completed successfully.")