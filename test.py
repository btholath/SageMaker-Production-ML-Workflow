#
# SageMaker Endpoint Inference Script
#
# Purpose: To send a sample of test data to the deployed SageMaker endpoint and
#          retrieve predictions.
# Steps:
#   1. Define endpoint configuration (name and region).
#   2. Load and prepare the preprocessed test data.
#   3. Initialize boto3 clients for SageMaker runtime.
#   4. Format the test data into the required JSON payload.
#   5. Invoke the endpoint with the payload.
#   6. Decode and print the prediction results.
# Outcome: Prints the model's predictions for the first 20 samples of the test set.
#

from data import test
import boto3
import json

# --- Configuration ---
# Step 1: Specify the name of the deployed endpoint and its AWS region.
endpoint_name = "prod-endpoint"
region = 'us-east-1'

# --- AWS Client Initialization ---
# Step 2: Initialize boto3 clients for SageMaker and SageMaker Runtime.
sm = boto3.client('sagemaker', region_name=region)
smrt = boto3.client('runtime.sagemaker', region_name=region)

# --- Data Preparation ---
# Step 3: Prepare the test data payload for the endpoint.
# The model expects a JSON object with an 'instances' key.
# We take the first 20 samples and convert the sparse matrix to a dense list.
test_samples = test[:20].toarray()[:, :-1].tolist()
test_data_json = json.dumps({'instances': test_samples})

# --- Endpoint Invocation ---
# Step 4: Send the request to the SageMaker endpoint.
print(f"Invoking endpoint '{endpoint_name}' with {len(test_samples)} samples...")
prediction = smrt.invoke_endpoint(
    EndpointName=endpoint_name,
    Body=test_data_json,
    ContentType='application/json'
)

# --- Process Results ---
# Step 5: Decode the response from the endpoint and print the predictions.
prediction_body = prediction['Body'].read().decode("ascii")
print("\nPrediction Results:")
print(prediction_body)