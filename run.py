#
# MLflow Project Execution Script
#
# Purpose: To execute the MLflow project locally, triggering the training pipeline
#          defined in the `MLproject` file.
# Steps:
#   1. Define the experiment name and the entry point to run.
#   2. Set the MLflow tracking URI to a remote server.
#   3. Execute the project using `mlflow.projects.run`.
# Outcome: A new run is initiated and logged under the specified experiment name
#          on the remote MLflow tracking server.
#

import mlflow

# --- Configuration ---
# Step 1: Define the experiment name for organizing runs and the entry point to execute.
experiment_name = "ElasticNet"
entry_point = "Training"

# --- MLflow Setup ---
# Step 2: Set the tracking URI to point to the remote MLflow server (e.g., an EC2 instance).
# IMPORTANT: Replace <ec2-public-ip> with the actual public IP of your server.
mlflow.set_tracking_uri("http://<ec2-public-ip>:5000")

# --- Project Execution ---
# Step 3: Run the MLflow project defined in the current directory.
print(f"Running MLflow project entry point '{entry_point}' for experiment '{experiment_name}'...")
mlflow.projects.run(
    uri=".",  # Specifies that the project is in the current directory.
    entry_point=entry_point,  # The entry point to run, as defined in the MLproject file.
    experiment_name=experiment_name,  # The experiment under which to log the run.
    env_manager="local"  # Uses the current local Python environment to run the script.
)
print("MLflow project run initiated successfully.")