#
# Data Loading and Preprocessing Script
#
# Purpose: To load the raw housing data, clean it, and transform it into a format
#          suitable for machine learning model training.
# Steps:
#   1. Load train and test datasets from CSV files.
#   2. Split the training data into training and validation sets.
#   3. Impute missing numerical values using KNNImputer.
#   4. Impute missing categorical values using the mode and group rare categories.
#   5. Convert categorical features into a numerical format using OneHotEncoder.
# Outcome: Exports preprocessed, sparse matrices (X_train, X_val, test) and
#          target arrays (y_train, y_val) for use in model training and inference.
#

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import logging

# Step 1: Set up logging for monitoring the script's execution.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 2: Load raw data from CSV files.
logging.info("Loading training and test data.")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Step 3: Separate features (X) from the target variable (y).
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Step 4: Split the data into training and validation sets for model evaluation.
logging.info("Splitting data into training and validation sets.")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Handle missing numerical values using K-Nearest Neighbors imputation.
logging.info("Imputing missing numerical features.")
imputer = KNNImputer()

# Identify numerical and non-numerical columns to apply specific transformations.
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
non_numeric_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

# Apply KNN imputation to the numerical columns of all datasets.
X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

# Step 6: Handle missing categorical values and consolidate rare categories.
logging.info("Imputing missing categorical features and handling rare categories.")
for column in non_numeric_cols:
    # Determine the most frequent category (mode) for imputation.
    mode_train = X_train[column].mode()[0]
    mode_val = X_val[column].mode()[0]
    mode_test = test[column].mode()[0]

    # Group infrequent categories (threshold < 10) into an 'Other' category.
    threshold = 10
    counts = X_train[column].value_counts()
    rare = counts[counts < threshold].index
    X_train[column] = X_train[column].replace(rare, 'Other').fillna(mode_train)
    X_val[column] = X_val[column].replace(rare, 'Other').fillna(mode_val)
    test[column] = test[column].replace(rare, 'Other').fillna(mode_test)

# Step 7: Apply One-Hot Encoding to convert categorical features into a sparse matrix.
logging.info("Applying One-Hot Encoding to categorical features.")
ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True)

# Fit the encoder on the training data and transform all datasets.
X_train = ohe.fit_transform(X_train)
X_val = ohe.transform(X_val)
test = ohe.transform(test)

logging.info("Data preprocessing complete.")