# Import tools to work with data
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import logging

# Set up a log to track what’s happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load house data from files
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Separate features (like house size) and target (house price)
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Split data into training (to teach) and validation (to test) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize tool to fill missing numbers
imputer = KNNImputer()

# Find number columns (like house size) and word columns (like neighborhood)
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
non_numeric_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

# Fill missing numbers using a smart guess (KNN)
X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

# Fill missing words with the most common word and group rare words
for column in non_numeric_cols:
    # Find the most common word for each dataset
    mode_train = X_train[column].mode()[0]
    mode_val = X_val[column].mode()[0]
    mode_test = test[column].mode()[0]
    # Replace rare categories (appearing less than 10 times) with "Other"
    threshold = 10
    counts = X_train[column].value_counts()
    rare = counts[counts < threshold].index
    X_train[column] = X_train[column].replace(rare, 'Other').fillna(mode_train)
    X_val[column] = X_val[column].replace(rare, 'Other').fillna(mode_val)
    test[column] = test[column].replace(rare, 'Other').fillna(mode_test)

# Turn word columns into numbers (like turning "red" into 1, "blue" into 2)
ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True)
X_train = ohe.fit_transform(X_train)  # Learn the words from training data
X_val = ohe.transform(X_val)  # Apply to validation data
test = ohe.transform(test)  # Apply to test data