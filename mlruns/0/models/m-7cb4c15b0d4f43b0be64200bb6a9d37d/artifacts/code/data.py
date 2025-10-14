import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load training and test datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Define features and target variable
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN imputer for numerical data
imputer = KNNImputer()

# Separate numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
non_numeric_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

# Impute missing values for numeric columns using KNN
X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
test[numeric_cols] = imputer.transform(test[numeric_cols])

# Impute missing values for non-numeric columns with mode (avoid inplace=True)
for column in non_numeric_cols:
    mode_train = X_train[column].mode()[0]
    mode_val = X_val[column].mode()[0]
    mode_test = test[column].mode()[0]
    X_train[column] = X_train[column].fillna(mode_train)
    X_val[column] = X_val[column].fillna(mode_val)
    test[column] = test[column].fillna(mode_test)

# Apply one-hot encoding to categorical features
ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=True)

# Fit encoder on training data
X_train = ohe.fit_transform(X_train)

# Transform validation and test data, checking for unknown categories
try:
    X_val = ohe.transform(X_val)
except ValueError as e:
    logging.warning(f"Unknown categories in X_val: {e}")
    # Optionally, inspect unique values in problematic columns
    for col in non_numeric_cols:
        val_unique = set(X_val[col].dropna()) - set(X_train[col].dropna())
        if val_unique:
            logging.info(f"Column {col}: Unknown values in X_val: {val_unique}")

try:
    test = ohe.transform(test)
except ValueError as e:
    logging.warning(f"Unknown categories in test: {e}")
    for col in non_numeric_cols:
        test_unique = set(test[col].dropna()) - set(X_train[col].dropna())
        if test_unique:
            logging.info(f"Column {col}: Unknown values in test: {test_unique}")