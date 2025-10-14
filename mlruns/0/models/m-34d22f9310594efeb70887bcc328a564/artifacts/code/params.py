# Hyperparameter grids for model tuning

# Ridge regression parameters
ridge_param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Regularization strength
    'fit_intercept': [True, False],  # Whether to fit intercept
}

# ElasticNet parameters
elasticnet_param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Regularization strength
    'l1_ratio': [0.2, 0.5, 0.8],  # Balance between L1 and L2 regularization
    'fit_intercept': [True, False],  # Whether to fit intercept
}

# XGBoost parameters
xgb_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
    'max_depth': [3, 4, 5],  # Maximum tree depth
    'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples for training
    'colsample_bytree': [0.8, 0.9, 1.0],  # Fraction of features for training
    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction for splits
    'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
    'reg_lambda': [0, 0.1, 1.0],  # L2 regularization
}