#
# Hyperparameter Configuration File
#
# Purpose: To define hyperparameter search spaces for different regression models.
#          These grids are used during model tuning to find the optimal combination
#          of parameters for each algorithm.
# Outcome: A set of dictionaries (param_grids) that can be iterated over by
#          training scripts (e.g., train.py) to perform a grid search.
#

# --- Ridge Regression Parameters ---
# Purpose: Defines the hyperparameter grid for the Ridge model. Ridge uses L2
#          regularization to prevent overfitting by penalizing large coefficients.
ridge_param_grid = {
    # 'alpha': Controls the strength of regularization. Higher values increase
    #          the penalty, leading to simpler models.
    'alpha': [0.1, 1.0, 10.0],

    # 'fit_intercept': Specifies whether to calculate the intercept for this model.
    #                  If set to False, no intercept will be used in calculations.
    'fit_intercept': [True, False],
}

# --- ElasticNet Parameters ---
# Purpose: Defines the hyperparameter grid for the ElasticNet model. ElasticNet
#          combines L1 and L2 regularization, making it useful when there are
#          multiple correlated features.
elasticnet_param_grid = {
    # 'alpha': The overall strength of the regularization (both L1 and L2).
    'alpha': [0.1, 1.0, 10.0],

    # 'l1_ratio': The mixing parameter. An l1_ratio of 1 corresponds to Lasso (L1),
    #             while a ratio of 0 corresponds to Ridge (L2).
    'l1_ratio': [0.2, 0.5, 0.8],

    # 'fit_intercept': Specifies whether to calculate the model's intercept.
    'fit_intercept': [True, False],
}

# --- XGBoost Parameters ---
# Purpose: Defines a comprehensive hyperparameter grid for the XGBoost model,
#          a powerful gradient boosting algorithm.
xgb_param_grid = {
    # 'n_estimators': The number of gradient boosted trees. Equivalent to the number of boosting rounds.
    'n_estimators': [100, 200, 300],

    # 'learning_rate': Step size shrinkage used to prevent overfitting.
    'learning_rate': [0.01, 0.1, 0.2],

    # 'max_depth': The maximum depth of a tree.
    'max_depth': [3, 4, 5],

    # 'min_child_weight': Minimum sum of instance weight needed in a child.
    'min_child_weight': [1, 2, 3],

    # 'subsample': The fraction of observations to be randomly sampled for each tree.
    'subsample': [0.8, 0.9, 1.0],

    # 'colsample_bytree': The fraction of columns to be randomly sampled for each tree.
    'colsample_bytree': [0.8, 0.9, 1.0],

    # 'gamma': Minimum loss reduction required to make a further partition on a leaf node.
    'gamma': [0, 0.1, 0.2],

    # 'reg_alpha': L1 regularization term on weights.
    'reg_alpha': [0, 0.1, 1.0],

    # 'reg_lambda': L2 regularization term on weights.
    'reg_lambda': [0, 0.1, 1.0],
}