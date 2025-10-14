#
# Model Evaluation Utilities
#
# Purpose: To provide reusable functions for calculating performance metrics
#          for machine learning models.
# Outcome: A centralized location for evaluation logic, ensuring consistent
#          metric calculation across different training scripts.
#

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def eval_metrics(y_true, y_pred):
    """
    Calculates and returns a dictionary of common regression evaluation metrics.

    Args:
        y_true (array-like): The ground truth (correct) target values.
        y_pred (array-like): The estimated target values predicted by the model.

    Returns:
        dict: A dictionary containing the calculated RMSE, MAPE, and R2 score.
    """
    # Calculate Root Mean Squared Error (RMSE).
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate Mean Absolute Percentage Error (MAPE).
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Calculate R-squared (R2) score, the coefficient of determination.
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }