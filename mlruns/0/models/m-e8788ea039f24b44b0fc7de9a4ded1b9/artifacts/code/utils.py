import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def eval_metrics(y_true, y_pred):
    """Calculate evaluation metrics for regression models."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
    mape = mean_absolute_percentage_error(y_true, y_pred)  # Mean Absolute Percentage Error
    r2 = r2_score(y_true, y_pred)  # R-squared score
    return {
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }