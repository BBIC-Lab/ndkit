import numpy as np
from sklearn.metrics import mean_squared_error


def VAF(y_pred, y):
    """
    Variance Accounted For (VAF).

    Args:
        y_pred (ndarray): Predicted values, shape (N, D)
        y      (ndarray): Ground truth values, shape (N, D)

    Returns:
        float: Mean VAF across feature dimensions.
    """
    if y.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    epsilon = 1e-10
    y_mean = y.mean(axis=0)

    residual_var = np.sum((y - y_pred) ** 2, axis=0)
    total_var = np.sum((y - y_mean) ** 2, axis=0) + epsilon

    vaf = 1 - residual_var / total_var
    return float(np.mean(vaf))


def RMSE(y_pred, y):
    """
    Root Mean Square Error (RMSE).

    Args:
        y_pred (ndarray): Predicted values, shape (N, D)
        y      (ndarray): Ground truth values, shape (N, D)

    Returns:
        float: RMSE value.
    """
    if y.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")
    return np.sqrt(mean_squared_error(y, y_pred))


def CC(y_pred, y):
    """
    Mean feature-wise Pearson correlation coefficient.

    Args:
        y_pred (ndarray): Predicted values, shape (N, D)
        y      (ndarray): Ground truth values, shape (N, D)

    Returns:
        float: Mean correlation across feature dimensions.
    """
    if y.shape != y_pred.shape:
        raise ValueError("Input arrays must have the same shape.")

    # Vectorized Pearson correlation
    y_centered = y - y.mean(axis=0)
    y_pred_centered = y_pred - y_pred.mean(axis=0)

    numerator = np.sum(y_centered * y_pred_centered, axis=0)
    denominator = np.sqrt(
        np.sum(y_centered ** 2, axis=0) * np.sum(y_pred_centered ** 2, axis=0) + 1e-10
    )

    cc = numerator / denominator
    return float(np.mean(cc))


def compute_metrics(y_pred, y):
    """
    Compute RMSE, CC, and VAF.

    Args:
        y_pred (ndarray): Predictions
        y      (ndarray): Ground truth

    Returns:
        dict: {"vaf": ..., "rmse": ..., "cc": ...}
    """
    return {
        "vaf":  VAF(y_pred, y),
        "rmse": RMSE(y_pred, y),
        "cc":   CC(y_pred, y),
    }
