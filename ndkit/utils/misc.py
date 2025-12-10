import time
import numpy as np
import hashlib


def get_time_dir() -> str:
    """
    Return a timestamp string (YYMMDD-HHMMSS), suitable for naming result directories.
    """
    return time.strftime("%y%m%d-%H%M%S", time.localtime())


def str2hash(s: str) -> str:
    """
    Compute a short SHA256 hash for a string (used for generating unique IDs).

    Args:
        s (str): Input string.

    Returns:
        str: First 8 characters of the SHA256 hash.
    """
    encoded = s.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest[:8]


def format_metrics(metrics: dict, precision: int = 4, exclude_keys=None) -> str:
    """
    Format a metrics dictionary into a readable string.

    Args:
        metrics (dict): A dictionary of metric_name → value.
        precision (int): Number of decimal places for floating numbers.
        exclude_keys (str or list[str], optional): Metric names to exclude.

    Returns:
        str: Formatted string such as "rmse: 0.1234, cc: 0.8765".
    """
    if isinstance(exclude_keys, str):
        exclude_keys = [exclude_keys]
    exclude_keys = exclude_keys or []

    formatted = []
    for k, v in metrics.items():
        if any(ex in k for ex in exclude_keys):
            continue
        if isinstance(v, (float, int, np.floating, np.integer)):
            formatted.append(f"{k}: {v:.{precision}f}")
        else:
            formatted.append(f"{k}: {v}")

    return ", ".join(formatted)