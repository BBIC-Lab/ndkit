import sys
import os
import logging
import csv
import pandas as pd


def set_logger(work_dir, log_name: str = "main") -> logging.Logger:
    """
    Create and configure a logger that writes messages to both console and a log file.

    Args:
        work_dir (str): Directory to store the log file.
        log_name (str): Name of the logger and log file prefix.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(log_name)
    log_file = os.path.join(work_dir, f"{log_name}.log")

    # Clear old handlers if re-initializing the logger
    logger.handlers = []
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%y.%m.%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class CSVLogger:
    """
    A simple CSV logger that writes rows to a CSV file.
    """

    def __init__(self, filepath: str, fieldnames=None):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.initialized = False

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def log(self, row: dict):
        """
        Append a row of values to the CSV file. On first call,
        header fields are inferred unless explicitly provided.
        """
        if not self.initialized:
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())

            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

            self.initialized = True

        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
