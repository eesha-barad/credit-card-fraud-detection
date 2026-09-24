"""
data_loader.py

Loads the credit card transaction dataset from disk.

Dataset: Credit Card Fraud Detection (ULB Machine Learning Group)
Link:    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]


def load_data(path: str) -> pd.DataFrame:
    """
    Load the transaction dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the creditcard.csv file (or a compatible dataset with the
        same schema: V1..V28, Time, Amount, Class).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file is missing expected columns.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it at this path, or update the path in config/config.yaml.\n"
            "See dataset/README.md for detailed instructions."
        )

    df = pd.read_csv(csv_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at '{path}' is missing expected columns: {sorted(missing)}. "
            "Make sure you downloaded the correct 'creditcard.csv' file."
        )

    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a small summary dict useful for logging / reports."""
    fraud_count = int(df["Class"].sum())
    total = len(df)
    return {
        "rows": total,
        "columns": df.shape[1],
        "fraud_count": fraud_count,
        "genuine_count": total - fraud_count,
        "fraud_percentage": round(100 * fraud_count / total, 4) if total else 0.0,
    }
