"""
preprocessing.py

Feature scaling, train/test splitting, and SMOTE-based class balancing for the
fraud detection pipeline.
"""

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scale_amount_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale the 'Amount' and 'Time' columns (V1-V28 are already PCA-scaled in the
    source dataset). Returns a new DataFrame with 'Amount_scaled' and
    'Time_scaled' columns added; does not mutate the input in place.
    """
    df = df.copy()
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of model feature columns (excludes raw Time/Amount/Class)."""
    return [c for c in df.columns if c not in ("Time", "Amount", "Class")]


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the scaled feature set."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def balance_with_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority (fraud) class in the training set using SMOTE."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def prepare_dataset(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """
    Full preprocessing pipeline: scale -> split -> SMOTE-balance the training set.

    Returns
    -------
    X_train_res, y_train_res, X_test, y_test
    """
    df_scaled = scale_amount_and_time(df)
    X_train, X_test, y_train, y_test = split_data(df_scaled, test_size, random_state)
    X_train_res, y_train_res = balance_with_smote(X_train, y_train, random_state)
    return X_train_res, y_train_res, X_test, y_test
