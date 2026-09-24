"""Shared pytest fixtures: a small synthetic dataset shaped like creditcard.csv."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """
    A small synthetic dataset with the same schema as the real
    Credit Card Fraud Detection dataset (V1-V28, Time, Amount, Class),
    used so tests run fast and never require the real (large, licensed) dataset.
    """
    rng = np.random.default_rng(42)
    n = 600
    n_fraud = 20

    data = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["Time"] = rng.integers(0, 172792, n).astype(float)
    data["Amount"] = np.abs(rng.normal(loc=20, scale=50, size=n))

    y = np.zeros(n, dtype=int)
    fraud_idx = rng.choice(n, n_fraud, replace=False)
    y[fraud_idx] = 1

    # Make fraud rows separable so trained models have something real to learn
    for c in [f"V{i}" for i in range(1, 6)]:
        arr = np.array(data[c])
        arr[fraud_idx] += 3.5
        data[c] = arr

    data["Class"] = y
    return pd.DataFrame(data)
