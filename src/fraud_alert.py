"""
fraud_alert.py

A small reusable module simulating how a bank's real-time transaction
monitoring system would use the trained AI model in production: scoring a
transaction and issuing a risk level and recommended action, rather than a
raw probability.
"""

from typing import Any, Dict


def score_transaction(
    transaction_row, model: Any, low: float = 0.3, high: float = 0.7
) -> Dict[str, Any]:
    """
    Score a single transaction and return a risk assessment.

    Parameters
    ----------
    transaction_row : pandas.DataFrame
        A single-row DataFrame with the same feature columns the model was
        trained on.
    model : a fitted classifier with predict_proba()
    low, high : float
        Probability thresholds separating LOW / MEDIUM / HIGH risk bands.

    Returns
    -------
    dict with keys: fraud_probability, risk_level, recommended_action
    """
    proba = model.predict_proba(transaction_row)[0][1]

    if proba < low:
        risk_level, action = "LOW", "Approve transaction"
    elif proba < high:
        risk_level, action = "MEDIUM", "Flag for manual review"
    else:
        risk_level, action = "HIGH", "Block transaction & alert customer"

    return {
        "fraud_probability": round(float(proba), 4),
        "risk_level": risk_level,
        "recommended_action": action,
    }
