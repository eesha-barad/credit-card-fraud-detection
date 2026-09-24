"""
explainability.py

Explainable AI (XAI) utilities using SHAP, so fraud decisions can be justified
to analysts, customers, and regulators rather than treated as a black box.
"""

from typing import Tuple

import numpy as np
import shap


def get_fraud_class_shap(shap_values, class_index: int = 1) -> np.ndarray:
    """
    Return SHAP values for a specific class, regardless of the installed SHAP
    library's output format.

    Older SHAP versions return a list [class0_values, class1_values] from
    TreeExplainer.shap_values() on a classifier. Newer versions (0.44+) return
    a single array of shape (n_samples, n_features, n_classes). This helper
    normalizes both cases to a single (n_samples, n_features) array.
    """
    if isinstance(shap_values, list):
        return np.array(shap_values[class_index])

    arr = np.array(shap_values)
    if arr.ndim == 3:
        return arr[:, :, class_index]
    return arr


def get_expected_value(explainer, class_index: int = 1):
    """
    Return the base/expected value for a specific class, handling both the
    scalar and per-class-array forms that explainer.expected_value can take
    across SHAP versions.
    """
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
        return expected_value[class_index]
    return expected_value


def build_explainer(model) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for a tree-based model (e.g. Random Forest)."""
    return shap.TreeExplainer(model)


def explain_global(model, X_sample, class_index: int = 1) -> Tuple[np.ndarray, shap.TreeExplainer]:
    """
    Compute SHAP values for a sample of transactions, for global feature
    importance analysis (which features drive fraud predictions overall).

    Returns (shap_values_for_class, explainer).
    """
    explainer = build_explainer(model)
    shap_values = explainer.shap_values(X_sample)
    return get_fraud_class_shap(shap_values, class_index), explainer


def explain_instance(explainer, row, class_index: int = 1) -> shap.Explanation:
    """
    Build a shap.Explanation object for a single transaction row, suitable
    for shap.plots.waterfall() -- explains why one specific transaction was
    scored the way it was.
    """
    single_shap_values = explainer.shap_values(row)
    single_shap_fraud = get_fraud_class_shap(single_shap_values, class_index)
    expected_value_fraud = get_expected_value(explainer, class_index)

    return shap.Explanation(
        values=single_shap_fraud[0],
        base_values=expected_value_fraud,
        data=row.iloc[0].values,
        feature_names=list(row.columns),
    )
