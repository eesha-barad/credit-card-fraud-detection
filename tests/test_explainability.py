import numpy as np

from src.explainability import (
    explain_global,
    explain_instance,
    get_expected_value,
    get_fraud_class_shap,
)
from src.models import train_random_forest
from src.preprocessing import prepare_dataset


def test_get_fraud_class_shap_handles_list_format():
    """Older SHAP versions: shap_values is a list [class0_array, class1_array]."""
    class0 = np.zeros((5, 3))
    class1 = np.ones((5, 3))
    result = get_fraud_class_shap([class0, class1], class_index=1)
    assert result.shape == (5, 3)
    assert (result == 1).all()


def test_get_fraud_class_shap_handles_3d_array_format():
    """Newer SHAP versions (0.44+): shap_values is one array (samples, features, classes)."""
    arr = np.zeros((5, 3, 2))
    arr[:, :, 1] = 7
    result = get_fraud_class_shap(arr, class_index=1)
    assert result.shape == (5, 3)
    assert (result == 7).all()


def test_get_fraud_class_shap_passthrough_2d_array():
    """A plain 2D array (single-output regressor style) should pass through unchanged."""
    arr = np.ones((4, 6))
    result = get_fraud_class_shap(arr, class_index=1)
    assert result.shape == (4, 6)


def test_get_expected_value_scalar():
    class FakeExplainer:
        expected_value = 0.42

    assert get_expected_value(FakeExplainer(), class_index=1) == 0.42


def test_get_expected_value_array():
    class FakeExplainer:
        expected_value = np.array([0.6, 0.4])

    assert get_expected_value(FakeExplainer(), class_index=1) == 0.4


def test_explain_global_end_to_end_with_real_shap(synthetic_df):
    """Full integration check against whatever SHAP version is actually installed."""
    X_train_res, y_train_res, X_test, y_test = prepare_dataset(synthetic_df, random_state=42)
    model = train_random_forest(
        X_train_res, y_train_res, total_trees=20, step=20, max_depth=4,
        random_state=42, show_progress=False,
    )
    sample_X = X_test.sample(min(20, len(X_test)), random_state=42)

    shap_values_fraud, explainer = explain_global(model, sample_X)
    assert shap_values_fraud.shape == sample_X.shape

    single_row = X_test.iloc[[0]]
    explanation = explain_instance(explainer, single_row)
    assert len(explanation.values) == single_row.shape[1]
