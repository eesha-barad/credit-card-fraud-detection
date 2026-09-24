from src.evaluation import evaluate_model, plot_roc_curves
from src.models import train_random_forest
from src.preprocessing import prepare_dataset


def test_evaluate_model_returns_expected_keys(synthetic_df, tmp_path):
    X_train_res, y_train_res, X_test, y_test = prepare_dataset(synthetic_df, random_state=42)
    model = train_random_forest(
        X_train_res, y_train_res, total_trees=20, step=20, max_depth=4,
        random_state=42, show_progress=False,
    )

    result = evaluate_model(model, X_test, y_test, name="Test Model", save_dir=str(tmp_path))

    assert result["name"] == "Test Model"
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert len(result["y_proba"]) == len(y_test)
    assert (tmp_path / "confusion_matrix_test_model.png").exists()


def test_plot_roc_curves_saves_file(synthetic_df, tmp_path):
    X_train_res, y_train_res, X_test, y_test = prepare_dataset(synthetic_df, random_state=42)
    model = train_random_forest(
        X_train_res, y_train_res, total_trees=20, step=20, max_depth=4,
        random_state=42, show_progress=False,
    )
    result = evaluate_model(model, X_test, y_test, name="RF", save_dir=str(tmp_path))

    plot_roc_curves([result], y_test, save_dir=str(tmp_path))
    assert (tmp_path / "roc_curve_comparison.png").exists()
