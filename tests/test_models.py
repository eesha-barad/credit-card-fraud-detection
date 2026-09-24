from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models import train_logistic_regression, train_random_forest
from src.preprocessing import prepare_dataset


def test_train_logistic_regression_returns_fitted_model(synthetic_df):
    X_train_res, y_train_res, _, _ = prepare_dataset(synthetic_df, random_state=42)
    model = train_logistic_regression(X_train_res, y_train_res, random_state=42, verbose=False)

    assert isinstance(model, LogisticRegression)
    # a fitted model should be able to predict without raising
    preds = model.predict(X_train_res.iloc[:5])
    assert len(preds) == 5


def test_train_random_forest_returns_fitted_model_with_correct_tree_count(synthetic_df):
    X_train_res, y_train_res, _, _ = prepare_dataset(synthetic_df, random_state=42)
    model = train_random_forest(
        X_train_res,
        y_train_res,
        total_trees=40,
        step=20,
        max_depth=6,
        random_state=42,
        show_progress=False,
    )

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 40
    preds = model.predict(X_train_res.iloc[:5])
    assert len(preds) == 5
