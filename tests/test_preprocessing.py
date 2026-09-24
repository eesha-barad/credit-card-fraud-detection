from src.preprocessing import (
    balance_with_smote,
    get_feature_columns,
    prepare_dataset,
    scale_amount_and_time,
    split_data,
)


def test_scale_amount_and_time_adds_columns(synthetic_df):
    scaled = scale_amount_and_time(synthetic_df)
    assert "Amount_scaled" in scaled.columns
    assert "Time_scaled" in scaled.columns
    # original df must not be mutated
    assert "Amount_scaled" not in synthetic_df.columns


def test_get_feature_columns_excludes_targets(synthetic_df):
    scaled = scale_amount_and_time(synthetic_df)
    feature_cols = get_feature_columns(scaled)
    assert "Class" not in feature_cols
    assert "Time" not in feature_cols
    assert "Amount" not in feature_cols
    assert "Time_scaled" in feature_cols
    assert "Amount_scaled" in feature_cols


def test_split_data_is_stratified(synthetic_df):
    scaled = scale_amount_and_time(synthetic_df)
    X_train, X_test, y_train, y_test = split_data(scaled, test_size=0.2, random_state=42)

    assert len(X_train) + len(X_test) == len(scaled)
    # both splits should contain at least one fraud case given stratification
    assert y_train.sum() > 0
    assert y_test.sum() > 0


def test_balance_with_smote_equalizes_classes(synthetic_df):
    scaled = scale_amount_and_time(synthetic_df)
    X_train, _, y_train, _ = split_data(scaled, test_size=0.2, random_state=42)
    X_res, y_res = balance_with_smote(X_train, y_train, random_state=42)

    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) == len(y_res)


def test_prepare_dataset_end_to_end(synthetic_df):
    X_train_res, y_train_res, X_test, y_test = prepare_dataset(
        synthetic_df, test_size=0.2, random_state=42
    )
    assert len(X_train_res) == len(y_train_res)
    assert len(X_test) == len(y_test)
    assert y_train_res.value_counts()[0] == y_train_res.value_counts()[1]
