import pytest

from src.data_loader import dataset_summary, load_data


def test_load_data_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_data(str(missing_path))


def test_load_data_missing_columns_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("A,B,C\n1,2,3\n")
    with pytest.raises(ValueError):
        load_data(str(bad_csv))


def test_load_data_valid_file(tmp_path, synthetic_df):
    csv_path = tmp_path / "creditcard.csv"
    synthetic_df.to_csv(csv_path, index=False)

    df = load_data(str(csv_path))
    assert df.shape == synthetic_df.shape
    assert "Class" in df.columns


def test_dataset_summary(synthetic_df):
    summary = dataset_summary(synthetic_df)
    assert summary["rows"] == len(synthetic_df)
    assert summary["fraud_count"] == int(synthetic_df["Class"].sum())
    assert summary["genuine_count"] + summary["fraud_count"] == summary["rows"]
    assert 0 <= summary["fraud_percentage"] <= 100
