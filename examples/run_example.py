"""
run_example.py

A minimal, runnable demonstration of the fraud detection pipeline using the
small synthetic sample_data.csv in this folder (NOT the real dataset -- this
is purely so anyone can try the project immediately without downloading
anything). Run from the project root:

    python examples/run_example.py

For real results, download the actual dataset (see dataset/README.md) and use
`python -m src.pipeline --config config/config.yaml` instead.
"""

import sys
from pathlib import Path

# Allow running this script directly from the examples/ folder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_loader import dataset_summary, load_data
from src.evaluation import evaluate_model
from src.explainability import explain_global
from src.fraud_alert import score_transaction
from src.models import train_random_forest
from src.preprocessing import prepare_dataset


def main():
    data_path = Path(__file__).parent / "sample_data.csv"

    print("Loading sample data...")
    df = load_data(str(data_path))
    print("Summary:", dataset_summary(df))

    print("\nPreparing dataset (scale -> split -> SMOTE)...")
    X_train_res, y_train_res, X_test, y_test = prepare_dataset(df, random_state=42)

    print("\nTraining a small Random Forest...")
    model = train_random_forest(
        X_train_res, y_train_res, total_trees=40, step=20, max_depth=6, random_state=42
    )

    print("\nEvaluating...")
    evaluate_model(model, X_test, y_test, name="Random Forest (example)")

    print("\nExplaining a handful of predictions with SHAP...")
    sample_X = X_test.sample(min(20, len(X_test)), random_state=42)
    shap_values_fraud, _ = explain_global(model, sample_X)
    print("SHAP values shape:", shap_values_fraud.shape)

    print("\nFraud alert simulator demo:")
    for idx, row in X_test.head(3).iterrows():
        result = score_transaction(row.to_frame().T, model)
        print(f"  Transaction {idx}: {result}")


if __name__ == "__main__":
    main()
