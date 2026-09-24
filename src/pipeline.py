"""
pipeline.py

End-to-end orchestration: load data -> preprocess -> train models -> evaluate
-> explain -> demo the fraud alert simulator. Run as a script:

    python -m src.pipeline --config config/config.yaml

Outputs (plots, metrics) are written to the configured output directory.
"""

import argparse
import json
from pathlib import Path

import yaml

from src.data_loader import dataset_summary, load_data
from src.evaluation import evaluate_model, plot_roc_curves
from src.explainability import explain_global
from src.fraud_alert import score_transaction
from src.models import train_logistic_regression, train_random_forest
from src.preprocessing import prepare_dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> dict:
    data_cfg = config["data"]
    model_cfg = config["model"]
    alert_cfg = config["alert"]
    output_dir = config.get("output_dir", "outputs")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(data_cfg["path"])
    summary = dataset_summary(df)
    print("Dataset summary:", summary)

    print("\nPreprocessing (scale -> split -> SMOTE)...")
    X_train_res, y_train_res, X_test, y_test = prepare_dataset(
        df,
        test_size=model_cfg.get("test_size", 0.2),
        random_state=model_cfg.get("random_state", 42),
    )

    print("\nTraining models...")
    log_reg = train_logistic_regression(
        X_train_res, y_train_res, random_state=model_cfg.get("random_state", 42)
    )
    rf = train_random_forest(
        X_train_res,
        y_train_res,
        total_trees=model_cfg.get("rf_n_estimators", 200),
        step=model_cfg.get("rf_step", 20),
        max_depth=model_cfg.get("rf_max_depth", 12),
        random_state=model_cfg.get("random_state", 42),
    )

    print("\nEvaluating models...")
    result_lr = evaluate_model(log_reg, X_test, y_test, name="Logistic Regression", save_dir=output_dir)
    result_rf = evaluate_model(rf, X_test, y_test, name="Random Forest", save_dir=output_dir)
    plot_roc_curves([result_lr, result_rf], y_test, save_dir=output_dir)

    print("\nComputing SHAP explanations (Random Forest)...")
    sample_X = X_test.sample(min(200, len(X_test)), random_state=model_cfg.get("random_state", 42))
    shap_values_fraud, _ = explain_global(rf, sample_X)
    print("SHAP values computed, shape:", shap_values_fraud.shape)

    print("\nRunning fraud alert simulator on a few test transactions...")
    demo_samples = X_test.sample(min(5, len(X_test)), random_state=1)
    alerts = []
    for idx, row in demo_samples.iterrows():
        result = score_transaction(
            row.to_frame().T, rf, low=alert_cfg.get("low", 0.3), high=alert_cfg.get("high", 0.7)
        )
        actual = "FRAUD" if y_test.loc[idx] == 1 else "genuine"
        alerts.append({"index": int(idx), "actual": actual, **result})
        print(f"Transaction {idx} (actual: {actual}) -> {result}")

    metrics_summary = {
        "dataset_summary": summary,
        "logistic_regression": {"roc_auc": result_lr["roc_auc"], "pr_auc": result_lr["pr_auc"]},
        "random_forest": {"roc_auc": result_rf["roc_auc"], "pr_auc": result_rf["pr_auc"]},
        "sample_alerts": alerts,
    }
    metrics_path = Path(output_dir) / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved metrics summary to {metrics_path}")

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Run the fraud detection pipeline.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to the YAML config file."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
