"""
evaluation.py

Metrics and plots for evaluating fraud classifiers. Accuracy is intentionally
not the headline metric here -- for a ~0.17% fraud rate, a model predicting
"genuine" every time would score ~99.8% accuracy while catching zero fraud.
Precision, Recall, F1, ROC-AUC and PR-AUC are used instead.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, name: str = "Model", save_dir: Optional[str] = None) -> dict:
    """
    Evaluate a trained classifier on the test set.

    Returns a dict of metrics and the predicted fraud probabilities (y_proba),
    and optionally saves a confusion matrix plot to save_dir.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, y_pred, target_names=["Genuine", "Fraud"], output_dict=True
    )
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Fraud"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {name}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved confusion matrix plot to {out_path}")
    plt.close()

    return {
        "name": name,
        "classification_report": report,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def plot_roc_curves(results: list, y_test, save_dir: Optional[str] = None) -> None:
    """
    Plot ROC curves for multiple models on one chart.

    Parameters
    ----------
    results : list of dicts
        Each dict as returned by evaluate_model() (must contain 'name' and 'y_proba').
    """
    plt.figure(figsize=(6, 5))
    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        plt.plot(fpr, tpr, label=f"{result['name']} (AUC={result['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Model Comparison")
    plt.legend()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / "roc_curve_comparison.png"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved ROC curve comparison to {out_path}")
    plt.close()
