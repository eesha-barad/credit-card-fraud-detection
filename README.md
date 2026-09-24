# AI-Powered Credit Card Fraud Detection

A FinTech application that uses AI to detect fraudulent credit card transactions in real
time, explains *why* each transaction was flagged (Explainable AI), and simulates how a
bank's transaction-monitoring system would act on that score.

Built for the **AI in Banking & Finance (AIBF)** course assignment — mapping directly to:
- **Unit 4:** Fraud Prevention using AI, Risk Management using AI, Banking & Finance Process Automation
- **Unit 3:** Explainable AI, Trust and Ethics in AI, Privacy issues in Banking & Finance

---

## 1. Overview

Credit card fraud is rare (well under 1% of transactions) but costly, and a fraud model
that is accurate but unexplainable is a compliance risk for a bank. This project builds a
small, complete pipeline that:

1. Loads real, privacy-anonymized transaction data
2. Handles the severe class imbalance with SMOTE
3. Trains and compares two AI models (Logistic Regression, Random Forest)
4. Evaluates them with fraud-appropriate metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
5. Explains individual predictions with **SHAP** (Explainable AI)
6. Wraps the trained model in a **fraud alert simulator** that outputs a risk level and
   recommended action, similar to a real bank's monitoring system

## 2. Features / Modules

| Module | File | What it does |
|---|---|---|
| Data loading | `src/data_loader.py` | Loads and validates the transaction CSV |
| Preprocessing | `src/preprocessing.py` | Scales features, splits data, balances classes with SMOTE |
| Model training | `src/models.py` | Trains Logistic Regression and Random Forest, with visible training progress |
| Evaluation | `src/evaluation.py` | Computes metrics and saves confusion matrix / ROC curve plots |
| Explainability | `src/explainability.py` | SHAP-based global and per-transaction explanations |
| Fraud alert simulator | `src/fraud_alert.py` | Scores a transaction and returns a risk level + action |
| Pipeline / CLI | `src/pipeline.py` | Orchestrates the full pipeline end to end, driven by `config/config.yaml` |

## 3. Architecture

```
                     ┌────────────────────┐
                     │  creditcard.csv     │
                     │ (raw transactions)  │
                     └─────────┬──────────┘
                               │
                        data_loader.py
                               │
                     preprocessing.py
              (scale → train/test split → SMOTE)
                               │
                 ┌─────────────┴─────────────┐
                 │                            │
          models.py                    models.py
     (Logistic Regression)           (Random Forest)
                 │                            │
                 └─────────────┬──────────────┘
                               │
                       evaluation.py
              (metrics, confusion matrix, ROC curve)
                               │
                    explainability.py (SHAP)
              (global feature importance, per-case why)
                               │
                     fraud_alert.py
        (fraud_probability → risk_level → recommended_action)
                               │
                     outputs/ (plots, metrics_summary.json)
```

`src/pipeline.py` ties every stage together and is the single entry point for a full run;
each module can also be imported and used independently (see `examples/run_example.py`).

## 4. Project Structure

```
fraud-detection-project/
├── src/                    # Core application code
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── fraud_alert.py
│   └── pipeline.py         # CLI entry point
├── tests/                  # Pytest unit tests (23 tests, synthetic fixtures)
├── examples/                # Runnable demo using bundled synthetic sample data
│   ├── run_example.py
│   └── sample_data.csv
├── dataset/                 # Real dataset goes here (not committed — see dataset/README.md)
│   └── README.md
├── config/
│   └── config.yaml          # All paths, hyperparameters, and alert thresholds
├── notebooks/
│   └── fraud_detection_aibf.ipynb   # Exploratory / presentation notebook version
├── outputs/                 # Generated plots and metrics (gitignored contents)
├── requirements.txt
├── Dockerfile
├── pytest.ini
└── README.md
```

## 5. Setup

### Prerequisites
- Python 3.10+
- The dataset (see Section 6)

### Local install
```bash
git clone <your-repo-url>
cd fraud-detection-project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Dataset

This project uses the **Credit Card Fraud Detection** dataset (ULB Machine Learning Group):
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The raw file is not committed to this repository (large + licensed). See
[`dataset/README.md`](dataset/README.md) for three ways to obtain it (Kaggle notebook,
manual download, or `kagglehub`). Once obtained, place it at `dataset/creditcard.csv`, or
update the path in `config/config.yaml`.

**Don't have the dataset yet and just want to try the code?** `examples/run_example.py`
runs the entire pipeline against a small bundled synthetic file
(`examples/sample_data.csv`) with no setup required — see Section 7.

## 7. Usage

### Quick demo (no dataset download needed)
```bash
python examples/run_example.py
```

### Full pipeline (real dataset)
```bash
python -m src.pipeline --config config/config.yaml
```
This prints training progress, evaluation metrics, and SHAP output to the console, and
writes the following to `outputs/`:
- `confusion_matrix_logistic_regression.png`
- `confusion_matrix_random_forest.png`
- `roc_curve_comparison.png`
- `metrics_summary.json`

### Using the fraud alert simulator directly
```python
from src.fraud_alert import score_transaction

result = score_transaction(transaction_row, trained_model)
# {'fraud_probability': 0.82, 'risk_level': 'HIGH', 'recommended_action': 'Block transaction & alert customer'}
```

### Notebook version
Open `notebooks/fraud_detection_aibf.ipynb` for the same pipeline presented step by step
with inline plots and explanations — useful for demos/presentations or Kaggle.

### Docker
```bash
docker build -t fraud-detection .

# Runs the bundled quick demo by default:
docker run --rm fraud-detection

# To run the full pipeline against the real dataset, mount it in:
docker run --rm -v $(pwd)/dataset:/app/dataset fraud-detection -m src.pipeline
```
> Note: the Dockerfile follows a standard `python:3.11-slim` + `pip install` pattern and
> was reviewed for correctness, but could not be build-tested in the environment this
> project was authored in (no Docker/registry access there) — verify the build once on
> your own machine before submission.

## 8. Testing

```bash
pytest
```
23 unit tests cover data loading/validation, preprocessing (scaling, splitting, SMOTE
balancing), model training, evaluation metrics, the fraud alert risk bands, and — notably —
the SHAP explainability helper against **both** SHAP output formats (list-based in older
versions, 3D-array-based in SHAP 0.44+), which was the root cause of a real bug hit and
fixed during development of this project.

## 9. Configuration

All tunable settings live in `config/config.yaml`: dataset path, train/test split ratio,
random seed, Random Forest tree count/depth, and the fraud-alert risk thresholds (`low`,
`high`). Change these without touching any code.

## 10. Results & Interpretation

Because fraud is rare, **accuracy is not the metric to trust** — always predicting
"genuine" would already score ~99.8% accuracy while catching zero fraud. This project
reports Precision, Recall, F1, ROC-AUC, and PR-AUC instead, and the SHAP module shows
*which features drove each individual prediction*, so a flagged transaction can be
explained to an analyst, auditor, or customer rather than trusted blindly.

## 11. Limitations & Possible Extensions

- The dataset is PCA-anonymized (`V1`-`V28`), so feature meanings are not human-readable —
  a production system would explain fraud in terms of raw, named features.
- No live transaction stream — `fraud_alert.py` simulates scoring a single transaction at
  a time, which is how a real-time API endpoint would call it.
- Natural next steps: wrap `score_transaction()` in a FastAPI endpoint, add a Streamlit
  review dashboard for flagged transactions, log decisions for an audit trail, and monitor
  for model drift as fraud patterns evolve over time.

## 12. License

This project is submitted as academic coursework. See `LICENSE` for terms.
