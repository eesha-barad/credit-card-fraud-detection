# Dataset

This project uses the **Credit Card Fraud Detection** dataset published by the
ULB Machine Learning Group.

- **Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** 284,807 transactions, 492 labeled as fraud (~0.17%)
- **Columns:** `V1`-`V28` (PCA-anonymized features), `Time`, `Amount`, `Class` (1 = fraud, 0 = genuine)

The raw CSV (~150 MB) is **not included in this repository** due to its size and
the dataset's license terms. You need to download it yourself:

## Option A — Kaggle Notebook
1. Open a new Kaggle Notebook
2. **Add Input** → search "Credit Card Fraud Detection" (by `mlg-ulb`) → Add
3. It will be available at `/kaggle/input/creditcardfraud/creditcard.csv`
4. Update `config/config.yaml` -> `data.path` to that path

## Option B — Local machine
1. Download `creditcard.csv` from the Kaggle link above (requires a free Kaggle account)
2. Place it in this `dataset/` folder, so the final path is `dataset/creditcard.csv`
3. This matches the default path already set in `config/config.yaml`

## Option C — kagglehub (requires Kaggle API credentials configured)
```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print(path)
```

## Quick testing without the real dataset

`examples/sample_data.csv` contains a small, fully synthetic sample with the
same column structure, so you can smoke-test the pipeline and unit tests
without downloading the full dataset. It is **not** real financial data and
should not be used to draw conclusions about model performance.
