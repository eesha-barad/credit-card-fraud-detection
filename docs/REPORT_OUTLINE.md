# Report Outline — AI-Powered Credit Card Fraud Detection

> **Read this first:** The assignment explicitly requires the report to be original and
> **not generated using AI tools**. What follows is a structural outline only — section
> headings, the questions each section should answer, and pointers to where in the repo
> you'll find the material to write about. The actual writing, analysis, and wording
> need to be yours. Delete this note before submitting.

---

## 1. Need Analysis / Statement of Need
Questions to answer in your own words:
- Why does fraud detection matter in banking today? (You can cite the ~0.17% fraud rate
  in the dataset, and the general cost of fraud to banks/customers — look this up and
  cite a source yourself if your instructor expects citations.)
- What's wrong with older, rule-based fraud detection approaches (fixed thresholds,
  manual review queues) that AI addresses?
- What specific gap does *this* project fill? (e.g., combining a working classifier with
  explainability, since a black-box fraud score is a compliance/trust problem for banks —
  tie this to Unit 3's Trust and Ethics in AI / Explainable AI topics.)

## 2. Technical Functionality
Describe, in your own words, what the system actually does. Suggested subsections:
- **Data handling**: what dataset, how it's loaded/validated (`src/data_loader.py`)
- **Preprocessing**: scaling, train/test split, and why SMOTE is needed for a ~0.17% fraud
  rate (`src/preprocessing.py`)
- **Models used**: Logistic Regression vs Random Forest — what each is, why both are
  included (interpretable baseline vs stronger ensemble) (`src/models.py`)
- **Evaluation approach**: why accuracy alone is misleading here, and which metrics you
  used instead (`src/evaluation.py`) — screenshot your confusion matrix / ROC curve
  outputs from `outputs/` and discuss them
- **Explainability**: how SHAP is used to explain individual predictions, and why that
  matters for a banking use case (`src/explainability.py`) — include a SHAP plot
  screenshot and explain what it shows in your own words
- **Fraud alert simulator**: how `score_transaction()` turns a probability into a risk
  band and action (`src/fraud_alert.py`)

## 3. Architecture
- Include the pipeline diagram from the README (or redraw it yourself)
- Explain the flow: raw data → preprocessing → training → evaluation → explainability →
  alerting → outputs
- Mention the project structure (src/tests/examples/config/notebooks) and why you
  organized it that way (separation of reusable code, tests, and demo/config)

## 4. Usage / Scope
- Who would use this and how? (e.g., a bank's fraud analytics team plugging real
  transaction data into the pipeline, or a data science team using it as a starting
  template)
- What is in scope (batch scoring of transactions, explainability, a simple alert
  simulation) vs out of scope (this is not a production real-time system, no live payment
  gateway integration, dataset is anonymized/historical, not live-streaming)
- How someone would actually run it (you can summarize the README's Setup/Usage sections
  here, in your own words, rather than copying them verbatim)

## 5. Impact Overview
- What is the practical benefit if a bank used something like this? (faster fraud
  detection, reduced manual review load via risk tiers, explainability supporting
  regulatory compliance and customer trust)
- What are the limitations/risks? (PCA-anonymized features limit real-world
  interpretability of the *specific* dataset used, class imbalance means results should
  be read via Precision/Recall/PR-AUC, ethical considerations around false positives
  blocking legitimate customers vs false negatives missing real fraud)
- Tie back to the syllabus topics you're demonstrating: Fraud Prevention using AI, Risk
  Management using AI, Explainable AI, Trust and Ethics in AI, Privacy issues in Banking
  and Finance (PCA anonymization protects customer identity)

---

## Practical tips for writing this
- Actually run `python -m src.pipeline --config config/config.yaml` (or the notebook)
  against the real dataset first, so your report describes *your own* results, not
  placeholder numbers.
- Use your own screenshots of the confusion matrices, ROC curve, and SHAP plots from your
  own run — these will differ slightly from any example output due to the real data.
- Write the sections in your own voice; this outline is scaffolding, not sentences to copy.
