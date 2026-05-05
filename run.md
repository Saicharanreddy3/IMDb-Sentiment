# Run Guide

This file gives the simplest commands to run the project and understand the results.

## 1) Setup (one-time)

```bash
cd /Users/anil/Documents/old-ucm-stuff/ucm-spring-2026/nlp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Quick complete run (recommended)

This is the fastest full pipeline and generates all key outputs.

```bash
cd /Users/anil/Documents/old-ucm-stuff/ucm-spring-2026/nlp
source .venv/bin/activate
python3 src/train_and_evaluate.py --fast --no-tune --explain --out_dir .
```

### What this run does

- Downloads/loads IMDb dataset
- Builds TF-IDF + SVD features
- Trains 2 models:
  - Logistic Regression
  - Random Forest (advanced ensemble model)
- Evaluates with multiple metrics
- Produces SHAP explainability artifacts

## 3) Optional stronger run (slower)

Includes hyperparameter tuning (cross-validation), so it takes longer.

```bash
cd /Users/anil/Documents/old-ucm-stuff/ucm-spring-2026/nlp
source .venv/bin/activate
python3 src/train_and_evaluate.py --fast --tune --explain --out_dir .
```

## 4) Where to check results

After a successful run, look in:

- `results/metrics.json`
- `results/svd_component_terms.json`
- `results/shap_summary.png`
- `results/shap_values.npy`
- `models/logreg.joblib`
- `models/rf.joblib`

## 5) What to expect in `results/metrics.json`

The JSON contains:

- `settings`: run configuration (fast/tune/sample sizes)
- `logistic_regression.test_metrics`
- `random_forest.test_metrics`

Each model has these metrics:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`

Typical expectation:

- Logistic Regression usually performs better on this TF-IDF setup.
- Random Forest acts as the advanced ensemble comparison model.

## 6) Quick rerun note

If you run again, files in `results/` are overwritten with the latest run outputs.
