# NLP ML Capstone (IMDb Sentiment)

This project implements an end-to-end NLP text classification pipeline on the **IMDb** dataset:
baseline **Logistic Regression** vs **Random Forest** (advanced ensemble model), with TF‑IDF feature engineering + dimensionality reduction (Truncated SVD).

## Team Information
- Sai Charan Reddy Chitla
- Sai Kumar Reddy Uppula
- Dileep Kumar Gaddam

## What you’ll get

- Reproducible training scripts under `src/`
- Saved models under `models/`
- Saved metrics under `results/`
- SHAP explainability output (optional) for the ensemble model under `results/`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run experiments

### Fast sanity run

```bash
python3 src/train_and_evaluate.py --fast
```

### Full run (uses the full training split)

```bash
python3 src/train_and_evaluate.py
```

## Important outputs

- `results/metrics.json`
- `results/plots/` (if enabled by the run)
- `results/shap_values.npy` (from explainability script, if enabled)

## Report (IEEE) checklist

Your proposal/final report should include:

- Introduction
- Objectives
- Approaches/Methods
- Workflow
- Datasets
- Parameters
- Evaluation & Discussion (multiple metrics + comparison)
- Conclusion
- References

## Notes for grading alignment

- Dataset size: IMDb has **50,000** labeled reviews total.
- Advanced model: Random Forest (ensemble method).
- Multiple metrics: accuracy, precision, recall, F1, ROC-AUC.
- Comparison: Logistic Regression vs Random Forest.
- Explainability: SHAP on the ensemble model.