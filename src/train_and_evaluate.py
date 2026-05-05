from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from imdb_data import load_imdb
from metrics import compute_metrics
from text_features import (
    extract_top_terms_for_svd_components,
    fit_tfidf_svd,
    transform_to_svd,
)


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    tune: bool,
    cv_folds: int,
    seed: int,
) -> tuple[LogisticRegression, dict[str, Any]]:
    base = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=seed,
    )

    if not tune:
        model = base.fit(X_train, y_train)
        return model, {"tuned": False}

    param_grid = {
        "C": [0.01, 0.1, 1.0, 3.0, 10.0],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, {"tuned": True, "best_params": search.best_params_}


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    tune: bool,
    cv_folds: int,
    seed: int,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    base = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )

    if not tune:
        model = base.fit(X_train, y_train)
        return model, {"tuned": False}

    # Small randomized search to keep runtime reasonable.
    param_distributions = {
        "n_estimators": [300, 500, 700],
        "max_depth": [None, 12, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        scoring="f1",
        cv=cv,
        n_iter=12,
        n_jobs=-1,
        verbose=0,
        random_state=seed,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, {"tuned": True, "best_params": search.best_params_}


def _evaluate_binary_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    y_proba = model.predict_proba(X_test)[:, 1]
    # Use probabilities for metrics as required.
    m = compute_metrics(y_test, y_proba=y_proba)
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="..", help="Project root for outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters with cross-validation.")
    parser.add_argument("--no-tune", dest="tune", action="store_false")
    parser.set_defaults(tune=True)
    parser.add_argument("--explain", action="store_true", help="Run SHAP explanation after training.")
    parser.add_argument("--no-explain", dest="explain", action="store_false")
    parser.set_defaults(explain=False)
    parser.add_argument("--fast", action="store_true", help="Use small dataset + smaller feature sizes.")

    # Feature engineering.
    parser.add_argument("--max_features", type=int, default=40000, help="TF-IDF max vocabulary size.")
    parser.add_argument("--svd_components", type=int, default=300, help="SVD latent components.")
    parser.add_argument("--ngram_high", type=int, default=2, help="TF-IDF ngram high (range=1..ngram_high).")

    # Subsampling for fast mode.
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    args = parser.parse_args()

    # Outputs.
    project_root = os.path.abspath(args.out_dir)
    results_dir = os.path.join(project_root, "results")
    models_dir = os.path.join(project_root, "models")
    _ensure_dirs(results_dir, models_dir)

    # Fast mode: shrink both samples and feature sizes.
    max_train_samples = args.max_train_samples
    max_val_samples = args.max_val_samples
    max_test_samples = args.max_test_samples
    max_features = args.max_features
    svd_components = args.svd_components

    if args.fast:
        max_train_samples = max_train_samples or 4000
        max_val_samples = max_val_samples or 2000
        max_test_samples = max_test_samples or 4000
        max_features = 15000
        svd_components = min(svd_components, 150)

    data = load_imdb(
        seed=args.seed,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )

    vectorizer, svd = fit_tfidf_svd(
        data.train_texts,
        max_features=max_features,
        ngram_range=(1, args.ngram_high),
        svd_components=svd_components,
        seed=args.seed,
    )

    X_train = transform_to_svd(vectorizer, svd, data.train_texts)
    X_val = transform_to_svd(vectorizer, svd, data.val_texts)
    X_test = transform_to_svd(vectorizer, svd, data.test_texts)

    # Save feature extractor for later explanations.
    joblib.dump(
        {"vectorizer": vectorizer, "svd": svd},
        os.path.join(models_dir, "feature_extractor.joblib"),
    )

    # Human-interpretable mapping of latent components.
    top_terms = extract_top_terms_for_svd_components(
        vectorizer,
        svd,
        top_k=8,
    )
    with open(os.path.join(results_dir, "svd_component_terms.json"), "w", encoding="utf-8") as f:
        json.dump(top_terms, f, indent=2)

    # Train models.
    logreg, logreg_info = train_logreg(
        X_train,
        data.train_labels,
        tune=args.tune,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )
    rf, rf_info = train_rf(
        X_train,
        data.train_labels,
        tune=args.tune,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )

    # Evaluate on test split (required for reporting).
    logreg_metrics = _evaluate_binary_classifier(logreg, X_test, data.test_labels)
    rf_metrics = _evaluate_binary_classifier(rf, X_test, data.test_labels)

    metrics_out = {
        "settings": {
            "seed": args.seed,
            "tune": args.tune,
            "fast": args.fast,
            "max_features": max_features,
            "svd_components": svd_components,
            "ngram_range": (1, args.ngram_high),
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "max_test_samples": max_test_samples,
        },
        "logistic_regression": {
            "training_info": logreg_info,
            "test_metrics": logreg_metrics,
        },
        "random_forest": {
            "training_info": rf_info,
            "test_metrics": rf_metrics,
        },
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"[OK] Saved metrics to {metrics_path}")

    # Optional explainability.
    joblib.dump(logreg, os.path.join(models_dir, "logreg.joblib"))
    joblib.dump(rf, os.path.join(models_dir, "rf.joblib"))
    joblib.dump(
        {
            "X_background": X_train[: min(500, len(X_train))],
            "X_eval": X_test[: min(1000, len(X_test))],
        },
        os.path.join(models_dir, "shap_background_data.joblib"),
    )

    if args.explain:
        # Lazy import to avoid hard failure if shap isn't installed.
        import subprocess

        script_path = os.path.join(project_root, "src", "explain_xgb.py")
        subprocess.check_call(
            ["python3", script_path, "--project_root", project_root, "--max_samples", "200"]
        )


if __name__ == "__main__":
    main()

