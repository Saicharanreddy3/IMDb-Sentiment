from __future__ import annotations

import argparse
import json
import os
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from imdb_data import load_imdb
from text_features import transform_to_svd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_roc(y_true: np.ndarray, y_proba: np.ndarray, title: str, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_pr(y_true: np.ndarray, y_proba: np.ndarray, title: str, out_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_metric_bars(metrics: dict[str, Any], out_path: str) -> None:
    model_names = ["logistic_regression", "random_forest"]
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    x = np.arange(len(metric_names))
    width = 0.35

    lr_vals = [metrics[model_names[0]]["test_metrics"][m] for m in metric_names]
    rf_vals = [metrics[model_names[1]]["test_metrics"][m] for m in metric_names]

    plt.figure(figsize=(9, 4.5))
    plt.bar(x - width / 2, lr_vals, width=width, label="Logistic Regression")
    plt.bar(x + width / 2, rf_vals, width=width, label="Random Forest")
    plt.xticks(x, metric_names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Model Comparison Across Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(results_dir, "plots")
    _ensure_dir(plots_dir)

    with open(os.path.join(results_dir, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)

    settings = metrics["settings"]
    data = load_imdb(
        seed=settings.get("seed", 42),
        max_train_samples=settings.get("max_train_samples"),
        max_val_samples=settings.get("max_val_samples"),
        max_test_samples=settings.get("max_test_samples"),
    )

    feature_bundle = joblib.load(os.path.join(models_dir, "feature_extractor.joblib"))
    vectorizer = feature_bundle["vectorizer"]
    svd = feature_bundle["svd"]

    logreg = joblib.load(os.path.join(models_dir, "logreg.joblib"))
    rf = joblib.load(os.path.join(models_dir, "rf.joblib"))

    X_test = transform_to_svd(vectorizer, svd, data.test_texts)
    y_true = np.asarray(data.test_labels)

    lr_proba = logreg.predict_proba(X_test)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]

    lr_pred = (lr_proba >= 0.5).astype(int)
    rf_pred = (rf_proba >= 0.5).astype(int)

    _plot_confusion_matrix(
        y_true, lr_pred, "Logistic Regression Confusion Matrix", os.path.join(plots_dir, "cm_logreg.png")
    )
    _plot_confusion_matrix(
        y_true, rf_pred, "Random Forest Confusion Matrix", os.path.join(plots_dir, "cm_random_forest.png")
    )

    _plot_roc(y_true, lr_proba, "Logistic Regression ROC Curve", os.path.join(plots_dir, "roc_logreg.png"))
    _plot_roc(y_true, rf_proba, "Random Forest ROC Curve", os.path.join(plots_dir, "roc_random_forest.png"))

    _plot_pr(y_true, lr_proba, "Logistic Regression PR Curve", os.path.join(plots_dir, "pr_logreg.png"))
    _plot_pr(y_true, rf_proba, "Random Forest PR Curve", os.path.join(plots_dir, "pr_random_forest.png"))

    _plot_metric_bars(metrics, os.path.join(plots_dir, "model_comparison_bars.png"))

    print(f"[OK] Saved report assets to {plots_dir}")


if __name__ == "__main__":
    main()

