from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import shap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default="..")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")

    os.makedirs(results_dir, exist_ok=True)

    rf = joblib.load(os.path.join(models_dir, "rf.joblib"))
    data_bundle = joblib.load(os.path.join(models_dir, "shap_background_data.joblib")) if os.path.exists(
        os.path.join(models_dir, "shap_background_data.joblib")
    ) else None

    if data_bundle is None:
        # Nothing saved; exit quietly with a hint.
        print("No SHAP background data found. Skipping SHAP explanation.")
        return

    X_eval = data_bundle["X_eval"]

    # Subsample for speed.
    if args.max_samples and args.max_samples < X_eval.shape[0]:
        rng = np.random.default_rng(42)
        idx = rng.choice(X_eval.shape[0], size=args.max_samples, replace=False)
        X_eval = X_eval[idx]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer(X_eval)

    shap_values_path = os.path.join(results_dir, "shap_values.npy")
    np.save(shap_values_path, shap_values.values)

    # Summary plot (optional for the report).
    try:
        shap.summary_plot(
            shap_values.values[:, :, 1] if shap_values.values.ndim == 3 else shap_values.values,
            X_eval,
            show=False,
        )
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "shap_summary.png"), dpi=150)
        plt.close()
    except Exception as exc:  # pragma: no cover - plotting is best-effort.
        print(f"Could not generate SHAP summary plot: {exc}")


if __name__ == "__main__":
    main()

