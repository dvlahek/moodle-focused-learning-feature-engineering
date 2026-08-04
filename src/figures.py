from __future__ import annotations

import os
from pathlib import Path
import tempfile

# Keep Matplotlib's cache inside a writable temporary directory on restricted
# systems. Users may override this variable before starting Python.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ijiet_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FEATURE_ORDER = [
    "Behavior only",
    "Behavior + weather",
    "Behavior + weather + interactions",
]


def ordered_subset(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    subset = frame[frame["Model"] == model].copy()
    subset["Feature set"] = pd.Categorical(subset["Feature set"], categories=FEATURE_ORDER, ordered=True)
    return subset.sort_values("Feature set")


def create_figures(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    regression = pd.read_csv(results_dir / "cv_regression_results.csv")
    classification = pd.read_csv(results_dir / "cv_classification_results.csv")
    rf_importance = pd.read_csv(results_dir / "model_importance_random_forest.csv").head(10).sort_values("Importance")
    xgb_importance = pd.read_csv(results_dir / "model_importance_xgboost.csv").head(10).sort_values("Importance")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    labels = ["Behavior", "Behavior +\nweather", "Full"]

    rf_reg = ordered_subset(regression, "Random Forest")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axes[0].bar(labels, rf_reg["R2_mean"], yerr=rf_reg["R2_std"], capsize=3, color="#4C78A8")
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_title("Dummy-data regression")
    axes[1].bar(labels, rf_reg["RMSE_mean"], yerr=rf_reg["RMSE_std"], capsize=3, color="#72B7B2")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Dummy-data prediction error")
    fig.tight_layout()
    fig.savefig(results_dir / "dummy_regression_performance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    rf_clf = ordered_subset(classification, "Random Forest")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(labels, rf_clf["ROC_AUC_mean"], yerr=rf_clf["ROC_AUC_std"], capsize=3, color="#F58518")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Dummy-data pass/fail classification")
    fig.tight_layout()
    fig.savefig(results_dir / "dummy_classification_auc.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, importance, title, color in (
        (axes[0], rf_importance, "Random Forest", "#4C78A8"),
        (axes[1], xgb_importance, "XGBoost", "#E45756"),
    ):
        axis.barh(importance["Feature"], importance["Importance"], color=color)
        axis.set_xlabel("Importance")
        axis.set_title(title)
    fig.suptitle("Dummy-data model-based feature importance")
    fig.tight_layout()
    fig.savefig(results_dir / "dummy_model_importance.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
