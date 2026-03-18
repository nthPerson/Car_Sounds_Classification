"""Evaluation utilities: metrics, confusion matrices, ROC curves, reporting.

Shared across all project phases. Loads tier definitions from the
preprocessing config so class names are never hardcoded in multiple places.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ─── Project paths ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "data" / "preprocessing_config.json"
RESULTS_DIR = PROJECT_ROOT / "results"


# ─── Tier configuration ───────────────────────────────────────────────────


def get_tier_config(tier: int) -> dict:
    """Return num_classes and class_names for a taxonomy tier.

    Loads from ``data/preprocessing_config.json`` so labels are defined in
    exactly one place.

    Args:
        tier: 1, 2, or 3.

    Returns:
        Dict with keys ``num_classes`` (int) and ``class_names`` (list[str]).
    """
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    key = f"tier{tier}"
    tier_def = config["tier_definitions"][key]
    return {
        "num_classes": tier_def["num_classes"],
        "class_names": tier_def["names"],
    }


# ─── Metrics ──────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    tier: int = 2,
) -> dict:
    """Compute classification metrics for a single model/tier.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_prob: Predicted class probabilities, shape (N, num_classes).
            Required for ROC AUC (Tier 1 only).
        tier: Taxonomy tier (1, 2, or 3).

    Returns:
        Dict containing accuracy, f1_macro, f1_weighted, classification
        report (string and dict), confusion matrix (raw and normalized),
        and optionally roc_auc for Tier 1.
    """
    cfg = get_tier_config(tier)
    class_names = cfg["class_names"]
    labels = list(range(cfg["num_classes"]))

    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)

    report_str = classification_report(
        y_true, y_pred, target_names=class_names, labels=labels, digits=4, zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, labels=labels,
        digits=4, output_dict=True, zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    np.nan_to_num(cm_norm, copy=False, nan=0.0)

    metrics: dict = {
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_wt),
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "tier": tier,
        "class_names": class_names,
    }

    # ROC AUC for binary Tier 1
    if tier == 1 and y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        metrics["roc_auc"] = float(roc_auc)

    return metrics


# ─── Plotting ─────────────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str | Path,
    normalized: bool = False,
) -> None:
    """Save a confusion matrix heatmap to disk.

    Args:
        cm: Confusion matrix array.
        class_names: Display labels for each class.
        title: Plot title.
        save_path: Output PNG path.
        normalized: If True, format values as floats (.2f); else as ints.
    """
    fmt = ".2f" if normalized else "d"
    figsize = max(6, len(class_names) * 0.9)
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    """Plot and save a ROC curve (binary / Tier 1 only).

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        title: Plot title.
        save_path: Output PNG path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─── Reporting ────────────────────────────────────────────────────────────


def print_evaluation_summary(results: dict) -> None:
    """Pretty-print a summary table of evaluation results to stdout.

    Args:
        results: Dict mapping model keys (e.g. ``"rf_tier2"``) to metric
            dicts returned by :func:`compute_metrics`.
    """
    header = f"{'Model':<20s} {'Accuracy':>10s} {'F1 Macro':>10s} {'F1 Weighted':>12s}"
    print("\n" + "=" * len(header))
    print("  Evaluation Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<20s} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
            f"{m['f1_weighted']:>12.4f}"
        )
    print("=" * len(header) + "\n")
