"""Phase 2: Train classical ML baselines (Random Forest + SVM) on all 3 tiers.

Usage:
    python src/train_classical.py

Reads pre-computed 441-dim feature vectors from data/classical_ml_features/,
runs hyperparameter search via cross-validation, evaluates on the held-out
test set, and saves all models and results to models/m1_classical/ and results/.
"""

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import compute_metrics, get_tier_config, plot_confusion_matrix, plot_roc_curve, print_evaluation_summary
from features import AGG_STAT_NAMES, CLASSICAL_FEATURE_NAMES

# ─── Constants ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLASSICAL_DIR = DATA_DIR / "classical_ml_features"
MODEL_DIR = PROJECT_ROOT / "models" / "m1_classical"
RESULTS_DIR = PROJECT_ROOT / "results"

TIERS = [1, 2, 3]
RANDOM_STATE = 42
CV_FOLDS = 5
RF_N_ITER = 100  # RandomizedSearchCV iterations for RF


# ─── Helpers ──────────────────────────────────────────────────────────────


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _build_feature_names() -> list[str]:
    """Build the 441 feature names in the same order as the feature vectors."""
    return [
        f"{feat}_{stat}"
        for feat in CLASSICAL_FEATURE_NAMES
        for stat in AGG_STAT_NAMES
    ]


# ─── Data Loading ─────────────────────────────────────────────────────────


def load_data(tier: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load classical ML features and labels, merging train+val for CV.

    For Tier 3, samples with label == -1 (combined faults) are removed.

    Returns:
        (X_train_val, y_train_val, X_test, y_test)
    """
    train = np.load(CLASSICAL_DIR / "train.npz")
    val = np.load(CLASSICAL_DIR / "val.npz")
    test = np.load(CLASSICAL_DIR / "test.npz")

    label_key = f"y_tier{tier}"

    # Merge train + val
    X_tv = np.concatenate([train["X"], val["X"]], axis=0)
    y_tv = np.concatenate([train[label_key], val[label_key]], axis=0)

    X_test = test["X"]
    y_test = test[label_key]

    # Filter tier 3 excluded labels
    if tier == 3:
        mask_tv = y_tv != -1
        X_tv, y_tv = X_tv[mask_tv], y_tv[mask_tv]
        mask_test = y_test != -1
        X_test, y_test = X_test[mask_test], y_test[mask_test]

    return X_tv, y_tv, X_test, y_test


# ─── Hyperparameter Search ────────────────────────────────────────────────


def train_rf(
    X: np.ndarray, y: np.ndarray, tier: int
) -> tuple:
    """Train a Random Forest with RandomizedSearchCV.

    Returns:
        (best_model, best_params, best_cv_score, cv_results_summary)
    """
    print(f"\n  [RF Tier {tier}] RandomizedSearchCV (n_iter={RF_N_ITER}, "
          f"{CV_FOLDS}-fold)...")

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    param_distributions = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.3],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=RF_N_ITER,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X, y)

    best = search.best_estimator_
    print(f"    Best CV F1 macro: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")

    return best, search.best_params_, float(search.best_score_), {
        "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
        "std_test_score": search.cv_results_["std_test_score"].tolist(),
        "params": [
            {k: (v if not isinstance(v, np.integer) else int(v))
             for k, v in p.items()}
            for p in search.cv_results_["params"]
        ],
    }


def train_svm(
    X: np.ndarray, y: np.ndarray, tier: int
) -> tuple:
    """Train an SVM (with StandardScaler pipeline) using GridSearchCV.

    Returns:
        (best_pipeline, best_params, best_cv_score, cv_results_summary)
    """
    print(f"\n  [SVM Tier {tier}] GridSearchCV ({CV_FOLDS}-fold)...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            probability=True,
        )),
    ])

    param_grid = [
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.01, 0.1, 1, 10, 100],
            "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
        {
            "svc__kernel": ["linear"],
            "svc__C": [0.01, 0.1, 1, 10, 100],
        },
    ]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        pipe,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)

    best = search.best_estimator_
    print(f"    Best CV F1 macro: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")

    return best, search.best_params_, float(search.best_score_), {
        "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
        "std_test_score": search.cv_results_["std_test_score"].tolist(),
    }


# ─── Evaluation ───────────────────────────────────────────────────────────


def evaluate_model(
    model, X_test: np.ndarray, y_test: np.ndarray,
    model_name: str, tier: int,
) -> dict:
    """Evaluate a trained model on the test set, save plots, return metrics."""
    y_pred = model.predict(X_test)

    # Get probabilities for ROC (Tier 1) and future use
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob, tier=tier)

    # Print classification report
    print(f"\n  --- {model_name} ---")
    print(metrics["classification_report_str"])

    # Save confusion matrices
    cfg = get_tier_config(tier)
    safe_name = model_name.lower().replace(" ", "_")

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        cfg["class_names"],
        title=f"{model_name} — Confusion Matrix",
        save_path=RESULTS_DIR / f"cm_{safe_name}.png",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix_normalized"],
        cfg["class_names"],
        title=f"{model_name} — Normalized Confusion Matrix",
        save_path=RESULTS_DIR / f"cm_{safe_name}_norm.png",
        normalized=True,
    )

    # ROC curve for Tier 1
    if tier == 1 and y_prob is not None:
        plot_roc_curve(
            y_test,
            y_prob[:, 1],
            title=f"{model_name} — ROC Curve",
            save_path=RESULTS_DIR / f"roc_{safe_name}.png",
        )

    return metrics


# ─── Feature Importance ───────────────────────────────────────────────────


def feature_importance_analysis(
    rf_models: dict[int, RandomForestClassifier],
    data_by_tier: dict[int, tuple],
) -> dict:
    """Analyse RF feature importances and retrain Tier 2 on top-50 features.

    Args:
        rf_models: {tier: trained_rf_model}
        data_by_tier: {tier: (X_train_val, y_train_val, X_test, y_test)}

    Returns:
        Dict with importance arrays, top-50 indices, and reduced-model metrics.
    """
    print("\n" + "-" * 60)
    print("  Feature Importance Analysis")
    print("-" * 60)

    feature_names = _build_feature_names()
    results = {}

    # ── Per-tier importance ──────────────────────────────────────────────
    for tier in TIERS:
        if tier not in rf_models:
            continue
        importances = rf_models[tier].feature_importances_
        results[f"tier{tier}_importances"] = importances.tolist()

    # ── Tier 2 top-50 analysis (primary target) ─────────────────────────
    tier2_model = rf_models[2]
    importances = tier2_model.feature_importances_
    top50_idx = np.argsort(importances)[::-1][:50]
    top50_names = [feature_names[i] for i in top50_idx]
    top50_values = importances[top50_idx]

    print("\n  Top-10 features (Tier 2 RF):")
    for i in range(10):
        print(f"    {i + 1:>2d}. {top50_names[i]:<35s}  importance={top50_values[i]:.4f}")

    # Save top-50 bar chart
    fig, ax = plt.subplots(figsize=(10, 14))
    y_pos = np.arange(50)
    ax.barh(y_pos, top50_values[::-1], color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top50_names[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Top-50 Feature Importances — Random Forest (Tier 2)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance_tier2.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: results/feature_importance_tier2.png")

    # Save top-50 indices
    np.save(MODEL_DIR / "top50_feature_indices.npy", top50_idx)

    # ── Retrain RF on top-50 features ────────────────────────────────────
    X_tv, y_tv, X_test, y_test = data_by_tier[2]
    X_tv_50 = X_tv[:, top50_idx]
    X_test_50 = X_test[:, top50_idx]

    best_params = tier2_model.get_params()
    rf_reduced = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        class_weight=best_params["class_weight"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_reduced.fit(X_tv_50, y_tv)
    joblib.dump(rf_reduced, MODEL_DIR / "rf_tier2_top50.joblib")

    # Evaluate reduced model
    reduced_metrics = evaluate_model(
        rf_reduced, X_test_50, y_test, "RF Tier 2 (top-50)", tier=2,
    )

    results["top50_indices"] = top50_idx.tolist()
    results["top50_names"] = top50_names
    results["reduced_model_metrics"] = {
        "accuracy": reduced_metrics["accuracy"],
        "f1_macro": reduced_metrics["f1_macro"],
        "f1_weighted": reduced_metrics["f1_weighted"],
    }

    return results


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()

    print("=" * 60)
    print("  Phase 2: Classical ML Baselines")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    search_results: dict[str, dict] = {}
    rf_models: dict[int, RandomForestClassifier] = {}
    data_by_tier: dict[int, tuple] = {}

    # ── Train + evaluate for each tier ───────────────────────────────────
    for tier in TIERS:
        cfg = get_tier_config(tier)
        print(f"\n{'─' * 60}")
        print(f"  Tier {tier}: {cfg['num_classes']}-class "
              f"({', '.join(cfg['class_names'])})")
        print(f"{'─' * 60}")

        X_tv, y_tv, X_test, y_test = load_data(tier)
        data_by_tier[tier] = (X_tv, y_tv, X_test, y_test)
        print(f"  Train+Val: {X_tv.shape[0]} samples, Test: {X_test.shape[0]} samples")

        # Random Forest
        rf_model, rf_params, rf_cv, rf_cv_res = train_rf(X_tv, y_tv, tier)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, f"RF Tier {tier}", tier)
        joblib.dump(rf_model, MODEL_DIR / f"rf_tier{tier}.joblib")
        rf_models[tier] = rf_model

        all_metrics[f"rf_tier{tier}"] = {
            "accuracy": rf_metrics["accuracy"],
            "f1_macro": rf_metrics["f1_macro"],
            "f1_weighted": rf_metrics["f1_weighted"],
            "cv_f1_macro": rf_cv,
            "best_params": rf_params,
        }
        search_results[f"rf_tier{tier}"] = {
            "best_params": rf_params,
            "best_cv_f1_macro": rf_cv,
            "cv_details": rf_cv_res,
        }

        # SVM
        svm_model, svm_params, svm_cv, svm_cv_res = train_svm(X_tv, y_tv, tier)
        svm_metrics = evaluate_model(svm_model, X_test, y_test, f"SVM Tier {tier}", tier)
        joblib.dump(svm_model, MODEL_DIR / f"svm_tier{tier}.joblib")

        all_metrics[f"svm_tier{tier}"] = {
            "accuracy": svm_metrics["accuracy"],
            "f1_macro": svm_metrics["f1_macro"],
            "f1_weighted": svm_metrics["f1_weighted"],
            "cv_f1_macro": svm_cv,
            "best_params": svm_params,
        }
        search_results[f"svm_tier{tier}"] = {
            "best_params": svm_params,
            "best_cv_f1_macro": svm_cv,
            "cv_details": svm_cv_res,
        }

    # ── Feature importance analysis ──────────────────────────────────────
    importance_results = feature_importance_analysis(rf_models, data_by_tier)

    # Add reduced-model metrics to all_metrics for the summary table
    if "reduced_model_metrics" in importance_results:
        all_metrics["rf_tier2_top50"] = importance_results["reduced_model_metrics"]

    # ── Save results ─────────────────────────────────────────────────────
    with open(MODEL_DIR / "search_results.json", "w") as f:
        json.dump(search_results, f, indent=2, cls=_NumpyEncoder)

    eval_results = {
        "models": all_metrics,
        "feature_importance": {
            "top50_names": importance_results.get("top50_names", []),
            "reduced_model_metrics": importance_results.get("reduced_model_metrics", {}),
        },
    }
    with open(MODEL_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2, cls=_NumpyEncoder)

    # ── Summary CSV ──────────────────────────────────────────────────────
    import csv

    csv_path = RESULTS_DIR / "classical_ml_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "f1_macro", "f1_weighted", "cv_f1_macro"])
        for name, m in all_metrics.items():
            writer.writerow([
                name,
                f"{m['accuracy']:.4f}",
                f"{m['f1_macro']:.4f}",
                f"{m['f1_weighted']:.4f}",
                f"{m.get('cv_f1_macro', ''):.4f}" if m.get("cv_f1_macro") else "",
            ])
    print(f"\n  Saved: results/classical_ml_summary.csv")

    # ── Print summary ────────────────────────────────────────────────────
    print_evaluation_summary(all_metrics)

    elapsed = time.time() - t0
    print(f"  DONE — Phase 2 complete in {elapsed:.1f}s")
    print("=" * 60)

    # List output artifacts
    print("\nOutput artifacts:")
    for d in [MODEL_DIR, RESULTS_DIR]:
        for p in sorted(d.iterdir()):
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  {p.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
