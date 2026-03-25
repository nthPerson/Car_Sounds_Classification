"""Phase 3: Train neural network models (M2, M5, M6) on all 3 tiers.

Usage:
    python src/train_nn.py

Trains 3 architectures × 3 taxonomy tiers = 9 float32 models.  Includes
a structured hyperparameter search on Tier 2 before final training.

Reads pre-computed features from data/features/, applies on-the-fly
SpecAugment during training, and saves models + results to models/ and
results/.
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress info/warning logs

import tensorflow as tf

# Allow memory growth so TF doesn't grab all GPU memory at once
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)
from sklearn.utils.class_weight import compute_class_weight

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from augmentation import spec_augment
from evaluate import (
    compute_metrics,
    get_tier_config,
    plot_confusion_matrix,
    plot_roc_curve,
    print_evaluation_summary,
)
from features import normalize_features
from models import MODEL_CONFIGS, get_model_config

# ─── Constants ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TIERS = [1, 2, 3]
MODEL_NAMES = ["m2", "m5", "m6"]
RANDOM_STATE = 42
MAX_EPOCHS = 100
BATCH_SIZE = 32
DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.3


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ─── Reproducibility ──────────────────────────────────────────────────────


def _set_seeds(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─── Data Loading ─────────────────────────────────────────────────────────


AUGMENTATION_VARIANT = "combined"  # Options: "none", "standard", "noise", "combined"
SKIP_HP_SEARCH = True           # Reuse existing hyperparameter search results


def load_data(
    tier: int, feature_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features, normalize, and filter Tier 3.

    Uses AUGMENTATION_VARIANT to select training data:
        "none"     — original 970 samples
        "standard" — waveform-augmented, class-balanced (3,317 samples)
        "noise"    — waveform-augmented + real-world noise injection (3,317)
        "combined" — standard + noise augmented copies together (5,664)

    Val and test sets are always unaugmented.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    feature_dir = (
        DATA_DIR / "features" / "mel_spectrograms"
        if feature_type == "mel"
        else DATA_DIR / "features" / "mfcc"
    )

    # Select training data and normalization stats based on augmentation variant
    if AUGMENTATION_VARIANT == "combined":
        train_file = feature_dir / "train_combined.npz"
        stats_path = DATA_DIR / "normalization_stats_combined.npz"
    elif AUGMENTATION_VARIANT == "noise":
        train_file = feature_dir / "train_noise_augmented.npz"
        stats_path = DATA_DIR / "normalization_stats_noise_augmented.npz"
    elif AUGMENTATION_VARIANT == "standard":
        train_file = feature_dir / "train_augmented.npz"
        stats_path = DATA_DIR / "normalization_stats_augmented.npz"
    else:  # "none"
        train_file = feature_dir / "train.npz"
        stats_path = DATA_DIR / "normalization_stats.npz"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            f"  AUGMENTATION_VARIANT={AUGMENTATION_VARIANT!r}\n"
            f"  Run the appropriate augmentation script first."
        )
    train = np.load(train_file)

    val = np.load(feature_dir / "val.npz")
    test = np.load(feature_dir / "test.npz")

    label_key = f"y_tier{tier}"

    # Load normalization stats (augmented stats if using augmented data)
    stats = np.load(stats_path)
    mean = stats[f"{feature_type}_mean"]
    std = stats[f"{feature_type}_std"]

    # Normalize
    X_train = normalize_features(train["X"], mean, std, feature_type).astype(np.float32)
    X_val = normalize_features(val["X"], mean, std, feature_type).astype(np.float32)
    X_test = normalize_features(test["X"], mean, std, feature_type).astype(np.float32)

    y_train = train[label_key].astype(np.int32)
    y_val = val[label_key].astype(np.int32)
    y_test = test[label_key].astype(np.int32)

    # Filter Tier 3 excluded labels
    if tier == 3:
        for arr_name in ["train", "val", "test"]:
            X_ref = locals()[f"X_{arr_name}"]
            y_ref = locals()[f"y_{arr_name}"]
            mask = y_ref != -1
            locals()[f"X_{arr_name}"] = X_ref[mask]
            locals()[f"y_{arr_name}"] = y_ref[mask]

        # Re-assign after filtering (locals() trick doesn't persist)
        mask_tr = y_train != -1
        X_train, y_train = X_train[mask_tr], y_train[mask_tr]
        mask_v = y_val != -1
        X_val, y_val = X_val[mask_v], y_val[mask_v]
        mask_te = y_test != -1
        X_test, y_test = X_test[mask_te], y_test[mask_te]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─── Augmented Data Sequence ──────────────────────────────────────────────


class AugmentedSequence(tf.keras.utils.Sequence):
    """Keras Sequence that applies on-the-fly SpecAugment during training.

    For mel-spectrograms (40, 92): applies spec_augment(F=7, T=10).
    For MFCCs (92, 13): transposes to (13, 92), applies spec_augment(F=2, T=10),
    transposes back.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        feature_type: str,
        augment: bool = True,
        aug_F: int | None = None,
        aug_T: int | None = None,
        seed: int = RANDOM_STATE,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.feature_type = feature_type
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.indices = np.arange(len(X))

        # Default augmentation params
        if feature_type == "mel":
            self.aug_F = aug_F if aug_F is not None else 7
            self.aug_T = aug_T if aug_T is not None else 10
        else:  # mfcc
            self.aug_F = aug_F if aug_F is not None else 2
            self.aug_T = aug_T if aug_T is not None else 10

        self._shuffle()

    def __len__(self) -> int:
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx: int):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch = self.X[batch_idx].copy()
        y_batch = self.y[batch_idx]

        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 1000 + idx)
            for i in range(len(X_batch)):
                if rng.random() < 0.5:  # 50% probability
                    if self.feature_type == "mel":
                        # Shape: (40, 92) — matches spec_augment expectation
                        X_batch[i] = spec_augment(
                            X_batch[i], F=self.aug_F, T=self.aug_T, rng=rng,
                        )
                    else:
                        # MFCC shape: (92, 13) → transpose to (13, 92) for augment
                        transposed = X_batch[i].T
                        transposed = spec_augment(
                            transposed, F=self.aug_F, T=self.aug_T, rng=rng,
                        )
                        X_batch[i] = transposed.T

        # Add channel dim for mel-spectrograms
        if self.feature_type == "mel":
            X_batch = X_batch[..., np.newaxis]

        return X_batch, y_batch

    def on_epoch_end(self):
        self.epoch += 1
        self._shuffle()

    def _shuffle(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(self.indices)


# ─── Class Weights ────────────────────────────────────────────────────────


def compute_class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    """Compute balanced class weights for model.fit()."""
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y,
    )
    return {i: float(w) for i, w in enumerate(weights)}


# ─── Training ─────────────────────────────────────────────────────────────


def train_single_model(
    model_name: str,
    tier: int,
    lr: float = DEFAULT_LR,
    dropout_rate: float = DEFAULT_DROPOUT,
    aug_F: int | None = None,
    aug_T: int | None = None,
    augment: bool = True,
    save_model: bool = True,
    verbose: int = 0,
) -> tuple:
    """Train a single model on a single tier.

    Returns:
        (model, history_dict, best_val_accuracy, training_seconds)
    """
    _set_seeds()

    cfg = get_model_config(model_name)
    tier_cfg = get_tier_config(tier)
    num_classes = tier_cfg["num_classes"]

    # Load data
    X_train, y_train, X_val, y_val, _, _ = load_data(tier, cfg["feature_type"])

    # Build model
    builder_kwargs = {"input_shape": cfg["input_shape"], "num_classes": num_classes}
    if model_name in ("m2", "m5"):
        builder_kwargs["dropout_rate"] = dropout_rate
    model = cfg["builder"](**builder_kwargs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Class weights
    class_weights = compute_class_weights(y_train, num_classes)

    # Training data sequence
    train_seq = AugmentedSequence(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        feature_type=cfg["feature_type"],
        augment=augment,
        aug_F=aug_F,
        aug_T=aug_T,
    )

    # Validation data (no augmentation, add channel dim for mel)
    X_val_input = X_val[..., np.newaxis] if cfg["feature_type"] == "mel" else X_val

    # Model save path
    model_dir = MODELS_DIR / cfg["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = str(model_dir / f"tier{tier}_best.h5")
    log_path = str(model_dir / f"tier{tier}_training_log.csv")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=verbose,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=verbose,
        ),
        tf.keras.callbacks.CSVLogger(log_path, append=False),
    ]

    if save_model:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=verbose,
            )
        )

    t0 = time.time()
    history = model.fit(
        train_seq,
        validation_data=(X_val_input, y_val),
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    best_val_acc = max(history.history["val_accuracy"])

    return model, history.history, best_val_acc, elapsed


# ─── Training Curves ──────────────────────────────────────────────────────


def plot_training_curves(
    history: dict, model_name: str, tier: int, save_dir: Path,
) -> None:
    """Plot loss and accuracy training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["loss"]) + 1)

    ax1.plot(epochs, history["loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name.upper()} Tier {tier} — Loss")
    ax1.legend()

    ax2.plot(epochs, history["accuracy"], label="Train")
    ax2.plot(epochs, history["val_accuracy"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name.upper()} Tier {tier} — Accuracy")
    ax2.legend()

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_dir / f"training_curves_{model_name}_tier{tier}.png", dpi=150)
    plt.close(fig)


# ─── Hyperparameter Search ────────────────────────────────────────────────


def run_hyperparameter_search() -> dict:
    """Structured manual search on Tier 2 for all architectures.

    Sweeps: learning rate → dropout → augmentation intensity.
    Returns dict mapping model_name to best config.
    """
    print("\n" + "=" * 60)
    print("  Hyperparameter Search (Tier 2)")
    print("=" * 60)

    search_results = {}

    for model_name in MODEL_NAMES:
        cfg = get_model_config(model_name)
        print(f"\n{'─' * 60}")
        print(f"  {cfg['display_name']} — Tier 2 hyperparameter search")
        print(f"{'─' * 60}")

        best_lr = DEFAULT_LR
        best_dropout = DEFAULT_DROPOUT
        best_aug_config = {"augment": True, "aug_F": None, "aug_T": None}
        model_search = {"lr_sweep": [], "dropout_sweep": [], "aug_sweep": []}

        # ── 1. Learning rate sweep ──
        print("\n  LR sweep: ", end="", flush=True)
        lr_results = []
        for lr in [0.0005, 0.001, 0.002]:
            _, _, val_acc, elapsed = train_single_model(
                model_name, tier=2, lr=lr, save_model=False, verbose=0,
            )
            lr_results.append({"lr": lr, "val_accuracy": val_acc, "time": elapsed})
            print(f"lr={lr} → {val_acc:.4f}  ", end="", flush=True)
        print()

        best_lr_entry = max(lr_results, key=lambda x: x["val_accuracy"])
        best_lr = best_lr_entry["lr"]
        model_search["lr_sweep"] = lr_results
        print(f"  → Best LR: {best_lr}")

        # ── 2. Dropout sweep ──
        if model_name in ("m2", "m5"):  # DS-CNN doesn't have dropout param
            print("  Dropout sweep: ", end="", flush=True)
            drop_results = []
            for dr in [0.2, 0.3, 0.5]:
                _, _, val_acc, elapsed = train_single_model(
                    model_name, tier=2, lr=best_lr, dropout_rate=dr,
                    save_model=False, verbose=0,
                )
                drop_results.append({"dropout": dr, "val_accuracy": val_acc, "time": elapsed})
                print(f"dr={dr} → {val_acc:.4f}  ", end="", flush=True)
            print()

            best_drop_entry = max(drop_results, key=lambda x: x["val_accuracy"])
            best_dropout = best_drop_entry["dropout"]
            model_search["dropout_sweep"] = drop_results
            print(f"  → Best Dropout: {best_dropout}")

        # ── 3. Augmentation sweep ──
        print("  Augmentation sweep: ", end="", flush=True)
        aug_configs = [
            {"label": "none", "augment": False, "aug_F": None, "aug_T": None},
            {"label": "default", "augment": True, "aug_F": None, "aug_T": None},
            {"label": "aggressive", "augment": True, "aug_F": 10, "aug_T": 15},
        ]
        aug_results = []
        for ac in aug_configs:
            _, _, val_acc, elapsed = train_single_model(
                model_name, tier=2, lr=best_lr, dropout_rate=best_dropout,
                augment=ac["augment"], aug_F=ac["aug_F"], aug_T=ac["aug_T"],
                save_model=False, verbose=0,
            )
            aug_results.append({
                "config": ac["label"], "val_accuracy": val_acc, "time": elapsed,
            })
            print(f"{ac['label']} → {val_acc:.4f}  ", end="", flush=True)
        print()

        best_aug_entry = max(aug_results, key=lambda x: x["val_accuracy"])
        best_aug_label = best_aug_entry["config"]
        best_aug_config = next(ac for ac in aug_configs if ac["label"] == best_aug_label)
        model_search["aug_sweep"] = aug_results
        print(f"  → Best Augmentation: {best_aug_label}")

        search_results[model_name] = {
            "best_lr": best_lr,
            "best_dropout": best_dropout,
            "best_augmentation": best_aug_label,
            "best_aug_F": best_aug_config.get("aug_F"),
            "best_aug_T": best_aug_config.get("aug_T"),
            "best_augment": best_aug_config["augment"],
            "sweeps": model_search,
        }

        print(f"\n  Best config for {cfg['display_name']}:")
        print(f"    LR={best_lr}, Dropout={best_dropout}, Aug={best_aug_label}")

    return search_results


# ─── Evaluation ───────────────────────────────────────────────────────────


def evaluate_model_on_test(
    model: tf.keras.Model,
    model_name: str,
    tier: int,
) -> dict:
    """Evaluate a trained model on the test set, save plots."""
    cfg = get_model_config(model_name)
    tier_cfg = get_tier_config(tier)

    _, _, _, _, X_test, y_test = load_data(tier, cfg["feature_type"])

    # Add channel dim for mel
    X_test_input = X_test[..., np.newaxis] if cfg["feature_type"] == "mel" else X_test

    y_prob = model.predict(X_test_input, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob, tier=tier)

    # Save confusion matrices
    safe_name = f"{model_name}_tier_{tier}"
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        tier_cfg["class_names"],
        title=f"{cfg['display_name']} Tier {tier} — Confusion Matrix",
        save_path=RESULTS_DIR / f"cm_{safe_name}.png",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix_normalized"],
        tier_cfg["class_names"],
        title=f"{cfg['display_name']} Tier {tier} — Normalized",
        save_path=RESULTS_DIR / f"cm_{safe_name}_norm.png",
        normalized=True,
    )

    # ROC for Tier 1
    if tier == 1 and y_prob is not None:
        plot_roc_curve(
            y_test, y_prob[:, 1],
            title=f"{cfg['display_name']} Tier 1 — ROC Curve",
            save_path=RESULTS_DIR / f"roc_{model_name}_tier_1.png",
        )

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    t0_total = time.time()
    _set_seeds()

    print("=" * 60)
    print("  Phase 3: Neural Network Training")
    print("=" * 60)
    print(f"\n  Augmentation variant: {AUGMENTATION_VARIANT}")
    print(f"  Skip HP search: {SKIP_HP_SEARCH}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Hyperparameter search on Tier 2 ─────────────────────────
    hp_path = MODELS_DIR / "hyperparameter_search.json"
    if SKIP_HP_SEARCH and hp_path.exists():
        print("\n  Skipping hyperparameter search — loading existing results...")
        with open(hp_path) as f:
            search_results = json.load(f)
        for model_name in MODEL_NAMES:
            sr = search_results[model_name]
            cfg = get_model_config(model_name)
            print(f"    {cfg['display_name']}: LR={sr['best_lr']}, "
                  f"Dropout={sr['best_dropout']}, Aug={sr['best_augmentation']}")
    else:
        search_results = run_hyperparameter_search()
        with open(hp_path, "w") as f:
            json.dump(search_results, f, indent=2, cls=_NumpyEncoder)
        print(f"\n  Saved: models/hyperparameter_search.json")

    # ── Step 2: Train all 9 models with best hyperparameters ─────────────
    print("\n" + "=" * 60)
    print("  Training All Models (3 architectures × 3 tiers)")
    print("=" * 60)

    all_metrics = {}
    all_histories = {}

    for model_name in MODEL_NAMES:
        cfg = get_model_config(model_name)
        sr = search_results[model_name]

        best_lr = sr["best_lr"]
        best_dropout = sr["best_dropout"]
        best_augment = sr["best_augment"]
        best_aug_F = sr["best_aug_F"]
        best_aug_T = sr["best_aug_T"]

        print(f"\n{'─' * 60}")
        print(f"  {cfg['display_name']} — LR={best_lr}, Dropout={best_dropout}, "
              f"Aug={sr['best_augmentation']}")
        print(f"{'─' * 60}")

        for tier in TIERS:
            tier_cfg = get_tier_config(tier)
            print(f"\n  Training {cfg['display_name']} Tier {tier} "
                  f"({tier_cfg['num_classes']}-class)...", flush=True)

            model, history, best_val_acc, elapsed = train_single_model(
                model_name, tier,
                lr=best_lr, dropout_rate=best_dropout,
                augment=best_augment, aug_F=best_aug_F, aug_T=best_aug_T,
                save_model=True, verbose=0,
            )

            n_epochs = len(history["loss"])
            print(f"    Epochs: {n_epochs}, Best Val Acc: {best_val_acc:.4f}, "
                  f"Time: {elapsed:.1f}s")

            # Plot training curves
            plot_training_curves(history, model_name, tier, RESULTS_DIR)

            # Evaluate on test set
            # Load the best checkpoint for evaluation
            model_dir = MODELS_DIR / cfg["model_dir"]
            best_model = tf.keras.models.load_model(
                str(model_dir / f"tier{tier}_best.h5")
            )
            metrics = evaluate_model_on_test(best_model, model_name, tier)

            key = f"{model_name}_tier{tier}"
            all_metrics[key] = {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "best_val_accuracy": best_val_acc,
                "epochs_trained": n_epochs,
                "training_time_s": elapsed,
            }
            if "roc_auc" in metrics:
                all_metrics[key]["roc_auc"] = metrics["roc_auc"]

            all_histories[key] = history

            print(f"    Test — Acc: {metrics['accuracy']:.4f}, "
                  f"F1 Macro: {metrics['f1_macro']:.4f}")
            print(metrics["classification_report_str"])

        # Save per-architecture evaluation results
        model_dir = MODELS_DIR / cfg["model_dir"]
        arch_results = {
            k: v for k, v in all_metrics.items() if k.startswith(model_name)
        }
        with open(model_dir / "evaluation_results.json", "w") as f:
            json.dump(arch_results, f, indent=2, cls=_NumpyEncoder)

    # ── Step 3: Save summary ─────────────────────────────────────────────

    # Summary CSV
    csv_path = RESULTS_DIR / "nn_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "tier", "accuracy", "f1_macro", "f1_weighted",
            "best_val_accuracy", "epochs_trained",
        ])
        for key, m in all_metrics.items():
            parts = key.split("_tier")
            writer.writerow([
                parts[0], parts[1],
                f"{m['accuracy']:.4f}",
                f"{m['f1_macro']:.4f}",
                f"{m['f1_weighted']:.4f}",
                f"{m['best_val_accuracy']:.4f}",
                m["epochs_trained"],
            ])
    print(f"\n  Saved: results/nn_summary.csv")

    # Full results JSON
    full_results = {
        "models": all_metrics,
        "hyperparameter_search": search_results,
    }
    with open(RESULTS_DIR / "nn_evaluation_results.json", "w") as f:
        json.dump(full_results, f, indent=2, cls=_NumpyEncoder)

    # ── Step 4: Print summary ────────────────────────────────────────────
    print_evaluation_summary(all_metrics)

    elapsed_total = time.time() - t0_total
    print(f"  DONE — Phase 3 complete in {elapsed_total:.1f}s "
          f"({elapsed_total / 60:.1f} min)")
    print("=" * 60)

    # List output artifacts
    print("\nOutput artifacts:")
    for d in [MODELS_DIR / "m2_cnn2d_float32",
              MODELS_DIR / "m5_cnn1d_int8_qat",
              MODELS_DIR / "m6_dscnn_int8_qat",
              RESULTS_DIR]:
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.name != ".gitkeep":
                size_kb = p.stat().st_size / 1024
                print(f"  {p.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
