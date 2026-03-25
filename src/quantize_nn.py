"""Phase 4: Quantize neural network models to int8 (PTQ & QAT).

Usage:
    python src/quantize_nn.py

Converts 3 float32 architectures x 3 tiers x 2 methods (PTQ + QAT) = 18
int8 TFLite models. Evaluates all quantized models, computes quantization
impact analysis, and generates visualizations.

Retrains M5/M6 on Condition B if needed (their best training condition).
M2 stays on Condition D.

Reads float32 .h5 models from models/, saves .tflite files alongside them.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # tfmot requires tf_keras (legacy)

import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

import tensorflow_model_optimization as tfmot
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import recall_score
from sklearn.utils.class_weight import compute_class_weight

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import (
    compute_metrics,
    get_tier_config,
    plot_confusion_matrix,
    plot_roc_curve,
    print_evaluation_summary,
)
from models import MODEL_CONFIGS, get_model_config

# ─── Constants ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TIERS = [1, 2, 3]
MODEL_NAMES = ["m2", "m5", "m6"]
RANDOM_STATE = 42
BATCH_SIZE = 32

# ─── Configuration ────────────────────────────────────────────────────────

# Best training condition per architecture (from quantization_readiness_analysis.md)
MODEL_CONDITIONS = {
    "m2": "combined",   # Condition D -- best for M2
    "m5": "standard",   # Condition B -- best for M5
    "m6": "standard",   # Condition B -- best for M6
}

# QAT fine-tuning settings (spec: 10-100x lower LR, ~20-30 epochs)
QAT_EPOCHS = 30
QAT_LR_FACTOR = 0.1       # multiply original LR by this
QAT_PATIENCE = 10          # early stopping patience

# PTQ calibration
REPRESENTATIVE_SAMPLES = 200  # stratified from training set

# TFLite output paths
TFLITE_DIRS = {
    "m2_ptq": "m3_cnn2d_int8_ptq",
    "m2_qat": "m4_cnn2d_int8_qat",
    "m5_ptq": "m5_cnn1d_int8_qat",
    "m5_qat": "m5_cnn1d_int8_qat",
    "m6_ptq": "m6_dscnn_int8_qat",
    "m6_qat": "m6_dscnn_int8_qat",
}


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


# ─── Reproducibility ─────────────────────────────────────────────────────


def _set_seeds(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─── Data Loading ─────────────────────────────────────────────────────────


def load_data(
    tier: int, feature_type: str, condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load features, normalize, and filter Tier 3.

    Args:
        tier: Taxonomy tier (1, 2, or 3).
        feature_type: "mel" or "mfcc".
        condition: "none", "standard", "noise", or "combined".

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test) all float32.
    """
    feat_dir_name = "mel_spectrograms" if feature_type == "mel" else "mfcc"
    feat_dir = DATA_DIR / "features" / feat_dir_name

    # Select training file and normalization stats
    suffix_map = {
        "none": ("train.npz", "normalization_stats.npz"),
        "standard": ("train_augmented.npz", "normalization_stats_augmented.npz"),
        "noise": ("train_noise_augmented.npz", "normalization_stats_noise_augmented.npz"),
        "combined": ("train_combined.npz", "normalization_stats_combined.npz"),
    }
    train_file, stats_file = suffix_map[condition]

    # Load features (allow_pickle needed for file_paths object array)
    train_data = np.load(feat_dir / train_file, allow_pickle=True)
    val_data = np.load(feat_dir / "val.npz", allow_pickle=True)
    test_data = np.load(feat_dir / "test.npz", allow_pickle=True)

    X_train = train_data["X"].astype(np.float32)
    y_train = train_data[f"y_tier{tier}"]
    X_val = val_data["X"].astype(np.float32)
    y_val = val_data[f"y_tier{tier}"]
    X_test = test_data["X"].astype(np.float32)
    y_test = test_data[f"y_tier{tier}"]

    # Load normalization stats
    stats = np.load(DATA_DIR / stats_file)
    stat_key = "mel" if feature_type == "mel" else "mfcc"
    mean = stats[f"{stat_key}_mean"].astype(np.float32)
    std = stats[f"{stat_key}_std"].astype(np.float32)

    # Apply z-score normalization
    if feature_type == "mel":
        # Shape: (N, 40, 92), mean/std shape: (40,)
        X_train = (X_train - mean[:, np.newaxis]) / std[:, np.newaxis]
        X_val = (X_val - mean[:, np.newaxis]) / std[:, np.newaxis]
        X_test = (X_test - mean[:, np.newaxis]) / std[:, np.newaxis]
    else:
        # Shape: (N, 92, 13), mean/std shape: (13,)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

    # Filter Tier 3 excluded labels
    if tier == 3:
        for name, (X, y) in [
            ("train", (X_train, y_train)),
            ("val", (X_val, y_val)),
            ("test", (X_test, y_test)),
        ]:
            mask = y != -1
            if name == "train":
                X_train, y_train = X[mask], y[mask]
            elif name == "val":
                X_val, y_val = X[mask], y[mask]
            else:
                X_test, y_test = X[mask], y[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─── Retrain Float32 Models ──────────────────────────────────────────────


def retrain_model(
    model_name: str, tier: int, condition: str, hp_config: dict,
) -> tf.keras.Model:
    """Train a float32 model on the specified condition and save it.

    Returns the trained Keras model.
    """
    _set_seeds()

    cfg = get_model_config(model_name)
    tier_cfg = get_tier_config(tier)
    num_classes = tier_cfg["num_classes"]

    X_train, y_train, X_val, y_val, _, _ = load_data(
        tier, cfg["feature_type"], condition,
    )

    # Build model
    builder_kwargs = {"input_shape": cfg["input_shape"], "num_classes": num_classes}
    if model_name in ("m2", "m5"):
        builder_kwargs["dropout_rate"] = hp_config.get("best_dropout", 0.3)
    model = cfg["builder"](**builder_kwargs)

    lr = hp_config.get("best_lr", 0.001)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Class weights
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_train,
    )
    class_weights = {i: float(w) for i, w in enumerate(weights)}

    # Prepare data (add channel dim for mel)
    X_val_input = X_val[..., np.newaxis] if cfg["feature_type"] == "mel" else X_val
    X_train_input = X_train[..., np.newaxis] if cfg["feature_type"] == "mel" else X_train

    # Save path
    model_dir = MODELS_DIR / cfg["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = str(model_dir / f"tier{tier}_best.h5")

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
    ]

    model.fit(
        X_train_input, y_train,
        validation_data=(X_val_input, y_val),
        epochs=100,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    # Reload best checkpoint
    model = tf.keras.models.load_model(checkpoint_path)
    return model


# ─── Representative Dataset ──────────────────────────────────────────────


def create_representative_dataset(
    X_train: np.ndarray, y_train: np.ndarray, feature_type: str,
    num_samples: int = REPRESENTATIVE_SAMPLES,
):
    """Create a stratified representative dataset generator for PTQ.

    Returns a callable generator that yields [sample] one at a time.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    classes = np.unique(y_train)

    # Stratified sampling
    selected_indices = []
    samples_per_class = max(1, num_samples // len(classes))
    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        n = min(samples_per_class, len(cls_indices))
        chosen = rng.choice(cls_indices, size=n, replace=False)
        selected_indices.extend(chosen)

    # Shuffle and limit
    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)
    selected_indices = selected_indices[:num_samples]

    # Add channel dim for mel
    X_selected = X_train[selected_indices].astype(np.float32)
    if feature_type == "mel":
        X_selected = X_selected[..., np.newaxis]

    def generator():
        for i in range(len(X_selected)):
            yield [X_selected[i : i + 1]]

    return generator


# ─── PTQ Conversion ──────────────────────────────────────────────────────


def convert_ptq(model: tf.keras.Model, representative_gen) -> bytes:
    """Convert a Keras model to int8 TFLite via Post-Training Quantization.

    Returns the TFLite model bytes.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


# ─── QAT Conversion ──────────────────────────────────────────────────────


def convert_qat(
    model_path: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_type: str,
    hp_config: dict,
    representative_gen,
) -> bytes:
    """Apply QAT fine-tuning to a float32 model, then convert to int8 TFLite.

    Returns the TFLite model bytes.
    """
    _set_seeds()

    # Load the float32 model
    model = tf.keras.models.load_model(model_path)

    # Apply quantization-aware training wrapper
    q_aware_model = tfmot.quantization.keras.quantize_model(model)

    # Reduced learning rate for QAT fine-tuning
    original_lr = hp_config.get("best_lr", 0.001)
    qat_lr = original_lr * QAT_LR_FACTOR

    q_aware_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=qat_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Class weights
    num_classes = model.output_shape[-1]
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_train,
    )
    class_weights = {i: float(w) for i, w in enumerate(weights)}

    # Prepare data (add channel dim for mel)
    X_train_input = X_train[..., np.newaxis] if feature_type == "mel" else X_train
    X_val_input = X_val[..., np.newaxis] if feature_type == "mel" else X_val

    # Fine-tune
    q_aware_model.fit(
        X_train_input, y_train,
        validation_data=(X_val_input, y_val),
        epochs=QAT_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=QAT_PATIENCE, restore_best_weights=True, verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=3, min_lr=1e-7, verbose=0,
            ),
        ],
        verbose=0,
    )

    # Convert QAT model to int8 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_gen

    return converter.convert()


# ─── TFLite Evaluation ───────────────────────────────────────────────────


def evaluate_tflite(
    tflite_path: str, X_test: np.ndarray, feature_type: str,
) -> dict:
    """Evaluate a TFLite int8 model on the test set.

    Returns dict with 'predictions' (argmax) and 'probabilities' (dequantized).
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale = input_details[0]["quantization"][0]
    input_zp = input_details[0]["quantization"][1]
    output_scale = output_details[0]["quantization"][0]
    output_zp = output_details[0]["quantization"][1]

    # Add channel dim for mel
    X = X_test[..., np.newaxis] if feature_type == "mel" else X_test

    predictions = []
    all_probs = []

    for i in range(len(X)):
        sample = X[i : i + 1].astype(np.float32)

        # Quantize input to int8
        quantized_input = np.clip(
            np.round(sample / input_scale + input_zp), -128, 127,
        ).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], quantized_input)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(np.argmax(output))

        # Dequantize output to get probabilities
        probs = (output.astype(np.float32) - output_zp) * output_scale
        all_probs.append(probs[0])

    return {
        "predictions": np.array(predictions),
        "probabilities": np.array(all_probs),
    }


def evaluate_float32(
    model_path: str, X_test: np.ndarray, feature_type: str,
) -> dict:
    """Evaluate a float32 Keras model on the test set.

    Returns dict with 'predictions' (argmax) and 'probabilities'.
    """
    model = tf.keras.models.load_model(model_path)

    X = X_test[..., np.newaxis] if feature_type == "mel" else X_test
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)

    return {"predictions": preds, "probabilities": probs}


# ─── Quantization Impact Analysis ────────────────────────────────────────


def run_mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
) -> dict:
    """McNemar's test comparing two classifiers on the same test set."""
    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    # n10: A correct, B wrong; n01: A wrong, B correct
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))

    # McNemar's test with continuity correction
    if n10 + n01 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n10": n10, "n01": n01}

    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p_value = float(1.0 - chi2_dist.cdf(chi2, df=1))

    return {"chi2": float(chi2), "p_value": p_value, "n10": n10, "n01": n01}


def compute_quantization_impact(
    f32_results: dict,
    int8_results: dict,
    y_test: np.ndarray,
    tier: int,
) -> dict:
    """Compute all quantization impact metrics."""
    f32_preds = f32_results["predictions"]
    int8_preds = int8_results["predictions"]
    f32_probs = f32_results["probabilities"]
    int8_probs = int8_results["probabilities"]

    # Basic metrics
    tier_cfg = get_tier_config(tier)
    num_classes = tier_cfg["num_classes"]

    f32_metrics = compute_metrics(y_test, f32_preds, tier=tier)
    int8_metrics = compute_metrics(y_test, int8_preds, tier=tier)

    # Agreement rate
    agreement = float(np.mean(f32_preds == int8_preds))

    # Per-class recall delta
    f32_recall = recall_score(y_test, f32_preds, average=None, labels=range(num_classes))
    int8_recall = recall_score(y_test, int8_preds, average=None, labels=range(num_classes))
    recall_delta = (int8_recall - f32_recall).tolist()

    # Confidence distribution stats
    f32_conf = np.max(f32_probs, axis=1)
    int8_conf = np.max(int8_probs, axis=1)

    # McNemar's test
    mcnemar = run_mcnemar_test(y_test, f32_preds, int8_preds)

    return {
        "f32_accuracy": f32_metrics["accuracy"],
        "f32_f1_macro": f32_metrics["f1_macro"],
        "int8_accuracy": int8_metrics["accuracy"],
        "int8_f1_macro": int8_metrics["f1_macro"],
        "delta_accuracy": float(f32_metrics["accuracy"] - int8_metrics["accuracy"]),
        "delta_f1_macro": float(f32_metrics["f1_macro"] - int8_metrics["f1_macro"]),
        "agreement_rate": agreement,
        "per_class_recall_delta": recall_delta,
        "f32_mean_confidence": float(np.mean(f32_conf)),
        "f32_std_confidence": float(np.std(f32_conf)),
        "int8_mean_confidence": float(np.mean(int8_conf)),
        "int8_std_confidence": float(np.std(int8_conf)),
        "mcnemar_chi2": mcnemar["chi2"],
        "mcnemar_p_value": mcnemar["p_value"],
        "mcnemar_n10": mcnemar["n10"],
        "mcnemar_n01": mcnemar["n01"],
    }


# ─── Visualizations ──────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confidence_comparison(
    f32_probs: np.ndarray,
    int8_probs: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Histogram overlay of float32 vs int8 confidence distributions."""
    f32_conf = np.max(f32_probs, axis=1)
    int8_conf = np.max(int8_probs, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(f32_conf, bins=bins, alpha=0.6, label="Float32", color="steelblue")
    ax.hist(int8_conf, bins=bins, alpha=0.6, label="Int8", color="coral")
    ax.set_xlabel("Prediction Confidence (max softmax)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_size_comparison(size_data: dict, save_path: str) -> None:
    """Bar chart comparing float32, int8 PTQ, int8 QAT model sizes."""
    models = list(size_data.keys())
    f32_sizes = [size_data[m]["float32_kb"] for m in models]
    ptq_sizes = [size_data[m]["ptq_kb"] for m in models]
    qat_sizes = [size_data[m]["qat_kb"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, f32_sizes, width, label="Float32", color="steelblue")
    ax.bar(x, ptq_sizes, width, label="Int8 PTQ", color="coral")
    ax.bar(x + width, qat_sizes, width, label="Int8 QAT", color="seagreen")

    ax.set_ylabel("Model Size (KB)")
    ax.set_title("Model Size Comparison: Float32 vs Int8")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add size labels
    for i, (f, p, q) in enumerate(zip(f32_sizes, ptq_sizes, qat_sizes)):
        ax.text(i - width, f + 1, f"{f:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, p + 1, f"{p:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width, q + 1, f"{q:.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_drop(impact_data: dict, save_path: str) -> None:
    """Grouped bar chart of PTQ and QAT F1 macro drop per model-tier."""
    labels = []
    ptq_drops = []
    qat_drops = []

    for key in sorted(impact_data.keys()):
        if "_ptq" in key:
            base_key = key.replace("_ptq", "_qat")
            ptq_d = impact_data[key]["impact_delta_f1_macro"]
            qat_d = impact_data.get(base_key, {}).get("impact_delta_f1_macro", 0)
            # Format label
            parts = key.replace("_ptq", "").split("_")
            label = f"{parts[0].upper()} T{parts[1][4:]}"
            labels.append(label)
            ptq_drops.append(ptq_d)
            qat_drops.append(qat_d)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, ptq_drops, width, label="PTQ Drop", color="coral")
    ax.bar(x + width / 2, qat_drops, width, label="QAT Drop", color="seagreen")

    ax.set_ylabel("F1 Macro Drop (Float32 - Int8)")
    ax.set_title("Quantization Accuracy Impact: PTQ vs QAT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    t_start = time.time()

    print("=" * 60)
    print("  Phase 4: Quantization (PTQ & QAT)")
    print("=" * 60)

    # Load hyperparameter configs
    hp_path = MODELS_DIR / "hyperparameter_search.json"
    with open(hp_path) as f:
        hp_configs = json.load(f)

    # ── Step 0: Retrain M5/M6 on Condition B ──────────────────────────

    # Retrain ALL models under legacy Keras (tf_keras) to ensure .h5
    # compatibility. M5/M6 switch to Condition B; M2 stays on Condition D.
    for model_name in MODEL_NAMES:
        condition = MODEL_CONDITIONS[model_name]
        cfg = get_model_config(model_name)
        hp = hp_configs[model_name]

        cond_label = "Condition B" if condition == "standard" else "Condition D"
        print(f"\n{'─' * 60}")
        print(f"  Retraining {cfg['display_name']} on {cond_label}")
        print(f"{'─' * 60}")

        for tier in TIERS:
            t0 = time.time()
            print(f"  Tier {tier}...", end="", flush=True)
            retrain_model(model_name, tier, condition, hp)
            print(f" done ({time.time() - t0:.1f}s)")

    # ── Step 1: Evaluate float32 baselines ────────────────────────────

    print(f"\n{'=' * 60}")
    print("  Float32 Baseline Evaluation")
    print(f"{'=' * 60}")

    f32_results = {}  # key: "{model}_tier{t}" -> {predictions, probabilities}
    f32_metrics = {}
    model_sizes = {}  # key: model_name -> {tierX_f32_kb, tierX_ptq_kb, ...}

    for model_name in MODEL_NAMES:
        cfg = get_model_config(model_name)
        condition = MODEL_CONDITIONS[model_name]

        for tier in TIERS:
            key = f"{model_name}_tier{tier}"
            model_path = str(MODELS_DIR / cfg["model_dir"] / f"tier{tier}_best.h5")

            # Load test data
            _, _, _, _, X_test, y_test = load_data(tier, cfg["feature_type"], condition)

            # Evaluate float32
            result = evaluate_float32(model_path, X_test, cfg["feature_type"])
            f32_results[key] = result
            f32_metrics[key] = compute_metrics(y_test, result["predictions"], tier=tier)

            # Record float32 model size
            f32_size_kb = os.path.getsize(model_path) / 1024
            if model_name not in model_sizes:
                model_sizes[model_name] = {}
            model_sizes[model_name][f"tier{tier}_f32_kb"] = f32_size_kb

            print(f"  {cfg['display_name']} Tier {tier}: "
                  f"acc={f32_metrics[key]['accuracy']:.4f} "
                  f"f1m={f32_metrics[key]['f1_macro']:.4f}")

    # ── Step 2: PTQ Conversion ────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("  Post-Training Quantization (PTQ)")
    print(f"{'=' * 60}")

    ptq_paths = {}  # key: "{model}_tier{t}" -> tflite path

    for model_name in MODEL_NAMES:
        cfg = get_model_config(model_name)
        condition = MODEL_CONDITIONS[model_name]

        print(f"\n  {cfg['display_name']}:")

        for tier in TIERS:
            key = f"{model_name}_tier{tier}"
            model_path = str(MODELS_DIR / cfg["model_dir"] / f"tier{tier}_best.h5")

            # Load training data for representative dataset
            X_train, y_train, _, _, _, _ = load_data(
                tier, cfg["feature_type"], condition,
            )

            # Create representative dataset
            rep_gen = create_representative_dataset(
                X_train, y_train, cfg["feature_type"],
            )

            # Load model and convert
            t0 = time.time()
            print(f"    Tier {tier}...", end="", flush=True)
            model = tf.keras.models.load_model(model_path)
            tflite_bytes = convert_ptq(model, rep_gen)

            # Save
            ptq_dir_key = f"{model_name}_ptq"
            out_dir = MODELS_DIR / TFLITE_DIRS[ptq_dir_key]
            os.makedirs(out_dir, exist_ok=True)

            if model_name == "m2":
                tflite_path = str(out_dir / f"tier{tier}.tflite")
            else:
                tflite_path = str(out_dir / f"tier{tier}_ptq.tflite")

            with open(tflite_path, "wb") as f:
                f.write(tflite_bytes)

            ptq_paths[key] = tflite_path
            size_kb = len(tflite_bytes) / 1024
            model_sizes[model_name][f"tier{tier}_ptq_kb"] = size_kb

            print(f" {size_kb:.1f} KB ({time.time() - t0:.1f}s)")

    # ── Step 3: QAT Fine-Tuning + Conversion ─────────────────────────

    print(f"\n{'=' * 60}")
    print("  Quantization-Aware Training (QAT)")
    print(f"{'=' * 60}")

    qat_paths = {}  # key: "{model}_tier{t}" -> tflite path

    for model_name in MODEL_NAMES:
        cfg = get_model_config(model_name)
        condition = MODEL_CONDITIONS[model_name]
        hp = hp_configs[model_name]

        print(f"\n  {cfg['display_name']} (LR={hp['best_lr'] * QAT_LR_FACTOR:.5f}):")

        for tier in TIERS:
            key = f"{model_name}_tier{tier}"
            model_path = str(MODELS_DIR / cfg["model_dir"] / f"tier{tier}_best.h5")

            # Load data
            X_train, y_train, X_val, y_val, _, _ = load_data(
                tier, cfg["feature_type"], condition,
            )

            # Representative dataset for conversion
            rep_gen = create_representative_dataset(
                X_train, y_train, cfg["feature_type"],
            )

            # QAT fine-tune + convert
            t0 = time.time()
            print(f"    Tier {tier}...", end="", flush=True)

            try:
                tflite_bytes = convert_qat(
                    model_path, X_train, y_train, X_val, y_val,
                    cfg["feature_type"], hp, rep_gen,
                )
            except Exception as e:
                print(f" FAILED: {e}")
                # Fall back to PTQ path if QAT fails
                qat_paths[key] = ptq_paths.get(key, "")
                continue

            # Save
            qat_dir_key = f"{model_name}_qat"
            out_dir = MODELS_DIR / TFLITE_DIRS[qat_dir_key]
            os.makedirs(out_dir, exist_ok=True)
            tflite_path = str(out_dir / f"tier{tier}.tflite")

            with open(tflite_path, "wb") as f:
                f.write(tflite_bytes)

            qat_paths[key] = tflite_path
            size_kb = len(tflite_bytes) / 1024
            model_sizes[model_name][f"tier{tier}_qat_kb"] = size_kb

            print(f" {size_kb:.1f} KB ({time.time() - t0:.1f}s)")

    # ── Step 4: Evaluate All Int8 Models ──────────────────────────────

    print(f"\n{'=' * 60}")
    print("  Int8 Model Evaluation")
    print(f"{'=' * 60}")

    all_results = {}  # Full results for JSON export

    for method, paths_dict, method_label in [
        ("ptq", ptq_paths, "PTQ"),
        ("qat", qat_paths, "QAT"),
    ]:
        print(f"\n  {method_label}:")
        for model_name in MODEL_NAMES:
            cfg = get_model_config(model_name)
            condition = MODEL_CONDITIONS[model_name]

            for tier in TIERS:
                key = f"{model_name}_tier{tier}"
                result_key = f"{key}_{method}"
                tflite_path = paths_dict.get(key, "")

                if not tflite_path or not os.path.exists(tflite_path):
                    print(f"    {cfg['display_name']} Tier {tier}: SKIPPED (no model)")
                    continue

                _, _, _, _, X_test, y_test = load_data(
                    tier, cfg["feature_type"], condition,
                )

                int8_result = evaluate_tflite(tflite_path, X_test, cfg["feature_type"])
                int8_metrics = compute_metrics(y_test, int8_result["predictions"], tier=tier)

                # Quantization impact
                impact = compute_quantization_impact(
                    f32_results[key], int8_result, y_test, tier,
                )

                all_results[result_key] = {
                    "model": model_name,
                    "tier": tier,
                    "method": method,
                    "condition": MODEL_CONDITIONS[model_name],
                    "tflite_path": tflite_path,
                    "tflite_size_kb": os.path.getsize(tflite_path) / 1024,
                    "int8_accuracy": int8_metrics["accuracy"],
                    "int8_f1_macro": int8_metrics["f1_macro"],
                    "int8_f1_weighted": int8_metrics["f1_weighted"],
                    **{f"impact_{k}": v for k, v in impact.items()},
                }

                print(
                    f"    {cfg['display_name']} Tier {tier}: "
                    f"acc={int8_metrics['accuracy']:.4f} "
                    f"f1m={int8_metrics['f1_macro']:.4f} "
                    f"(delta={impact['delta_f1_macro']:+.4f}, "
                    f"agree={impact['agreement_rate']:.1%})"
                )

                # Generate confusion matrices
                if method == "ptq":
                    cm_model_id = {"m2": "m3", "m5": "m5_ptq", "m6": "m6_ptq"}[model_name]
                else:
                    cm_model_id = {"m2": "m4", "m5": "m5_qat", "m6": "m6_qat"}[model_name]

                tier_cfg = get_tier_config(tier)
                class_names = tier_cfg["class_names"]

                from sklearn.metrics import confusion_matrix as sk_cm
                labels = list(range(tier_cfg["num_classes"]))
                cm_raw = sk_cm(y_test, int8_result["predictions"], labels=labels)
                cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1, keepdims=True)
                np.nan_to_num(cm_norm, copy=False, nan=0.0)

                plot_confusion_matrix(
                    cm_raw, class_names,
                    title=f"CM - {cfg['display_name']} {method_label} (Tier {tier})",
                    save_path=str(RESULTS_DIR / f"cm_{cm_model_id}_tier_{tier}.png"),
                    normalized=False,
                )
                plot_confusion_matrix(
                    cm_norm, class_names,
                    title=f"Normalized CM - {cfg['display_name']} {method_label} (Tier {tier})",
                    save_path=str(RESULTS_DIR / f"cm_{cm_model_id}_tier_{tier}_norm.png"),
                    normalized=True,
                )

                # ROC curve for Tier 1
                if tier == 1 and int8_result["probabilities"].shape[1] >= 2:
                    plot_roc_curve(
                        y_test,
                        int8_result["probabilities"][:, 1],
                        title=f"ROC - {cfg['display_name']} {method_label} (Tier 1)",
                        save_path=str(RESULTS_DIR / f"roc_{cm_model_id}_tier_1.png"),
                    )

    # ── Step 5: Confidence Comparison Plots ───────────────────────────

    print(f"\n  Generating confidence comparison plots...")

    for model_name in MODEL_NAMES:
        cfg = get_model_config(model_name)
        key = f"{model_name}_tier2"
        condition = MODEL_CONDITIONS[model_name]
        _, _, _, _, X_test, y_test = load_data(2, cfg["feature_type"], condition)

        ptq_path = ptq_paths.get(key)
        qat_path = qat_paths.get(key)

        if ptq_path and os.path.exists(ptq_path):
            ptq_result = evaluate_tflite(ptq_path, X_test, cfg["feature_type"])
            plot_confidence_comparison(
                f32_results[key]["probabilities"],
                ptq_result["probabilities"],
                f"{cfg['display_name']} Tier 2: Float32 vs PTQ Confidence",
                str(RESULTS_DIR / f"confidence_{model_name}_tier2_ptq.png"),
            )

        if qat_path and os.path.exists(qat_path):
            qat_result = evaluate_tflite(qat_path, X_test, cfg["feature_type"])
            plot_confidence_comparison(
                f32_results[key]["probabilities"],
                qat_result["probabilities"],
                f"{cfg['display_name']} Tier 2: Float32 vs QAT Confidence",
                str(RESULTS_DIR / f"confidence_{model_name}_tier2_qat.png"),
            )

    # ── Step 6: Size Comparison Chart ─────────────────────────────────

    size_chart_data = {}
    for model_name in MODEL_NAMES:
        cfg = get_model_config(model_name)
        ms = model_sizes.get(model_name, {})
        size_chart_data[cfg["display_name"]] = {
            "float32_kb": ms.get("tier2_f32_kb", 0),
            "ptq_kb": ms.get("tier2_ptq_kb", 0),
            "qat_kb": ms.get("tier2_qat_kb", 0),
        }

    plot_size_comparison(
        size_chart_data,
        str(RESULTS_DIR / "model_size_comparison.png"),
    )

    plot_accuracy_drop(all_results, str(RESULTS_DIR / "quantization_accuracy_drop.png"))

    # ── Step 7: Deployment Recommendation ─────────────────────────────

    print(f"\n{'=' * 60}")
    print("  Deployment Recommendation (Tier 2)")
    print(f"{'=' * 60}")

    tier2_candidates = []
    for result_key, data in all_results.items():
        if data["tier"] == 2:
            tier2_candidates.append((result_key, data))

    tier2_candidates.sort(key=lambda x: x[1]["int8_f1_macro"], reverse=True)

    print(f"\n  {'Rank':<5} {'Model':<25} {'Method':<5} {'F1 Macro':<10} "
          f"{'Size (KB)':<10} {'deltaF1':<8} {'Agreement':<10}")
    print(f"  {'─' * 75}")
    for rank, (rk, d) in enumerate(tier2_candidates[:6], 1):
        cfg = get_model_config(d["model"])
        print(f"  {rank:<5} {cfg['display_name']:<25} {d['method'].upper():<5} "
              f"{d['int8_f1_macro']:<10.4f} {d['tflite_size_kb']:<10.1f} "
              f"{d['impact_delta_f1_macro']:+<8.4f} "
              f"{d['impact_agreement_rate']:<10.1%}")

    # ── Step 8: Save Results ──────────────────────────────────────────

    print(f"\n  Saving results...")

    # Full JSON
    output = {
        "float32_baselines": {
            k: {
                "accuracy": v["accuracy"],
                "f1_macro": v["f1_macro"],
                "f1_weighted": v["f1_weighted"],
            }
            for k, v in f32_metrics.items()
        },
        "quantized_models": all_results,
        "model_sizes": model_sizes,
        "config": {
            "model_conditions": MODEL_CONDITIONS,
            "qat_epochs": QAT_EPOCHS,
            "qat_lr_factor": QAT_LR_FACTOR,
            "representative_samples": REPRESENTATIVE_SAMPLES,
        },
    }

    json_path = RESULTS_DIR / "quantization_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, cls=_NumpyEncoder)
    print(f"    {json_path}")

    # Summary CSV
    csv_path = RESULTS_DIR / "quantization_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "tier", "method", "condition",
            "f32_accuracy", "f32_f1_macro",
            "int8_accuracy", "int8_f1_macro",
            "delta_accuracy", "delta_f1_macro",
            "agreement_rate", "mcnemar_p",
            "tflite_size_kb",
        ])
        for rk, d in sorted(all_results.items()):
            f32_key = f"{d['model']}_tier{d['tier']}"
            f32 = f32_metrics.get(f32_key, {})
            writer.writerow([
                d["model"], d["tier"], d["method"], d["condition"],
                f"{f32.get('accuracy', 0):.4f}",
                f"{f32.get('f1_macro', 0):.4f}",
                f"{d['int8_accuracy']:.4f}",
                f"{d['int8_f1_macro']:.4f}",
                f"{d.get('impact_delta_accuracy', 0):.4f}",
                f"{d.get('impact_delta_f1_macro', 0):.4f}",
                f"{d.get('impact_agreement_rate', 0):.4f}",
                f"{d.get('impact_mcnemar_p_value', 0):.4f}",
                f"{d.get('tflite_size_kb', 0):.1f}",
            ])
    print(f"    {csv_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  DONE -- Phase 4 complete in {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")

    # Print artifact summary
    print(f"\nOutput artifacts:")
    for p in sorted(RESULTS_DIR.glob("quantization_*")):
        print(f"  {p.relative_to(PROJECT_ROOT)} ({p.stat().st_size / 1024:.1f} KB)")
    for model_dir in ["m3_cnn2d_int8_ptq", "m4_cnn2d_int8_qat",
                       "m5_cnn1d_int8_qat", "m6_dscnn_int8_qat"]:
        for p in sorted((MODELS_DIR / model_dir).glob("*.tflite")):
            print(f"  {p.relative_to(PROJECT_ROOT)} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
