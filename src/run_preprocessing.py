#!/usr/bin/env python3
"""Run the complete Phase 1 preprocessing pipeline.

Reads split manifests, preprocesses all audio, extracts features
(mel-spectrograms, MFCCs, classical ML vectors), computes normalization
statistics, and saves all artifacts.

Usage:
    python src/run_preprocessing.py

Outputs:
    data/features/mel_spectrograms/{train,val,test}.npz
    data/features/mfcc/{train,val,test}.npz
    data/classical_ml_features/{train,val,test}.npz
    data/normalization_stats.npz
    data/preprocessing_config.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add src/ to path for imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import preprocess_split, TARGET_SR, TARGET_SAMPLES, TARGET_DURATION_S
from features import (
    extract_mel_spectrograms,
    extract_mfccs,
    extract_classical_features,
    compute_normalization_stats,
    MEL_PARAMS,
    MFCC_N_COEFFS,
    CLASSICAL_FEATURE_NAMES,
    AGG_STAT_NAMES,
    N_CLASSICAL_FEATURES,
)

# ─── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = SRC_DIR.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MEL_DIR = PROJECT_ROOT / "data" / "features" / "mel_spectrograms"
MFCC_DIR = PROJECT_ROOT / "data" / "features" / "mfcc"
CLASSICAL_DIR = PROJECT_ROOT / "data" / "classical_ml_features"
DATA_DIR = PROJECT_ROOT / "data"


def save_features(
    X: np.ndarray,
    labels: dict[str, np.ndarray],
    file_paths: list[str],
    output_path: Path,
) -> None:
    """Save features and labels to a compressed npz file."""
    np.savez_compressed(
        output_path,
        X=X,
        y_tier1=labels["tier1"],
        y_tier2=labels["tier2"],
        y_tier3=labels["tier3"],
        file_paths=np.array(file_paths, dtype=object),
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def save_normalization_stats(
    mel_stats: dict[str, np.ndarray],
    mfcc_stats: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Save normalization statistics to npz."""
    np.savez(
        output_path,
        mel_mean=mel_stats["mean"],
        mel_std=mel_stats["std"],
        mfcc_mean=mfcc_stats["mean"],
        mfcc_std=mfcc_stats["std"],
    )
    print(f"  Saved: {output_path}")


def save_preprocessing_config(output_path: Path) -> None:
    """Save all preprocessing parameters to JSON for reproducibility."""
    config = {
        "target_sr": TARGET_SR,
        "target_duration_s": TARGET_DURATION_S,
        "target_samples": TARGET_SAMPLES,
        "normalization_method": "peak",
        "duration_method": "center_crop",
        "mel_spectrogram": {
            **MEL_PARAMS,
            "ref": "np.max",
            "output_shape": [40, 92],
        },
        "mfcc": {
            "n_mfcc": MFCC_N_COEFFS,
            "n_fft": MEL_PARAMS["n_fft"],
            "hop_length": MEL_PARAMS["hop_length"],
            "n_mels": MEL_PARAMS["n_mels"],
            "fmin": MEL_PARAMS["fmin"],
            "fmax": MEL_PARAMS["fmax"],
            "include_deltas_for_classical": True,
            "output_shape": [92, 13],
        },
        "classical_ml_features": {
            "n_frame_features": 63,
            "n_agg_stats": 7,
            "total_features": N_CLASSICAL_FEATURES,
            "frame_feature_names": CLASSICAL_FEATURE_NAMES,
            "agg_stat_names": AGG_STAT_NAMES,
        },
        "augmentation": {
            "time_shift_max_samples": 1600,
            "noise_snr_range_db": [5.0, 25.0],
            "pitch_shift_range_semitones": [-2.0, 2.0],
            "speed_perturb_range": [0.9, 1.1],
            "gain_range": [0.7, 1.3],
            "specaugment_freq_mask_param": 7,
            "specaugment_time_mask_param": 10,
            "specaugment_num_freq_masks": 1,
            "specaugment_num_time_masks": 1,
        },
        "split": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": 42,
            "stratify_by": "condition",
            "manifest_dir": "data/splits",
        },
        "tier_definitions": {
            "tier1": {
                "num_classes": 2,
                "names": ["Normal", "Anomaly"],
            },
            "tier2": {
                "num_classes": 6,
                "names": [
                    "Normal Braking",
                    "Braking Fault",
                    "Normal Idle",
                    "Idle Fault",
                    "Normal Start-Up",
                    "Start-Up Fault",
                ],
            },
            "tier3": {
                "num_classes": 9,
                "names": [
                    "Normal Braking",
                    "Worn Brakes",
                    "Normal Idle",
                    "Low Oil",
                    "Power Steering Fault",
                    "Serpentine Belt Fault",
                    "Normal Start-Up",
                    "Bad Ignition",
                    "Dead Battery",
                ],
                "note": "combined-fault files excluded (tier3_label=-1)",
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Saved: {output_path}")


def main() -> None:
    """Run the full preprocessing pipeline."""
    start_time = time.time()

    print("=" * 60)
    print("  Phase 1: Data Preprocessing Pipeline")
    print("=" * 60)

    # ── Ensure output directories exist ──────────────────────────────────
    MEL_DIR.mkdir(parents=True, exist_ok=True)
    MFCC_DIR.mkdir(parents=True, exist_ok=True)
    CLASSICAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Process each split ───────────────────────────────────────────────
    splits = {}
    for split_name in ["train", "val", "test"]:
        manifest_path = SPLITS_DIR / f"{split_name}_manifest.csv"
        print(f"\n[{split_name.upper()}] Loading and preprocessing audio...")
        audio, labels, file_paths = preprocess_split(
            str(manifest_path), project_root=str(PROJECT_ROOT)
        )
        print(f"  Audio shape: {audio.shape}, dtype: {audio.dtype}")
        splits[split_name] = {
            "audio": audio,
            "labels": labels,
            "file_paths": file_paths,
        }

    # ── Extract features ─────────────────────────────────────────────────
    for split_name, data in splits.items():
        audio = data["audio"]
        labels = data["labels"]
        file_paths = data["file_paths"]

        print(f"\n[{split_name.upper()}] Extracting features...")

        # Mel-spectrograms
        mel_specs = extract_mel_spectrograms(audio)
        print(f"  Mel-spectrograms: {mel_specs.shape}")
        save_features(mel_specs, labels, file_paths, MEL_DIR / f"{split_name}.npz")

        # MFCCs
        mfccs = extract_mfccs(audio)
        print(f"  MFCCs: {mfccs.shape}")
        save_features(mfccs, labels, file_paths, MFCC_DIR / f"{split_name}.npz")

        # Classical ML features
        classical = extract_classical_features(audio)
        print(f"  Classical features: {classical.shape}")
        save_features(
            classical, labels, file_paths, CLASSICAL_DIR / f"{split_name}.npz"
        )

        # Free audio memory after feature extraction
        del audio
        data["audio"] = None

    # ── Compute normalization stats from training set ────────────────────
    print("\n[NORMALIZATION] Computing statistics from training set...")

    train_mel = np.load(MEL_DIR / "train.npz")["X"]
    mel_stats = compute_normalization_stats(train_mel, feature_type="mel")
    print(f"  Mel stats: mean shape {mel_stats['mean'].shape}, std shape {mel_stats['std'].shape}")
    del train_mel

    train_mfcc = np.load(MFCC_DIR / "train.npz")["X"]
    mfcc_stats = compute_normalization_stats(train_mfcc, feature_type="mfcc")
    print(f"  MFCC stats: mean shape {mfcc_stats['mean'].shape}, std shape {mfcc_stats['std'].shape}")
    del train_mfcc

    save_normalization_stats(mel_stats, mfcc_stats, DATA_DIR / "normalization_stats.npz")

    # ── Save preprocessing config ────────────────────────────────────────
    print("\n[CONFIG] Saving preprocessing configuration...")
    save_preprocessing_config(DATA_DIR / "preprocessing_config.json")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  DONE — Preprocessing complete in {elapsed:.1f}s")
    print("=" * 60)

    print("\nOutput artifacts:")
    for d in [MEL_DIR, MFCC_DIR, CLASSICAL_DIR]:
        for f in sorted(d.glob("*.npz")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")
    print(f"  {(DATA_DIR / 'normalization_stats.npz').relative_to(PROJECT_ROOT)}")
    print(f"  {(DATA_DIR / 'preprocessing_config.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
