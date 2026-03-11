#!/usr/bin/env python3
"""
Dataset Verification & Train/Val/Test Split Generation

Phase 0, Task 0.5: Verify the Car Diagnostics Dataset matches expectations.
Phase 1, Task 1.4: Generate stratified train/val/test split manifests with
                    label mappings for all 3 taxonomy tiers.

Usage:
    python project_setup/verify_dataset_and_split.py

Outputs:
    data/splits/train_manifest.csv
    data/splits/val_manifest.csv
    data/splits/test_manifest.csv

Each manifest contains columns:
    file_path, filename, state, condition,
    sample_rate, duration_s, channels,
    tier1_label, tier1_name,
    tier2_label, tier2_name,
    tier3_label, tier3_name
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

# ─── Configuration ──────────────────────────────────────────────────────────

DATASET_ROOT = Path("car_diagnostics_dataset")
SPLITS_DIR = Path("data/splits")
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Expected counts (from EDA notebook) ────────────────────────────────────

EXPECTED_TOTAL = 1386
EXPECTED_COUNTS = {
    ("braking state", "normal_brakes"): 77,
    ("braking state", "worn_out_brakes"): 76,
    ("idle state", "normal_engine_idle"): 264,
    ("idle state", "low_oil"): 107,
    ("idle state", "power_steering"): 129,
    ("idle state", "serpentine_belt"): 116,
    ("idle state", "combined/no oil_serpentine belt"): 107,
    ("idle state", "combined/power steering combined_no oil"): 107,
    ("idle state", "combined/power steering combined_no oil_serpentine belt"): 107,
    ("idle state", "combined/power steering combined_serpentine belt"): 116,
    ("startup state", "normal_engine_startup"): 61,
    ("startup state", "bad_ignition"): 62,
    ("startup state", "dead_battery"): 57,
}

# ─── Tier 1: Binary Anomaly Detection (2 classes) ──────────────────────────

TIER1_MAP = {
    "normal_brakes": (0, "Normal"),
    "worn_out_brakes": (1, "Anomaly"),
    "normal_engine_idle": (0, "Normal"),
    "low_oil": (1, "Anomaly"),
    "power_steering": (1, "Anomaly"),
    "serpentine_belt": (1, "Anomaly"),
    "combined/no oil_serpentine belt": (1, "Anomaly"),
    "combined/power steering combined_no oil": (1, "Anomaly"),
    "combined/power steering combined_no oil_serpentine belt": (1, "Anomaly"),
    "combined/power steering combined_serpentine belt": (1, "Anomaly"),
    "normal_engine_startup": (0, "Normal"),
    "bad_ignition": (1, "Anomaly"),
    "dead_battery": (1, "Anomaly"),
}

# ─── Tier 2: State-Aware Anomaly Detection (6 classes) ─────────────────────

TIER2_MAP = {
    "normal_brakes": (0, "Normal Braking"),
    "worn_out_brakes": (1, "Braking Fault"),
    "normal_engine_idle": (2, "Normal Idle"),
    "low_oil": (3, "Idle Fault"),
    "power_steering": (3, "Idle Fault"),
    "serpentine_belt": (3, "Idle Fault"),
    "combined/no oil_serpentine belt": (3, "Idle Fault"),
    "combined/power steering combined_no oil": (3, "Idle Fault"),
    "combined/power steering combined_no oil_serpentine belt": (3, "Idle Fault"),
    "combined/power steering combined_serpentine belt": (3, "Idle Fault"),
    "normal_engine_startup": (4, "Normal Start-Up"),
    "bad_ignition": (5, "Start-Up Fault"),
    "dead_battery": (5, "Start-Up Fault"),
}

# ─── Tier 3: Full Fault Identification (9 classes, excl. combined) ──────────

TIER3_MAP = {
    "normal_brakes": (0, "Normal Braking"),
    "worn_out_brakes": (1, "Worn Brakes"),
    "normal_engine_idle": (2, "Normal Idle"),
    "low_oil": (3, "Low Oil"),
    "power_steering": (4, "Power Steering Fault"),
    "serpentine_belt": (5, "Serpentine Belt Fault"),
    "normal_engine_startup": (6, "Normal Start-Up"),
    "bad_ignition": (7, "Bad Ignition"),
    "dead_battery": (8, "Dead Battery"),
    # Combined-fault conditions are excluded from Tier 3
    "combined/no oil_serpentine belt": (-1, "EXCLUDED"),
    "combined/power steering combined_no oil": (-1, "EXCLUDED"),
    "combined/power steering combined_no oil_serpentine belt": (-1, "EXCLUDED"),
    "combined/power steering combined_serpentine belt": (-1, "EXCLUDED"),
}


def scan_dataset(dataset_root: Path) -> pd.DataFrame:
    """Scan the dataset directory and build a file catalog."""
    records = []
    for wav_path in sorted(dataset_root.rglob("*.wav")):
        parts = wav_path.relative_to(dataset_root).parts
        state = parts[0]
        condition = "/".join(parts[1:-1])
        records.append({
            "file_path": str(wav_path),
            "filename": wav_path.name,
            "state": state,
            "condition": condition,
        })
    return pd.DataFrame(records)


def verify_audio_properties(df: pd.DataFrame) -> bool:
    """Read audio metadata for every file and verify properties."""
    print("\n  Reading audio metadata for all files...")
    durations = []
    sample_rates = []
    channels_list = []
    errors = []

    for i, row in df.iterrows():
        try:
            info = sf.info(row["file_path"])
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
            channels_list.append(info.channels)
        except Exception as e:
            errors.append((row["file_path"], str(e)))
            durations.append(None)
            sample_rates.append(None)
            channels_list.append(None)

    df["duration_s"] = durations
    df["sample_rate"] = sample_rates
    df["channels"] = channels_list

    ok = True
    if errors:
        print(f"  ERROR: {len(errors)} files could not be read:")
        for path, err in errors[:5]:
            print(f"    {path}: {err}")
        ok = False

    # Check all mono
    non_mono = df[df["channels"] != 1]
    if len(non_mono) > 0:
        print(f"  ERROR: {len(non_mono)} files are not mono!")
        ok = False
    else:
        print("  OK: All files are mono")

    # Check sample rates
    srs = sorted(df["sample_rate"].dropna().unique())
    print(f"  Sample rates found: {srs}")
    expected_srs = [24000, 44100, 48000]
    if set(srs) != set(expected_srs):
        print(f"  WARNING: Expected sample rates {expected_srs}, got {srs}")
    else:
        print("  OK: Sample rates match expectations (24kHz, 44.1kHz, 48kHz)")

    # Check durations
    dur_stats = df.groupby("state")["duration_s"].agg(["min", "max", "mean"]).round(3)
    print(f"  Duration by state:")
    for state, row in dur_stats.iterrows():
        print(f"    {state}: min={row['min']:.3f}s, max={row['max']:.3f}s, mean={row['mean']:.3f}s")

    return ok


def verify_counts(df: pd.DataFrame) -> bool:
    """Verify file counts match expectations."""
    ok = True

    # Total count
    if len(df) != EXPECTED_TOTAL:
        print(f"  ERROR: Expected {EXPECTED_TOTAL} total files, found {len(df)}")
        ok = False
    else:
        print(f"  OK: Total file count = {EXPECTED_TOTAL}")

    # Per-condition counts
    for (state, condition), expected_count in EXPECTED_COUNTS.items():
        actual = len(df[(df["state"] == state) & (df["condition"] == condition)])
        if actual != expected_count:
            print(f"  ERROR: {state}/{condition}: expected {expected_count}, found {actual}")
            ok = False

    if ok:
        print(f"  OK: All {len(EXPECTED_COUNTS)} condition counts match expectations")

    return ok


def apply_tier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add tier label columns to the dataframe."""
    tier1_labels = []
    tier1_names = []
    tier2_labels = []
    tier2_names = []
    tier3_labels = []
    tier3_names = []

    for _, row in df.iterrows():
        cond = row["condition"]

        t1_label, t1_name = TIER1_MAP[cond]
        tier1_labels.append(t1_label)
        tier1_names.append(t1_name)

        t2_label, t2_name = TIER2_MAP[cond]
        tier2_labels.append(t2_label)
        tier2_names.append(t2_name)

        t3_label, t3_name = TIER3_MAP[cond]
        tier3_labels.append(t3_label)
        tier3_names.append(t3_name)

    df["tier1_label"] = tier1_labels
    df["tier1_name"] = tier1_names
    df["tier2_label"] = tier2_labels
    df["tier2_name"] = tier2_names
    df["tier3_label"] = tier3_labels
    df["tier3_name"] = tier3_names

    return df


def generate_splits(df: pd.DataFrame) -> tuple:
    """Generate stratified train/val/test splits."""
    # Stratify by the finest-grained label (condition)
    # This ensures proportional representation in all sets
    file_paths = df.index.values
    conditions = df["condition"].values

    # First split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        file_paths,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        stratify=conditions,
    )

    # Second split: 50% of temp → 15% val, 15% test
    temp_conditions = conditions[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_conditions,
    )

    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]


def print_split_summary(name: str, split_df: pd.DataFrame):
    """Print a summary of a data split."""
    print(f"\n  {name}: {len(split_df)} files")

    # Tier 1 distribution
    t1_counts = split_df["tier1_name"].value_counts().sort_index()
    print(f"    Tier 1: {dict(t1_counts)}")

    # Tier 2 distribution
    t2_counts = split_df["tier2_name"].value_counts().sort_index()
    print(f"    Tier 2: {dict(t2_counts)}")

    # Tier 3 distribution (excluding EXCLUDED)
    t3_df = split_df[split_df["tier3_label"] >= 0]
    t3_counts = t3_df["tier3_name"].value_counts().sort_index()
    t3_excluded = len(split_df) - len(t3_df)
    print(f"    Tier 3: {dict(t3_counts)} (+{t3_excluded} excluded combined-fault files)")


def main():
    print("=" * 60)
    print("  Dataset Verification & Split Generation")
    print("=" * 60)

    # ---- Check dataset exists ----
    if not DATASET_ROOT.exists():
        print(f"\nERROR: Dataset root not found at '{DATASET_ROOT}'")
        print("Please download the Car Diagnostics Dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset")
        sys.exit(1)

    # ---- Scan files ----
    print("\n[1/5] Scanning dataset directory...")
    df = scan_dataset(DATASET_ROOT)
    print(f"  Found {len(df)} WAV files")

    # ---- Verify counts ----
    print("\n[2/5] Verifying file counts...")
    counts_ok = verify_counts(df)

    # ---- Verify audio properties ----
    print("\n[3/5] Verifying audio properties...")
    audio_ok = verify_audio_properties(df)

    if not counts_ok or not audio_ok:
        print("\nWARNING: Some verifications failed. Review the errors above.")
        print("Continuing with split generation anyway...")
    else:
        print("\n  All verifications passed!")

    # ---- Apply tier labels ----
    print("\n[4/5] Applying tier labels...")
    df = apply_tier_labels(df)

    # Print tier label distributions
    print("\n  Tier 1 (Binary) distribution:")
    for name, count in df["tier1_name"].value_counts().sort_index().items():
        print(f"    {name}: {count}")

    print("\n  Tier 2 (6-class) distribution:")
    for name, count in df["tier2_name"].value_counts().sort_index().items():
        print(f"    {name}: {count}")

    print("\n  Tier 3 (9-class, excl. combined) distribution:")
    t3_valid = df[df["tier3_label"] >= 0]
    for name, count in t3_valid["tier3_name"].value_counts().sort_index().items():
        print(f"    {name}: {count}")
    t3_excluded = len(df) - len(t3_valid)
    print(f"    (Excluded combined-fault files: {t3_excluded})")

    # ---- Generate splits ----
    print("\n[5/5] Generating stratified train/val/test splits...")
    print(f"  Ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  Stratified by: condition (finest-grained label)")

    train_df, val_df, test_df = generate_splits(df)

    print_split_summary("Training set", train_df)
    print_split_summary("Validation set", val_df)
    print_split_summary("Test set", test_df)

    # ---- Save manifests ----
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "file_path", "filename", "state", "condition",
        "sample_rate", "duration_s", "channels",
        "tier1_label", "tier1_name",
        "tier2_label", "tier2_name",
        "tier3_label", "tier3_name",
    ]

    train_path = SPLITS_DIR / "train_manifest.csv"
    val_path = SPLITS_DIR / "val_manifest.csv"
    test_path = SPLITS_DIR / "test_manifest.csv"

    train_df[columns].to_csv(train_path, index=False)
    val_df[columns].to_csv(val_path, index=False)
    test_df[columns].to_csv(test_path, index=False)

    print(f"\n  Saved: {train_path} ({len(train_df)} rows)")
    print(f"  Saved: {val_path} ({len(val_df)} rows)")
    print(f"  Saved: {test_path} ({len(test_df)} rows)")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  DONE — Dataset verified and splits generated.")
    print("=" * 60)
    print(f"\nSplit manifests are in: {SPLITS_DIR}/")
    print("These manifests are the source of truth for all training,")
    print("validation, and test set membership. All team members should")
    print("use these exact splits for reproducibility.")


if __name__ == "__main__":
    main()
