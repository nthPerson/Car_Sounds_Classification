"""Generate pre-computed augmented training data via waveform augmentation.

Loads the original training audio from disk, applies waveform-domain
augmentations (time shift, noise, pitch shift, speed perturbation, random
gain), re-extracts features (mel-spectrograms and MFCCs), and saves the
expanded training set alongside the originals.

Class-aware augmentation: minority classes receive more augmented copies
to mitigate class imbalance.  The target is to bring every class to
roughly the same effective count.

Usage:
    python src/generate_augmented_data.py

Outputs:
    data/features/mel_spectrograms/train_augmented.npz
    data/features/mfcc/train_augmented.npz
    data/normalization_stats_augmented.npz  (recomputed from augmented train)
    data/augmentation_report.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from augmentation import augment_waveform
from features import (
    compute_log_mel_spectrogram,
    compute_mfcc,
    compute_normalization_stats,
)
from preprocessing import preprocess_split

# ─── Constants ────────────────────────────────────────────────────────────

PROJECT_ROOT = SRC_DIR.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MEL_DIR = PROJECT_ROOT / "data" / "features" / "mel_spectrograms"
MFCC_DIR = PROJECT_ROOT / "data" / "features" / "mfcc"
DATA_DIR = PROJECT_ROOT / "data"

RANDOM_STATE = 42
TARGET_SR = 16000

# ─── Augmentation multipliers ─────────────────────────────────────────────
#
# Goal: bring every Tier 2 class to a target count of ~550 (matching the
# largest class, Idle Fault).  This simultaneously balances Tier 1 and
# improves Tier 3 balance.
#
# Original Tier 2 training counts:
#   Normal Braking     54   → need ~500 augmented  → ~9x multiplier
#   Braking Fault      53   → need ~500 augmented  → ~9x multiplier
#   Normal Idle       185   → need ~365 augmented  → ~2x multiplier
#   Idle Fault        552   → need ~0   augmented  → 0x (keep original only)
#   Normal Start-Up    43   → need ~510 augmented  → ~12x multiplier
#   Start-Up Fault     83   → need ~470 augmented  → ~6x multiplier
#
# We use per-condition multipliers (since each sample has a unique
# condition in the original dataset, and Tier 2 classes map 1:1 to
# conditions grouped by state+fault).
#
# Rather than exact balancing, we target a range of 400-600 per Tier 2
# class to avoid generating too many near-duplicate augmented samples
# from the very smallest classes.

TARGET_PER_CLASS = 550  # approximate target per Tier 2 class


def compute_multipliers(y_tier2: np.ndarray, num_classes: int = 6) -> dict[int, int]:
    """Compute per-class augmentation multipliers to reach TARGET_PER_CLASS.

    Returns:
        Dict mapping Tier 2 class label → number of augmented copies per
        original sample (0 means no augmentation for that class).
    """
    counts = {i: int((y_tier2 == i).sum()) for i in range(num_classes)}
    multipliers = {}
    for cls, count in counts.items():
        if count >= TARGET_PER_CLASS:
            multipliers[cls] = 0
        else:
            # Compute how many augmented copies per original sample
            needed = TARGET_PER_CLASS - count
            mult = max(1, round(needed / count))
            multipliers[cls] = mult
    return multipliers


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    t0 = time.time()

    print("=" * 60)
    print("  Waveform Augmentation — Expanding Training Set")
    print("=" * 60)

    # ── Load original training audio ─────────────────────────────────────
    print("\n[1/4] Loading original training audio...")
    manifest_path = SPLITS_DIR / "train_manifest.csv"
    audio_orig, labels_orig, file_paths_orig = preprocess_split(
        str(manifest_path), project_root=str(PROJECT_ROOT)
    )
    n_orig = len(audio_orig)
    y_tier1 = labels_orig["tier1"]
    y_tier2 = labels_orig["tier2"]
    y_tier3 = labels_orig["tier3"]
    print(f"  Original training set: {n_orig} samples")

    # ── Compute per-class multipliers ────────────────────────────────────
    multipliers = compute_multipliers(y_tier2)

    print("\n  Augmentation plan (Tier 2 classes):")
    print(f"  {'Class':<20s} {'Original':>8s} {'Mult':>5s} {'Augmented':>10s} {'Total':>7s}")
    print(f"  {'-'*55}")

    import json as _json
    with open(DATA_DIR / "preprocessing_config.json") as f:
        tier2_names = _json.load(f)["tier_definitions"]["tier2"]["names"]

    total_aug = 0
    for cls in range(6):
        count = int((y_tier2 == cls).sum())
        mult = multipliers[cls]
        n_aug = count * mult
        total_aug += n_aug
        total = count + n_aug
        print(f"  {tier2_names[cls]:<20s} {count:>8d} {mult:>5d}x {n_aug:>10d} {total:>7d}")

    print(f"\n  Total augmented samples to generate: {total_aug}")
    print(f"  Final training set size: {n_orig + total_aug}")

    # ── Generate augmented waveforms ─────────────────────────────────────
    print("\n[2/4] Generating augmented waveforms...")

    aug_audio_list = []
    aug_tier1_list = []
    aug_tier2_list = []
    aug_tier3_list = []
    aug_paths_list = []

    rng = np.random.default_rng(RANDOM_STATE)

    for i in tqdm(range(n_orig), desc="Augmenting"):
        cls = y_tier2[i]
        mult = multipliers[cls]
        for j in range(mult):
            # Each augmented copy gets a unique seed
            aug_rng = np.random.default_rng(RANDOM_STATE + i * 100 + j)
            y_aug = augment_waveform(audio_orig[i], sr=TARGET_SR, rng=aug_rng)
            aug_audio_list.append(y_aug)
            aug_tier1_list.append(y_tier1[i])
            aug_tier2_list.append(y_tier2[i])
            aug_tier3_list.append(y_tier3[i])
            aug_paths_list.append(f"augmented/{file_paths_orig[i]}::aug{j}")

    aug_audio = np.stack(aug_audio_list, axis=0).astype(np.float32)
    print(f"  Generated {len(aug_audio)} augmented waveforms")

    # Combine original + augmented
    all_audio = np.concatenate([audio_orig, aug_audio], axis=0)
    all_tier1 = np.concatenate([y_tier1, np.array(aug_tier1_list, dtype=np.int32)])
    all_tier2 = np.concatenate([y_tier2, np.array(aug_tier2_list, dtype=np.int32)])
    all_tier3 = np.concatenate([y_tier3, np.array(aug_tier3_list, dtype=np.int32)])
    all_paths = file_paths_orig + aug_paths_list

    print(f"  Combined training set: {len(all_audio)} samples")

    # Print final class distribution
    print("\n  Final Tier 2 distribution:")
    for cls in range(6):
        count = int((all_tier2 == cls).sum())
        print(f"    {tier2_names[cls]:<20s}: {count}")

    # Free memory
    del audio_orig, aug_audio, aug_audio_list

    # ── Extract features from combined set ───────────────────────────────
    print("\n[3/4] Extracting features from combined training set...")

    # Mel-spectrograms
    mel_specs = []
    for i in tqdm(range(len(all_audio)), desc="Mel-spectrograms"):
        spec = compute_log_mel_spectrogram(all_audio[i], sr=TARGET_SR)
        mel_specs.append(spec)
    mel_specs = np.stack(mel_specs, axis=0).astype(np.float32)
    print(f"  Mel-spectrograms: {mel_specs.shape}")

    # MFCCs
    mfccs = []
    for i in tqdm(range(len(all_audio)), desc="MFCCs"):
        mfcc = compute_mfcc(all_audio[i], sr=TARGET_SR)
        mfccs.append(mfcc)
    mfccs = np.stack(mfccs, axis=0).astype(np.float32)
    print(f"  MFCCs: {mfccs.shape}")

    del all_audio  # free waveform memory

    # ── Save augmented features ──────────────────────────────────────────
    print("\n[4/4] Saving augmented training features...")

    labels_dict = {
        "tier1": all_tier1,
        "tier2": all_tier2,
        "tier3": all_tier3,
    }

    # Mel-spectrograms
    mel_path = MEL_DIR / "train_augmented.npz"
    np.savez_compressed(
        mel_path,
        X=mel_specs,
        y_tier1=all_tier1, y_tier2=all_tier2, y_tier3=all_tier3,
        file_paths=np.array(all_paths, dtype=object),
    )
    size_mb = mel_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {mel_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")

    # MFCCs
    mfcc_path = MFCC_DIR / "train_augmented.npz"
    np.savez_compressed(
        mfcc_path,
        X=mfccs,
        y_tier1=all_tier1, y_tier2=all_tier2, y_tier3=all_tier3,
        file_paths=np.array(all_paths, dtype=object),
    )
    size_mb = mfcc_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {mfcc_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")

    # ── Recompute normalization stats from augmented training set ─────────
    print("\n  Recomputing normalization stats from augmented training set...")
    mel_stats = compute_normalization_stats(mel_specs, feature_type="mel")
    mfcc_stats = compute_normalization_stats(mfccs, feature_type="mfcc")

    stats_path = DATA_DIR / "normalization_stats_augmented.npz"
    np.savez(
        stats_path,
        mel_mean=mel_stats["mean"], mel_std=mel_stats["std"],
        mfcc_mean=mfcc_stats["mean"], mfcc_std=mfcc_stats["std"],
    )
    print(f"  Saved: {stats_path.relative_to(PROJECT_ROOT)}")

    # ── Save augmentation report ─────────────────────────────────────────
    report = {
        "original_train_size": n_orig,
        "augmented_samples_generated": total_aug,
        "final_train_size": n_orig + total_aug,
        "target_per_class": TARGET_PER_CLASS,
        "multipliers": {tier2_names[k]: v for k, v in multipliers.items()},
        "final_tier2_distribution": {
            tier2_names[i]: int((all_tier2 == i).sum()) for i in range(6)
        },
        "augmentation_techniques": [
            "time_shift (50% prob, ±100ms)",
            "add_noise (50% prob, SNR 5-25 dB)",
            "pitch_shift (30% prob, ±2 semitones)",
            "speed_perturb (30% prob, 0.9-1.1x)",
            "random_gain (50% prob, 0.7-1.3x)",
        ],
        "random_state": RANDOM_STATE,
    }
    report_path = DATA_DIR / "augmentation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path.relative_to(PROJECT_ROOT)}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DONE — Augmentation complete in {elapsed:.1f}s")
    print(f"  Training set expanded: {n_orig} → {n_orig + total_aug} samples")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
