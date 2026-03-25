"""Feature parity validation: compare librosa pipeline vs simulated on-device algorithm.

Verifies that the on-device mel-spectrogram computation (manual FFT + sparse
mel filterbank + global-max log + z-score normalization + int8 quantization)
produces results matching the Python training pipeline closely enough to
preserve model accuracy.

Run BEFORE flashing the Arduino to catch training-serving skew early.

Usage:
    python src/validate_features.py
"""

import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import MEL_PARAMS, compute_log_mel_spectrogram
from preprocessing import peak_normalize, standardize_duration

# ─── Paths ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
DATASET_DIR = PROJECT_ROOT / "car_diagnostics_dataset"

# ─── Constants ────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 40
FMIN = 20
FMAX = 8000
N_FRAMES = 92
AUDIO_SAMPLES = 24000

# M6 PTQ Tier 2 quantization params
INPUT_SCALE = 0.03472795709967613
INPUT_ZERO_POINT = 40


# ─── Simulated On-Device Pipeline ────────────────────────────────────────


def compute_mel_spectrogram_ondevice(audio_int16: np.ndarray) -> np.ndarray:
    """Simulate the on-device mel-spectrogram pipeline in Python.

    This uses the exact same algorithm as feature_extraction.cpp:
    1. Peak normalize (int16 -> float32)
    2. Frame extraction with Hann window, no centering
    3. FFT -> power spectrum
    4. Sparse mel filterbank
    5. Global max log conversion (ref=max, top_db=80)

    Args:
        audio_int16: Raw int16 audio samples (24000,).

    Returns:
        Log-mel spectrogram in dB, shape (40, 92).
    """
    # Phase A: Peak normalization (like Arduino: find max abs, divide)
    max_abs = np.max(np.abs(audio_int16.astype(np.float32)))
    if max_abs == 0:
        max_abs = 1.0
    norm_factor = 1.0 / max_abs

    # Load mel filterbank (same as exported to Arduino)
    mel_fb = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )

    # Hann window (periodic, matching np.hanning)
    window = np.hanning(N_FFT).astype(np.float32)

    # Phase B: Frame-by-frame FFT + mel energy
    mel_spectrogram = np.zeros((N_MELS, N_FRAMES), dtype=np.float32)

    for t in range(N_FRAMES):
        offset = t * HOP_LENGTH

        # Extract frame and normalize
        frame = np.zeros(N_FFT, dtype=np.float32)
        end = min(offset + N_FFT, len(audio_int16))
        frame[: end - offset] = audio_int16[offset:end].astype(np.float32) * norm_factor

        # Apply Hann window
        frame *= window

        # FFT -> power spectrum
        fft_out = np.fft.rfft(frame)  # (257,) complex
        power = np.abs(fft_out) ** 2  # (257,) float

        # Mel filterbank (using sparse representation matching Arduino)
        for m in range(N_MELS):
            nz = np.nonzero(mel_fb[m])[0]
            if len(nz) > 0:
                start = nz[0]
                end_bin = nz[-1] + 1
                mel_spectrogram[m, t] = np.dot(
                    mel_fb[m, start:end_bin], power[start:end_bin],
                )

    # Phase C: Global max log conversion
    max_energy = np.max(mel_spectrogram)
    if max_energy < 1e-10:
        max_energy = 1e-10

    mel_spectrogram = 10.0 * np.log10(mel_spectrogram / max_energy + 1e-10)

    # Clamp to -80 dB (librosa's default top_db=80)
    mel_spectrogram = np.maximum(mel_spectrogram, -80.0)

    return mel_spectrogram.astype(np.float32)


def normalize_spectrogram(spectrogram: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply per-band z-score normalization."""
    return (spectrogram - mean[:, np.newaxis]) / std[:, np.newaxis]


def quantize_to_int8(features: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Quantize float32 features to int8."""
    quantized = np.round(features / scale) + zero_point
    return np.clip(quantized, -128, 127).astype(np.int8)


# ─── Reference Pipeline ──────────────────────────────────────────────────


def compute_reference_pipeline(audio_float32: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Run the full Python/librosa training pipeline.

    Returns (mel_db, mel_normalized, mel_int8).
    """
    mel_db = compute_log_mel_spectrogram(audio_float32, sr=SAMPLE_RATE)
    mel_norm = normalize_spectrogram(mel_db, mean, std)
    mel_int8 = quantize_to_int8(mel_norm, INPUT_SCALE, INPUT_ZERO_POINT)
    return mel_db, mel_norm, mel_int8


# ─── Comparison Metrics ──────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    import pandas as pd

    print("=" * 60)
    print("  Feature Parity Validation")
    print("=" * 60)

    # Load normalization stats (Condition B for M6)
    stats = np.load(DATA_DIR / "normalization_stats_augmented.npz")
    mel_mean = stats["mel_mean"].astype(np.float32)
    mel_std = stats["mel_std"].astype(np.float32)

    # Load test manifest and select stratified subset
    manifest = pd.read_csv(SPLITS_DIR / "test_manifest.csv")

    # Stratified: take up to 4 samples per Tier 2 class (24 total)
    selected = []
    for label in range(6):
        cls_samples = manifest[manifest["tier2_label"] == label]
        n = min(4, len(cls_samples))
        selected.append(cls_samples.sample(n=n, random_state=42))
    selected = pd.concat(selected).reset_index(drop=True)

    print(f"\nValidating {len(selected)} test clips across {6} classes")
    print(f"Normalization: normalization_stats_augmented.npz (Condition B)")
    print(f"Quantization: scale={INPUT_SCALE:.6f}, zp={INPUT_ZERO_POINT}")
    print()

    results = []

    for idx, row in selected.iterrows():
        file_path = PROJECT_ROOT / row["file_path"]
        if not file_path.exists():
            print(f"  SKIP: {file_path} not found")
            continue

        # Load audio
        y, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        y = peak_normalize(y)
        y = standardize_duration(y, AUDIO_SAMPLES)

        # Reference pipeline (librosa)
        ref_db, ref_norm, ref_int8 = compute_reference_pipeline(y, mel_mean, mel_std)

        # Simulated on-device pipeline
        # Convert float32 peak-normalized audio to int16 (as Arduino would see it)
        audio_int16 = (y * 32767).astype(np.int16)

        ondev_db = compute_mel_spectrogram_ondevice(audio_int16)
        ondev_norm = normalize_spectrogram(ondev_db, mel_mean, mel_std)
        ondev_int8 = quantize_to_int8(ondev_norm, INPUT_SCALE, INPUT_ZERO_POINT)

        # Metrics
        cos_db = cosine_similarity(ref_db, ondev_db)
        cos_norm = cosine_similarity(ref_norm, ondev_norm)
        mae_db = float(np.mean(np.abs(ref_db - ondev_db)))
        max_ae_db = float(np.max(np.abs(ref_db - ondev_db)))
        mae_int8 = float(np.mean(np.abs(ref_int8.astype(np.float32) - ondev_int8.astype(np.float32))))
        max_ae_int8 = float(np.max(np.abs(ref_int8.astype(np.int32) - ondev_int8.astype(np.int32))))
        int8_match = float(np.mean(ref_int8 == ondev_int8))

        results.append({
            "file": row["filename"],
            "class": row["tier2_name"],
            "cos_db": cos_db,
            "cos_norm": cos_norm,
            "mae_db": mae_db,
            "max_ae_db": max_ae_db,
            "mae_int8": mae_int8,
            "max_ae_int8": max_ae_int8,
            "int8_match": int8_match,
        })

        status = "PASS" if cos_db >= 0.99 else "WARN" if cos_db >= 0.95 else "FAIL"
        print(f"  [{status}] {row['filename']:40s} cos={cos_db:.4f}  "
              f"mae_dB={mae_db:.2f}  int8_match={int8_match:.1%}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary ({len(results)} clips)")
    print(f"{'=' * 60}")

    cos_dbs = [r["cos_db"] for r in results]
    cos_norms = [r["cos_norm"] for r in results]
    mae_dbs = [r["mae_db"] for r in results]
    mae_int8s = [r["mae_int8"] for r in results]
    int8_matches = [r["int8_match"] for r in results]

    print(f"  Cosine similarity (dB):      {np.mean(cos_dbs):.4f} +/- {np.std(cos_dbs):.4f}  "
          f"(min: {np.min(cos_dbs):.4f})")
    print(f"  Cosine similarity (norm):    {np.mean(cos_norms):.4f} +/- {np.std(cos_norms):.4f}  "
          f"(min: {np.min(cos_norms):.4f})")
    print(f"  MAE (dB):                    {np.mean(mae_dbs):.3f} +/- {np.std(mae_dbs):.3f}  "
          f"(max: {np.max(mae_dbs):.3f})")
    print(f"  MAE (int8):                  {np.mean(mae_int8s):.2f} +/- {np.std(mae_int8s):.2f}  "
          f"(max: {np.max(mae_int8s):.2f})")
    print(f"  Int8 exact match rate:       {np.mean(int8_matches):.1%} +/- {np.std(int8_matches):.1%}")

    # Pass/fail
    min_cos = np.min(cos_dbs)
    mean_mae_int8 = np.mean(mae_int8s)
    passed = min_cos >= 0.99 and mean_mae_int8 <= 2.0

    print(f"\n  Acceptance criteria:")
    print(f"    Cosine similarity >= 0.99:  {'PASS' if min_cos >= 0.99 else 'FAIL'} (min={min_cos:.4f})")
    print(f"    Int8 MAE <= 2.0:            {'PASS' if mean_mae_int8 <= 2.0 else 'FAIL'} (mean={mean_mae_int8:.2f})")
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")

    # Save results
    results_path = PROJECT_ROOT / "results" / "feature_parity_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "clips": results,
            "summary": {
                "num_clips": len(results),
                "cosine_db_mean": float(np.mean(cos_dbs)),
                "cosine_db_min": float(np.min(cos_dbs)),
                "cosine_norm_mean": float(np.mean(cos_norms)),
                "mae_db_mean": float(np.mean(mae_dbs)),
                "mae_int8_mean": float(np.mean(mae_int8s)),
                "int8_match_mean": float(np.mean(int8_matches)),
                "passed": bool(passed),
            },
        }, f, indent=2)
    print(f"\n  Results saved: {results_path}")


if __name__ == "__main__":
    main()
