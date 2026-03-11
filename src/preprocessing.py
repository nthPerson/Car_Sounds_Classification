"""Audio loading, resampling, normalization, and duration standardization.

Phase 1, Tasks 1.1–1.3: Transform raw WAV files into fixed-length,
normalized waveforms ready for feature extraction.
"""

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─── Constants ───────────────────────────────────────────────────────────────

TARGET_SR = 16000
TARGET_DURATION_S = 1.5
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION_S)  # 24000


# ─── Core Functions ──────────────────────────────────────────────────────────


def load_and_resample(file_path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load a WAV file and resample to target_sr.

    Uses librosa's high-quality polyphase resampling filter.

    Args:
        file_path: Path to WAV file.
        target_sr: Target sample rate in Hz.

    Returns:
        Audio signal as 1-D float32 numpy array at target_sr.
    """
    y, _ = librosa.load(file_path, sr=target_sr, mono=True)
    return y


def peak_normalize(y: np.ndarray) -> np.ndarray:
    """Normalize audio to peak amplitude of 1.0.

    Preserves relative dynamics within the clip. Silent clips
    (all zeros) are returned unchanged.

    Args:
        y: Audio signal.

    Returns:
        Peak-normalized audio signal.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def standardize_duration(
    y: np.ndarray, target_samples: int = TARGET_SAMPLES
) -> np.ndarray:
    """Pad or center-crop audio to exactly target_samples.

    Longer clips are center-cropped (equal trim from both ends).
    Shorter clips are zero-padded symmetrically.

    Args:
        y: Audio signal.
        target_samples: Desired output length in samples.

    Returns:
        Fixed-length audio signal of shape (target_samples,).
    """
    if len(y) > target_samples:
        start = (len(y) - target_samples) // 2
        y = y[start : start + target_samples]
    elif len(y) < target_samples:
        pad_left = (target_samples - len(y)) // 2
        pad_right = target_samples - len(y) - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant", constant_values=0)
    return y


def preprocess_audio(file_path: str) -> np.ndarray:
    """Full preprocessing chain: load -> resample -> normalize -> standardize.

    Args:
        file_path: Path to WAV file.

    Returns:
        Preprocessed audio array of shape (24000,).
    """
    y = load_and_resample(file_path)
    y = peak_normalize(y)
    y = standardize_duration(y)
    return y


# ─── Manifest and Split Processing ──────────────────────────────────────────


def load_manifest(manifest_path: str) -> pd.DataFrame:
    """Load a split manifest CSV.

    Args:
        manifest_path: Path to the manifest CSV file.

    Returns:
        DataFrame with columns: file_path, condition, tier labels, etc.
    """
    return pd.read_csv(manifest_path)


def preprocess_split(
    manifest_path: str, project_root: str = "."
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Load and preprocess all audio files listed in a manifest.

    Args:
        manifest_path: Path to split manifest CSV.
        project_root: Root directory for resolving relative file paths.

    Returns:
        Tuple of:
        - audio_arrays: np.ndarray of shape (N, 24000)
        - labels: dict with keys 'tier1', 'tier2', 'tier3', each np.ndarray of shape (N,)
        - file_paths: list of original relative file paths
    """
    df = load_manifest(manifest_path)
    root = Path(project_root)

    audio_arrays = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing audio"):
        full_path = str(root / row["file_path"])
        y = preprocess_audio(full_path)
        audio_arrays.append(y)

    audio_arrays = np.stack(audio_arrays, axis=0).astype(np.float32)

    labels = {
        "tier1": df["tier1_label"].values.astype(np.int32),
        "tier2": df["tier2_label"].values.astype(np.int32),
        "tier3": df["tier3_label"].values.astype(np.int32),
    }

    file_paths = df["file_path"].tolist()

    return audio_arrays, labels, file_paths
