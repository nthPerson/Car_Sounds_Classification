"""Feature extraction: mel-spectrograms, MFCCs, and classical ML features.

Phase 1, Tasks 1.5–1.7, 1.9–1.10: Extract model-ready features from
preprocessed audio waveforms.
"""

import numpy as np
import librosa
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# ─── Feature Parameters (must match data_preprocessing_spec.md exactly) ──────

MEL_PARAMS = {
    "n_fft": 512,
    "hop_length": 256,
    "n_mels": 40,
    "fmin": 20,
    "fmax": 8000,
    "power": 2.0,
    "window": "hann",
}

MFCC_N_COEFFS = 13

# Classical ML feature dimensions
N_CLASSICAL_FRAME_FEATURES = 63  # per frame
N_AGG_STATS = 7
N_CLASSICAL_FEATURES = N_CLASSICAL_FRAME_FEATURES * N_AGG_STATS  # 441

# Feature ordering for classical ML (for feature importance mapping)
CLASSICAL_FEATURE_NAMES = (
    [f"mfcc_{i}" for i in range(13)]
    + [f"delta_mfcc_{i}" for i in range(13)]
    + [f"delta2_mfcc_{i}" for i in range(13)]
    + ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]
    + [f"spectral_contrast_{i}" for i in range(7)]
    + ["rms", "zcr"]
    + [f"chroma_{i}" for i in range(12)]
)

AGG_STAT_NAMES = ["mean", "std", "min", "max", "median", "skewness", "kurtosis"]


# ─── Mel-Spectrogram ────────────────────────────────────────────────────────


def compute_log_mel_spectrogram(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute log-mel spectrogram.

    Args:
        y: Audio signal (1-D, 16 kHz).
        sr: Sample rate.

    Returns:
        Log-mel spectrogram in dB, shape (40, 92).
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=MEL_PARAMS["n_fft"],
        hop_length=MEL_PARAMS["hop_length"],
        n_mels=MEL_PARAMS["n_mels"],
        fmin=MEL_PARAMS["fmin"],
        fmax=MEL_PARAMS["fmax"],
        power=MEL_PARAMS["power"],
        window=MEL_PARAMS["window"],
        center=False,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


# ─── MFCCs ──────────────────────────────────────────────────────────────────


def compute_mfcc(
    y: np.ndarray, sr: int = 16000, include_deltas: bool = False
) -> np.ndarray:
    """Compute MFCCs, optionally with delta and delta-delta.

    Args:
        y: Audio signal (1-D, 16 kHz).
        sr: Sample rate.
        include_deltas: If True, append delta and delta-delta MFCCs.

    Returns:
        MFCCs of shape (92, 13) or (92, 39) if include_deltas=True.
    """
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=MFCC_N_COEFFS,
        n_fft=MEL_PARAMS["n_fft"],
        hop_length=MEL_PARAMS["hop_length"],
        n_mels=MEL_PARAMS["n_mels"],
        fmin=MEL_PARAMS["fmin"],
        fmax=MEL_PARAMS["fmax"],
        center=False,
    )  # shape: (13, 92)

    if include_deltas:
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        mfccs = np.vstack([mfccs, delta, delta2])  # shape: (39, 92)

    return mfccs.T  # shape: (92, 13) or (92, 39)


# ─── Classical ML Features ──────────────────────────────────────────────────


def _aggregate_frame_features(features: np.ndarray) -> np.ndarray:
    """Aggregate frame-level features into a fixed-length vector.

    Computes 7 statistics per feature across the time axis:
    mean, std, min, max, median, skewness, kurtosis.

    Args:
        features: Frame-level features, shape (n_features, n_frames).

    Returns:
        Aggregated vector, shape (n_features * 7,).
    """
    stats = np.stack(
        [
            np.mean(features, axis=1),
            np.std(features, axis=1),
            np.min(features, axis=1),
            np.max(features, axis=1),
            np.median(features, axis=1),
            skew(features, axis=1, nan_policy="omit"),
            kurtosis(features, axis=1, nan_policy="omit"),
        ],
        axis=1,
    )  # shape: (n_features, 7)
    # Replace any NaN from degenerate distributions with 0.0
    np.nan_to_num(stats, copy=False, nan=0.0)
    return stats.flatten()  # shape: (n_features * 7,)


def compute_classical_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute 441-dim classical ML feature vector.

    Extracts 63 frame-level features, aggregates each with 7 statistics.

    Frame-level features (63 total):
        [0:13]  MFCCs (13 coefficients)
        [13:26] Delta MFCCs (13)
        [26:39] Delta-delta MFCCs (13)
        [39]    Spectral centroid
        [40]    Spectral bandwidth
        [41]    Spectral rolloff (85th percentile)
        [42:49] Spectral contrast (7 bands)
        [49]    RMS energy
        [50]    Zero-crossing rate
        [51:63] Chroma features (12 pitch classes)

    Args:
        y: Audio signal (1-D, 16 kHz).
        sr: Sample rate.

    Returns:
        Feature vector of shape (441,).
    """
    n_fft = MEL_PARAMS["n_fft"]
    hop_length = MEL_PARAMS["hop_length"]

    # MFCCs with deltas (39 features)
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=MFCC_N_COEFFS,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=MEL_PARAMS["n_mels"],
        fmin=MEL_PARAMS["fmin"],
        fmax=MEL_PARAMS["fmax"],
        center=False,
    )  # (13, n_frames)
    delta_mfccs = librosa.feature.delta(mfccs)  # (13, n_frames)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)  # (13, n_frames)

    # Spectral features (3 features)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, center=False
    )  # (1, n_frames)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, center=False
    )  # (1, n_frames)
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85, center=False
    )  # (1, n_frames)

    # Spectral contrast (7 features)
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, center=False
    )  # (7, n_frames)

    # Energy and temporal features (2 features)
    rms = librosa.feature.rms(
        y=y, frame_length=n_fft, hop_length=hop_length, center=False
    )  # (1, n_frames)
    zcr = librosa.feature.zero_crossing_rate(
        y=y, frame_length=n_fft, hop_length=hop_length, center=False
    )  # (1, n_frames)

    # Chroma features (12 features)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, center=False
    )  # (12, n_frames)

    # Stack all frame-level features: (63, n_frames)
    all_features = np.vstack(
        [
            mfccs,              # 13
            delta_mfccs,        # 13
            delta2_mfccs,       # 13
            spectral_centroid,  # 1
            spectral_bandwidth, # 1
            spectral_rolloff,   # 1
            spectral_contrast,  # 7
            rms,                # 1
            zcr,                # 1
            chroma,             # 12
        ]
    )  # (63, n_frames)

    return _aggregate_frame_features(all_features)  # (441,)


# ─── Batch Feature Extraction ───────────────────────────────────────────────


def extract_mel_spectrograms(
    audio_arrays: np.ndarray, sr: int = 16000
) -> np.ndarray:
    """Extract log-mel spectrograms for a batch of audio clips.

    Args:
        audio_arrays: Shape (N, 24000).
        sr: Sample rate.

    Returns:
        Mel-spectrograms, shape (N, 40, 92).
    """
    specs = []
    for i in tqdm(range(len(audio_arrays)), desc="Extracting mel-spectrograms"):
        spec = compute_log_mel_spectrogram(audio_arrays[i], sr=sr)
        specs.append(spec)
    return np.stack(specs, axis=0).astype(np.float32)


def extract_mfccs(audio_arrays: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract MFCCs for a batch of audio clips.

    Args:
        audio_arrays: Shape (N, 24000).
        sr: Sample rate.

    Returns:
        MFCCs, shape (N, 92, 13).
    """
    mfccs = []
    for i in tqdm(range(len(audio_arrays)), desc="Extracting MFCCs"):
        mfcc = compute_mfcc(audio_arrays[i], sr=sr)
        mfccs.append(mfcc)
    return np.stack(mfccs, axis=0).astype(np.float32)


def extract_classical_features(
    audio_arrays: np.ndarray, sr: int = 16000
) -> np.ndarray:
    """Extract classical ML feature vectors for a batch of audio clips.

    Args:
        audio_arrays: Shape (N, 24000).
        sr: Sample rate.

    Returns:
        Feature vectors, shape (N, 441).
    """
    features = []
    for i in tqdm(range(len(audio_arrays)), desc="Extracting classical features"):
        feat = compute_classical_features(audio_arrays[i], sr=sr)
        features.append(feat)
    return np.stack(features, axis=0).astype(np.float32)


# ─── Normalization ──────────────────────────────────────────────────────────


def compute_normalization_stats(
    X_train: np.ndarray, feature_type: str
) -> dict[str, np.ndarray]:
    """Compute per-channel mean and std from training set.

    Args:
        X_train: Training features.
            mel: shape (N, 40, 92) — stats computed per mel band.
            mfcc: shape (N, 92, 13) — stats computed per MFCC coefficient.
            classical: shape (N, 441) — stats computed per feature dimension.
        feature_type: One of 'mel', 'mfcc', 'classical'.

    Returns:
        Dict with 'mean' and 'std' arrays (float32).
    """
    eps = 1e-6

    if feature_type == "mel":
        # Mean/std per mel band, across all samples and time frames
        mean = np.mean(X_train, axis=(0, 2)).astype(np.float32)  # (40,)
        std = np.std(X_train, axis=(0, 2)).astype(np.float32) + eps  # (40,)
    elif feature_type == "mfcc":
        # Mean/std per MFCC coefficient, across all samples and time frames
        mean = np.mean(X_train, axis=(0, 1)).astype(np.float32)  # (13,)
        std = np.std(X_train, axis=(0, 1)).astype(np.float32) + eps  # (13,)
    elif feature_type == "classical":
        # Mean/std per feature dimension
        mean = np.mean(X_train, axis=0).astype(np.float32)  # (441,)
        std = np.std(X_train, axis=0).astype(np.float32) + eps  # (441,)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return {"mean": mean, "std": std}


def normalize_features(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray, feature_type: str
) -> np.ndarray:
    """Apply z-score normalization using precomputed stats.

    Args:
        X: Features to normalize.
        mean: Per-channel mean from training set.
        std: Per-channel std from training set (with epsilon).
        feature_type: One of 'mel', 'mfcc', 'classical'.

    Returns:
        Normalized features (same shape as X).
    """
    if feature_type == "mel":
        # Reshape for broadcasting: (40,) -> (1, 40, 1)
        return (X - mean[np.newaxis, :, np.newaxis]) / std[np.newaxis, :, np.newaxis]
    elif feature_type == "mfcc":
        # Reshape for broadcasting: (13,) -> (1, 1, 13)
        return (X - mean[np.newaxis, np.newaxis, :]) / std[np.newaxis, np.newaxis, :]
    elif feature_type == "classical":
        # Reshape for broadcasting: (441,) -> (1, 441)
        return (X - mean[np.newaxis, :]) / std[np.newaxis, :]
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")
