"""Data augmentation functions for waveform and spectrogram domains.

Phase 1, Task 1.8: Augmentation functions designed for on-the-fly use
during training. Not applied during preprocessing — called per-sample
in the training data pipeline.
"""

import numpy as np
import librosa


# ─── Waveform-Domain Augmentations ──────────────────────────────────────────


def time_shift(
    y: np.ndarray,
    max_shift_samples: int = 1600,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Shift waveform by a random offset, zero-padding the gap.

    Args:
        y: Audio signal.
        max_shift_samples: Maximum shift in samples (1600 = 100ms at 16kHz).
        rng: Random generator for reproducibility.

    Returns:
        Shifted audio signal (same length as input).
    """
    if rng is None:
        rng = np.random.default_rng()
    shift = rng.integers(-max_shift_samples, max_shift_samples + 1)
    result = np.zeros_like(y)
    if shift > 0:
        result[shift:] = y[:-shift]
    elif shift < 0:
        result[:shift] = y[-shift:]
    else:
        result[:] = y
    return result


def add_noise(
    y: np.ndarray,
    snr_db_range: tuple[float, float] = (5.0, 25.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian noise at a random SNR.

    Args:
        y: Audio signal.
        snr_db_range: (min_snr_db, max_snr_db).
        rng: Random generator for reproducibility.

    Returns:
        Noisy audio signal.
    """
    if rng is None:
        rng = np.random.default_rng()
    snr_db = rng.uniform(*snr_db_range)
    signal_power = np.mean(y**2)
    if signal_power == 0:
        return y.copy()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(y)) * np.sqrt(noise_power)
    return y + noise.astype(y.dtype)


def pitch_shift(
    y: np.ndarray,
    sr: int = 16000,
    n_steps_range: tuple[float, float] = (-2.0, 2.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pitch shift by a random number of semitones.

    Args:
        y: Audio signal.
        sr: Sample rate.
        n_steps_range: (min_semitones, max_semitones).
        rng: Random generator for reproducibility.

    Returns:
        Pitch-shifted audio signal.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_steps = rng.uniform(*n_steps_range)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def speed_perturb(
    y: np.ndarray,
    rate_range: tuple[float, float] = (0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Time-stretch by a random factor, then trim/pad to original length.

    Args:
        y: Audio signal.
        rate_range: (min_rate, max_rate).
        rng: Random generator for reproducibility.

    Returns:
        Speed-perturbed audio signal (same length as input).
    """
    if rng is None:
        rng = np.random.default_rng()
    rate = rng.uniform(*rate_range)
    y_stretched = librosa.effects.time_stretch(y=y, rate=rate)
    target_len = len(y)
    if len(y_stretched) > target_len:
        y_stretched = y_stretched[:target_len]
    elif len(y_stretched) < target_len:
        y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)))
    return y_stretched


def random_gain(
    y: np.ndarray,
    gain_range: tuple[float, float] = (0.7, 1.3),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Scale amplitude by a random gain factor.

    Args:
        y: Audio signal.
        gain_range: (min_gain, max_gain).
        rng: Random generator for reproducibility.

    Returns:
        Gain-adjusted audio signal.
    """
    if rng is None:
        rng = np.random.default_rng()
    gain = rng.uniform(*gain_range)
    return y * gain


# ─── Spectrogram-Domain Augmentations ───────────────────────────────────────


def frequency_mask(
    S: np.ndarray,
    F: int = 7,
    num_masks: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply SpecAugment frequency masking.

    Masks contiguous mel bands with zero.

    Args:
        S: Spectrogram of shape (n_mels, n_frames).
        F: Maximum mask width in mel bands.
        num_masks: Number of masks to apply.
        rng: Random generator for reproducibility.

    Returns:
        Masked spectrogram (copy of input).
    """
    if rng is None:
        rng = np.random.default_rng()
    S = S.copy()
    n_mels = S.shape[0]
    for _ in range(num_masks):
        f = rng.integers(0, F + 1)
        f0 = rng.integers(0, max(1, n_mels - f))
        S[f0 : f0 + f, :] = 0
    return S


def time_mask(
    S: np.ndarray,
    T: int = 10,
    num_masks: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply SpecAugment time masking.

    Masks contiguous time frames with zero.

    Args:
        S: Spectrogram of shape (n_mels, n_frames).
        T: Maximum mask width in time frames.
        num_masks: Number of masks to apply.
        rng: Random generator for reproducibility.

    Returns:
        Masked spectrogram (copy of input).
    """
    if rng is None:
        rng = np.random.default_rng()
    S = S.copy()
    n_frames = S.shape[1]
    for _ in range(num_masks):
        t = rng.integers(0, T + 1)
        t0 = rng.integers(0, max(1, n_frames - t))
        S[:, t0 : t0 + t] = 0
    return S


def spec_augment(
    S: np.ndarray,
    F: int = 7,
    T: int = 10,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply both frequency and time masking (convenience wrapper).

    Args:
        S: Spectrogram of shape (n_mels, n_frames).
        F: Maximum frequency mask width.
        T: Maximum time mask width.
        num_freq_masks: Number of frequency masks.
        num_time_masks: Number of time masks.
        rng: Random generator for reproducibility.

    Returns:
        Augmented spectrogram (copy of input).
    """
    S = frequency_mask(S, F=F, num_masks=num_freq_masks, rng=rng)
    S = time_mask(S, T=T, num_masks=num_time_masks, rng=rng)
    return S


# ─── Augmentation Pipelines ────────────────────────────────────────────────


def augment_waveform(
    y: np.ndarray,
    sr: int = 16000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a random subset of waveform augmentations.

    Each augmentation is applied with its specified probability:
        - time_shift: 50%
        - add_noise: 50%
        - pitch_shift: 30%
        - speed_perturb: 30%
        - random_gain: 50%

    Args:
        y: Audio signal.
        sr: Sample rate.
        rng: Random generator for reproducibility.

    Returns:
        Augmented audio signal.
    """
    if rng is None:
        rng = np.random.default_rng()

    y = y.copy()

    if rng.random() < 0.5:
        y = time_shift(y, rng=rng)
    if rng.random() < 0.5:
        y = add_noise(y, rng=rng)
    if rng.random() < 0.3:
        y = pitch_shift(y, sr=sr, rng=rng)
    if rng.random() < 0.3:
        y = speed_perturb(y, rng=rng)
    if rng.random() < 0.5:
        y = random_gain(y, rng=rng)

    return y


def augment_spectrogram(
    S: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply SpecAugment with 50% probability.

    Args:
        S: Spectrogram of shape (n_mels, n_frames).
        rng: Random generator for reproducibility.

    Returns:
        Augmented spectrogram.
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < 0.5:
        S = spec_augment(S, rng=rng)

    return S
