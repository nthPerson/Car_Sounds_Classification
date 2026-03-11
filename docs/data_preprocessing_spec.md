# Data Preprocessing Specification

**Parent document:** [project_overview.md](project_overview.md)
**Date:** March 10, 2026

---

## Table of Contents

1. [Raw Dataset Description](#1-raw-dataset-description)
2. [Preprocessing Pipeline Overview](#2-preprocessing-pipeline-overview)
3. [Step 1 — Audio Loading and Resampling](#3-step-1--audio-loading-and-resampling)
4. [Step 2 — Amplitude Normalization](#4-step-2--amplitude-normalization)
5. [Step 3 — Duration Standardization](#5-step-3--duration-standardization)
6. [Step 4 — Train/Validation/Test Split](#6-step-4--trainvalidationtest-split)
7. [Step 5 — Feature Extraction](#7-step-5--feature-extraction)
8. [Step 6 — Data Augmentation](#8-step-6--data-augmentation)
9. [Step 7 — Feature Normalization](#9-step-7--feature-normalization)
10. [Label Encoding and Class Mappings](#10-label-encoding-and-class-mappings)
11. [On-Device Feature Extraction Parity](#11-on-device-feature-extraction-parity)
12. [Preprocessing Outputs and Artifacts](#12-preprocessing-outputs-and-artifacts)
13. [Implementation Notes](#13-implementation-notes)

---

## 1. Raw Dataset Description

### 1.1 Directory Structure

The Car Diagnostics Dataset is organized by operational state, then by condition:

```
car_diagnostics_dataset/
├── braking state/
│   ├── normal_brakes/          (77 WAV files)
│   └── worn_out_brakes/        (76 WAV files)
├── idle state/
│   ├── normal_engine_idle/     (264 WAV files)
│   ├── low_oil/                (107 WAV files)
│   ├── power_steering/         (129 WAV files)
│   ├── serpentine_belt/        (116 WAV files)
│   └── combined/
│       ├── no oil_serpentine belt/                              (107 WAV files)
│       ├── power steering combined_no oil/                     (107 WAV files)
│       ├── power steering combined_no oil_serpentine belt/     (107 WAV files)
│       └── power steering combined_serpentine belt/            (116 WAV files)
└── startup state/
    ├── normal_engine_startup/  (61 WAV files)
    ├── bad_ignition/           (62 WAV files)
    └── dead_battery/           (57 WAV files)
```

**Total: 1,386 WAV files across 3 operational states and 13 conditions.**

### 1.2 Audio Properties (from EDA)

| Property | Value |
|----------|-------|
| Sample rates | 24,000 Hz (some files), 44,100 Hz (some files), 48,000 Hz (some files) — **mixed** |
| Channels | 1 (mono, all files) |
| Bit depth | 16-bit PCM |
| Duration — Braking | 1.500 s (constant across all braking files) |
| Duration — Idle | 1.500 s (constant across all idle files) |
| Duration — Start-Up | 1.700 s (constant across all start-up files) |
| File format | WAV (uncompressed) |

### 1.3 Class Distribution

| Condition | State | Count | % of Total |
|-----------|-------|-------|-----------|
| `normal_engine_idle` | Idle | 264 | 19.0% |
| `power_steering` | Idle | 129 | 9.3% |
| `serpentine_belt` | Idle | 116 | 8.4% |
| `combined/power steering combined_serpentine belt` | Idle | 116 | 8.4% |
| `low_oil` | Idle | 107 | 7.7% |
| `combined/no oil_serpentine belt` | Idle | 107 | 7.7% |
| `combined/power steering combined_no oil` | Idle | 107 | 7.7% |
| `combined/power steering combined_no oil_serpentine belt` | Idle | 107 | 7.7% |
| `normal_brakes` | Braking | 77 | 5.6% |
| `worn_out_brakes` | Braking | 76 | 5.5% |
| `bad_ignition` | Start-Up | 62 | 4.5% |
| `normal_engine_startup` | Start-Up | 61 | 4.4% |
| `dead_battery` | Start-Up | 57 | 4.1% |

**Imbalance ratio (largest to smallest class):** 264 / 57 = **4.63:1**

### 1.4 Feature Separability (from EDA)

The exploratory data analysis notebook (`explore_dataset.ipynb`) computed four summary features for each file: RMS energy, zero-crossing rate (ZCR), spectral centroid, and spectral bandwidth. Key observations:

- **Braking:** Moderate separation between normal and worn-out brakes on RMS energy and spectral bandwidth. Worn brakes show higher mean ZCR (0.169 vs. 0.093) and higher spectral centroid (2332 Hz vs. 2037 Hz), suggesting more high-frequency content (consistent with squealing/grinding).
- **Idle:** Strong separation. `serpentine_belt` has dramatically higher ZCR (0.337 vs. 0.112 for normal idle) and spectral centroid (4285 Hz vs. 2466 Hz), indicating a distinctive high-frequency whine. Combined-fault categories cluster between single-fault and normal values. `normal_engine_idle` has the lowest RMS energy among idle conditions.
- **Start-Up:** `dead_battery` has notably lower RMS energy (0.033 vs. 0.108 for normal startup), consistent with a weak or absent engine crank. `bad_ignition` and `normal_engine_startup` have more similar RMS but differ in ZCR and spectral centroid.

These results confirm that spectral and temporal features carry discriminative information even in their simplest aggregate form, which is encouraging for richer representations (mel-spectrograms, MFCCs) used in deep learning.

---

## 2. Preprocessing Pipeline Overview

The preprocessing pipeline transforms raw WAV files into model-ready feature tensors. Every step is designed to be reproducible and deterministic (except augmentation, which uses seeded randomness).

```
Raw WAV files (1,386 files, mixed sample rates, 1.5–1.7 s)
    │
    ├── [1] Audio Loading & Resampling → 16 kHz mono PCM
    │
    ├── [2] Amplitude Normalization → peak or RMS normalization
    │
    ├── [3] Duration Standardization → pad/trim to exactly T samples
    │
    ├── [4] Train/Val/Test Split → stratified by condition (70/15/15)
    │        (split happens BEFORE augmentation to prevent data leakage)
    │
    ├── [5] Feature Extraction → mel-spectrograms and/or MFCCs
    │
    ├── [6] Data Augmentation → applied ONLY to training set
    │        (augmentation in waveform domain before feature extraction,
    │         and/or in spectrogram domain after feature extraction)
    │
    ├── [7] Feature Normalization → per-channel mean/std normalization
    │        (statistics computed on training set only, applied to all sets)
    │
    └── Output: Feature tensors + labels for train/val/test
```

**Critical design principle:** The train/validation/test split is performed BEFORE any augmentation. Augmented samples are generated ONLY from training set originals. Validation and test sets contain only original (unaugmented) samples to ensure unbiased evaluation.

---

## 3. Step 1 — Audio Loading and Resampling

### 3.1 Rationale

The dataset contains files at three different sample rates (24 kHz, 44.1 kHz, 48 kHz). All files must be resampled to a common rate before feature extraction to ensure consistent frequency resolution across the dataset.

### 3.2 Target Sample Rate: 16 kHz

**Why 16 kHz:**
- **Matches the on-device microphone.** The Arduino Nano 33 BLE Sense Rev2's PDM microphone is practically operated at 16 kHz in TinyML applications. Training at the same rate eliminates a source of training-serving skew.
- **Sufficient frequency range.** At 16 kHz sample rate, the Nyquist frequency is 8 kHz. Engine sounds have their dominant energy below 5 kHz, with harmonics extending to ~8 kHz. Braking sounds (squealing, grinding) also have primary energy below 8 kHz.
- **Reduces computation.** Fewer samples per clip means smaller FFT windows, fewer spectrogram frames, and faster on-device processing.
- **Industry precedent.** 16 kHz is the standard for TinyML audio classification (e.g., TensorFlow's micro_speech example, Edge Impulse's default for keyword spotting).

### 3.3 Implementation

```python
import librosa

TARGET_SR = 16000

def load_and_resample(file_path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and resample to TARGET_SR."""
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    return y, TARGET_SR
```

`librosa.load` with `sr=TARGET_SR` handles resampling internally using a high-quality polyphase filter (via `resampy` or `soxr`). This is equivalent to what `scipy.signal.resample_poly` would produce.

### 3.4 Verification

After resampling, every file should have:
- Sample rate: 16,000 Hz
- Channels: 1
- Samples: 24,000 (for 1.5 s clips) or 27,200 (for 1.7 s clips)

These values should be programmatically verified for every file before proceeding.

---

## 4. Step 2 — Amplitude Normalization

### 4.1 Rationale

Audio files in the dataset may have been recorded at different gain levels, and different sample rates can subtly affect amplitude after resampling. Normalization ensures that the model learns from spectral shape rather than absolute loudness, improving generalization.

### 4.2 Method: Peak Normalization

We normalize each clip so that its maximum absolute amplitude equals 1.0:

```python
def peak_normalize(y: np.ndarray) -> np.ndarray:
    """Normalize audio to peak amplitude of 1.0."""
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y
```

**Why peak normalization over RMS normalization:**
- Peak normalization is simpler and more predictable — every clip's dynamic range is maximally utilized
- It preserves the relative dynamics within a clip (quiet parts stay quiet relative to loud parts)
- RMS normalization would boost quiet clips (like `dead_battery`) to the same loudness as loud clips, potentially destroying a discriminative signal. The EDA showed that `dead_battery` has notably lower RMS energy (0.033 vs. 0.108 for normal startup) — this amplitude difference is itself a useful feature

**Alternative considered — no normalization:** We could also skip normalization entirely and let the model learn to handle varying amplitudes. This is a valid approach, especially if we plan to use augmentation that varies the gain. We should experiment with both approaches.

### 4.3 On-Device Consideration

On the Arduino, the PDM library delivers 16-bit signed integer samples. The amplitude depends on the microphone's sensitivity and the sound source's loudness. We do NOT normalize on-device (it would require a full-buffer scan before processing). Instead, the model must be robust to amplitude variation, which we achieve through gain augmentation during training (see Section 8).

---

## 5. Step 3 — Duration Standardization

### 5.1 Rationale

Feature extraction (especially mel-spectrograms) produces fixed-size tensors only if the input waveform has a fixed length. The dataset has two distinct durations: 1.5 s (braking, idle) and 1.7 s (start-up).

### 5.2 Target Duration: 1.5 seconds (24,000 samples at 16 kHz)

**Why 1.5 s and not 1.7 s:**
- 1.5 s is the duration of the majority (87%) of the dataset
- A shorter window uses less SRAM on-device (48 KB vs. 54.4 KB for the audio buffer)
- Start-up clips at 1.7 s can be trimmed to 1.5 s from the center or the beginning. Engine start-up events typically begin within the first 0.5 s, so 1.5 s still captures the relevant content
- Alternatively, we can zero-pad 1.5 s clips to 1.7 s to match start-up duration — but this wastes memory on-device for 87% of use cases

**Handling the 0.2 s difference:**
- **Start-up clips (1.7 s → 1.5 s):** Trim 0.1 s from each end (center crop), or trim 0.2 s from the end. Center cropping is preferred to avoid edge effects from the start and end of the recording.
- **If we choose 1.7 s instead:** Braking and idle clips would need 0.2 s of zero-padding, which adds a silent region that the model must learn to ignore. This is less clean but preserves all start-up audio.

**Recommendation:** Use **1.5 s** as the standard duration. Center-crop start-up clips. This keeps the on-device buffer at 48 KB and avoids introducing artificial silence.

### 5.3 Implementation

```python
TARGET_DURATION = 1.5  # seconds
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 24,000

def standardize_duration(y: np.ndarray, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
    """Pad or center-crop audio to exactly target_samples."""
    if len(y) > target_samples:
        # Center crop
        start = (len(y) - target_samples) // 2
        y = y[start : start + target_samples]
    elif len(y) < target_samples:
        # Zero-pad symmetrically
        pad_left = (target_samples - len(y)) // 2
        pad_right = target_samples - len(y) - pad_left
        y = np.pad(y, (pad_left, pad_right), mode='constant', constant_values=0)
    return y
```

---

## 6. Step 4 — Train/Validation/Test Split

### 6.1 Split Ratios

| Set | Ratio | Purpose |
|-----|-------|---------|
| **Training** | 70% | Model training (augmentation applied only here) |
| **Validation** | 15% | Hyperparameter tuning, early stopping, model selection |
| **Test** | 15% | Final held-out evaluation (never seen during training or tuning) |

### 6.2 Stratification

The split must be **stratified by condition** (the finest-grained label) to ensure that each condition is proportionally represented in all three sets. This is critical because:
- The dataset is imbalanced (264 vs. 57 files per class)
- Some conditions have very few samples (e.g., `dead_battery` with 57 files → only ~8 test samples at 15%)
- Without stratification, a random split could leave some conditions severely under-represented in the test set

```python
from sklearn.model_selection import train_test_split

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    file_paths, labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# Second split: 50% of temp → 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
```

### 6.3 Expected Split Sizes (for 13-class Tier 3)

| Condition | Total | Train (70%) | Val (15%) | Test (15%) |
|-----------|-------|------------|-----------|------------|
| `normal_engine_idle` | 264 | 184 | 40 | 40 |
| `power_steering` | 129 | 90 | 19 | 20 |
| `serpentine_belt` | 116 | 81 | 17 | 18 |
| `low_oil` | 107 | 74 | 16 | 17 |
| `combined/*` (each ~107–116) | ~437 | ~306 | ~65 | ~66 |
| `normal_brakes` | 77 | 53 | 12 | 12 |
| `worn_out_brakes` | 76 | 53 | 11 | 12 |
| `bad_ignition` | 62 | 43 | 9 | 10 |
| `normal_engine_startup` | 61 | 42 | 9 | 10 |
| `dead_battery` | 57 | 39 | 9 | 9 |
| **Total** | **1,386** | **~970** | **~208** | **~208** |

> **Note:** For Tier 1 (binary) and Tier 2 (6-class), the SAME split is used — only the label mapping changes. This ensures that exactly the same audio files are in the training, validation, and test sets across all tiers, enabling fair comparison.

### 6.4 Split Persistence

The split must be saved to disk (e.g., as a CSV or JSON manifest) so that:
- All team members use the same split
- The split is reproducible even if the random seed or scikit-learn version changes
- The split can be referenced in the final report

```
data/splits/
├── train_manifest.csv   # columns: file_path, condition, state, tier1_label, tier2_label, tier3_label
├── val_manifest.csv
└── test_manifest.csv
```

---

## 7. Step 5 — Feature Extraction

Two feature representations will be used: **mel-spectrograms** (primary, for CNN models) and **MFCCs** (for 1-D CNN and classical ML baselines). Both are derived from the Short-Time Fourier Transform (STFT).

### 7.1 Mel-Spectrogram — Primary Feature for 2-D CNNs

A mel-spectrogram is a time-frequency representation where the frequency axis is warped to the mel scale, which approximates human auditory perception and compresses the frequency axis into a manageable number of bands.

#### 7.1.1 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Sample rate** | 16,000 Hz | Matches on-device mic and training data |
| **n_fft** | 512 | FFT window size in samples. 512 / 16000 = 32 ms window, providing good frequency resolution for engine sounds (31.25 Hz per bin). Must be a power of 2 for efficient FFT on the Cortex-M4. |
| **hop_length** | 256 | Hop between successive frames in samples. 256 / 16000 = 16 ms hop, giving 50% overlap with the FFT window. Balances time resolution against the number of output frames. |
| **n_mels** | 40 | Number of mel-frequency bands. 40 is standard for TinyML audio classification (used in TF micro_speech, Edge Impulse defaults). Provides sufficient spectral detail for engine sounds while keeping the input tensor small. |
| **fmin** | 20 Hz | Minimum frequency for the mel filterbank. Engine idle fundamentals can be as low as ~25-30 Hz. |
| **fmax** | 8,000 Hz | Maximum frequency (= Nyquist at 16 kHz). Captures the full available spectrum. |
| **power** | 2.0 | Power spectrogram (magnitude squared). Standard for mel-spectrograms. |
| **Window function** | Hann | Standard for STFT. Minimizes spectral leakage. |

#### 7.1.2 Output Dimensions

With 1.5 s audio at 16 kHz (24,000 samples):
- Number of frames: floor((24000 - 512) / 256) + 1 = **92 frames**
- Mel bands: **40**
- Output tensor shape: **(40, 92, 1)** — mel bands × time frames × 1 channel

Total values: 40 × 92 = 3,680 values per sample. At int8 (1 byte each), the model input is ~3.6 KB.

#### 7.1.3 Log-Mel Spectrogram

The raw mel-spectrogram is converted to log scale (decibels) before being fed to the model:

```python
import librosa

def compute_log_mel_spectrogram(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute log-mel spectrogram."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=512,
        hop_length=256,
        n_mels=40,
        fmin=20,
        fmax=8000,
        power=2.0,
        window='hann'
    )
    S_dB = librosa.power_to_db(S, ref=np.max)  # dB scale, max = 0 dB
    return S_dB  # shape: (40, 92)
```

The `ref=np.max` normalization sets the loudest bin to 0 dB, with all other values negative (typically ranging from ~-80 to 0 dB). This per-clip normalization provides some amplitude invariance.

#### 7.1.4 Alternative Parameter Set (Reduced)

If the model's tensor arena doesn't fit in SRAM, we can reduce the input dimensions:

| Parameter | Reduced Value | Output Shape |
|-----------|--------------|-------------|
| n_mels | 32 | (32, 92, 1) |
| hop_length | 320 (20 ms) | (40, 74, 1) |
| Both | — | (32, 74, 1) = 2,368 values |

This reduces the input tensor by ~36% at the cost of some spectral and temporal resolution. Engine sounds are broadband and relatively stationary, so this reduction is unlikely to cause significant accuracy loss.

### 7.2 MFCCs — Feature for 1-D CNN and Classical ML

MFCCs (Mel-Frequency Cepstral Coefficients) are derived from the mel-spectrogram by applying a log transform and then a Discrete Cosine Transform (DCT). They produce a compact, decorrelated representation.

#### 7.2.1 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **n_mfcc** | 13 | Standard number of MFCC coefficients. Captures the spectral envelope shape. |
| **n_fft, hop_length, n_mels, fmin, fmax** | Same as mel-spectrogram | Consistency with mel-spectrogram pipeline |
| **Include deltas** | Yes (Δ and ΔΔ) | Delta (first derivative) and delta-delta (second derivative) MFCCs capture temporal dynamics. Total: 13 × 3 = 39 coefficients per frame. Used for classical ML feature vectors. |

#### 7.2.2 Output Dimensions

- Per-frame: 13 MFCCs (or 39 with deltas)
- Number of frames: 92 (same as mel-spectrogram)
- For 1-D CNN input: shape **(92, 13)** — time frames × MFCC coefficients
- For classical ML: aggregate statistics over frames (see Section 7.3)

```python
def compute_mfccs(y: np.ndarray, sr: int = 16000, include_deltas: bool = False) -> np.ndarray:
    """Compute MFCCs, optionally with delta and delta-delta."""
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=13,
        n_fft=512,
        hop_length=256,
        n_mels=40,
        fmin=20,
        fmax=8000
    )  # shape: (13, 92)

    if include_deltas:
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        mfccs = np.vstack([mfccs, delta, delta2])  # shape: (39, 92)

    return mfccs.T  # shape: (92, 13) or (92, 39)
```

### 7.3 Feature Vectors for Classical ML Baselines

Classical ML models (Random Forest, SVM) require fixed-length feature vectors, not 2-D spectrograms. We derive these by computing summary statistics over the time axis of frame-level features:

#### 7.3.1 Frame-Level Features

For each audio clip, compute the following frame-level features:
- **MFCCs** (13 coefficients) + Δ-MFCCs (13) + ΔΔ-MFCCs (13) = 39 features/frame
- **Spectral centroid** (1 feature/frame)
- **Spectral bandwidth** (1 feature/frame)
- **Spectral rolloff** (1 feature/frame, at 85th percentile)
- **Spectral contrast** (7 bands → 7 features/frame)
- **RMS energy** (1 feature/frame)
- **Zero-crossing rate** (1 feature/frame)
- **Chroma features** (12 pitch classes → 12 features/frame)

Total frame-level features: 39 + 1 + 1 + 1 + 7 + 1 + 1 + 12 = **63 features per frame**

#### 7.3.2 Aggregation Statistics

For each of the 63 frame-level features, compute over all frames:
- **Mean**
- **Standard deviation**
- **Minimum**
- **Maximum**
- **Median**
- **Skewness**
- **Kurtosis**

Total feature vector length: 63 × 7 = **441 features per clip**

This produces a rich, fixed-length representation suitable for Random Forest and SVM classifiers. The high dimensionality relative to the small dataset size (1,386 samples) means regularization will be important — Random Forest handles this naturally, and SVM can use an RBF kernel with appropriate C/gamma tuning.

---

## 8. Step 6 — Data Augmentation

### 8.1 Rationale

With only ~970 training samples (after the 70% split), overfitting is a major risk. Data augmentation artificially expands the effective training set by applying realistic transformations to existing samples. Augmentation is applied **ONLY to the training set** and is performed **on-the-fly** during training (preferred) or as a pre-processing expansion step.

### 8.2 Waveform-Domain Augmentations

These augmentations are applied to the raw audio waveform BEFORE feature extraction. They simulate real-world recording variability.

#### 8.2.1 Time Shift

Shift the waveform forward or backward in time by a random offset, wrapping around or zero-padding the edges.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max shift | ±1,600 samples (±100 ms) | Engine sounds are relatively stationary; shifting by up to 100 ms simulates slight timing variations in recording onset |
| Padding | Zero-padding | Wrapping could create discontinuities |

```python
def time_shift(y: np.ndarray, max_shift: int = 1600) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(y, shift)
```

#### 8.2.2 Background Noise Injection

Add random noise to simulate real-world recording conditions (road noise, wind, other vehicles, environmental sound).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Noise type | Gaussian white noise, optionally pink noise | Simulates broadband environmental noise |
| SNR range | 5–25 dB | 5 dB is quite noisy; 25 dB is nearly clean. This wide range forces the model to be robust to varying noise levels |
| Probability | 50% per training sample | Not every sample should be noisy |

```python
def add_noise(y: np.ndarray, snr_db: float = None) -> np.ndarray:
    if snr_db is None:
        snr_db = np.random.uniform(5, 25)
    noise = np.random.randn(len(y))
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(noise_power)
    return y + noise
```

**Domain gap note:** The dataset was likely recorded in controlled/studio conditions. Adding noise augmentation is critical for bridging the gap to real-world deployment, where the Arduino's microphone will capture engine sounds mixed with ambient noise. Consider using real-world ambient noise recordings (urban sounds, road noise) from freely available datasets (e.g., ESC-50, UrbanSound8K) as noise sources for a more realistic augmentation.

#### 8.2.3 Pitch Perturbation

Shift the pitch of the audio up or down without changing duration, simulating different engine RPMs or recording conditions.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pitch shift range | ±2 semitones (±~12%) | ±2 semitones covers a reasonable range of RPM variation without making the audio sound unnatural |
| Probability | 30% per training sample | Moderate application rate |

```python
def pitch_shift(y: np.ndarray, sr: int = 16000, n_steps: float = None) -> np.ndarray:
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
```

#### 8.2.4 Speed Perturbation (Time Stretch)

Change the playback speed of the audio, which alters both tempo and pitch. This is distinct from pitch shifting (which preserves duration).

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Speed factor range | 0.9–1.1 (±10%) | Small speed variations simulate different engine speeds |
| Post-stretch | Trim/pad to original length | Maintain fixed input size |

```python
def speed_perturb(y: np.ndarray, rate: float = None) -> np.ndarray:
    if rate is None:
        rate = np.random.uniform(0.9, 1.1)
    y_stretched = librosa.effects.time_stretch(y=y, rate=rate)
    # Trim or pad to original length
    target_len = len(y)
    if len(y_stretched) > target_len:
        y_stretched = y_stretched[:target_len]
    else:
        y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)))
    return y_stretched
```

#### 8.2.5 Random Gain

Adjust the overall amplitude of the clip by a random factor, simulating varying recording distances and microphone sensitivity differences.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gain range | 0.7–1.3 (in linear scale) | ±30% amplitude variation. This is especially important because on-device audio will have unpredictable amplitude depending on mic placement relative to engine |
| Probability | 50% per training sample | Frequent application to build amplitude robustness |

```python
def random_gain(y: np.ndarray, gain_range: tuple = (0.7, 1.3)) -> np.ndarray:
    gain = np.random.uniform(*gain_range)
    return y * gain
```

### 8.3 Spectrogram-Domain Augmentations (SpecAugment)

These augmentations are applied to the mel-spectrogram AFTER feature extraction. They are computationally cheap and highly effective for audio classification.

#### 8.3.1 Frequency Masking

Randomly zero out a contiguous block of frequency (mel) bands.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max mask width (F) | 7 mel bands (out of 40) | Masks up to ~17% of the frequency range, forcing the model to not rely on any single frequency region |
| Number of masks | 1–2 | Standard for small spectrograms |

#### 8.3.2 Time Masking

Randomly zero out a contiguous block of time frames.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max mask width (T) | 10 frames (out of 92) | Masks up to ~11% of the time range |
| Number of masks | 1–2 | Standard for small spectrograms |

```python
def spec_augment(S: np.ndarray, freq_mask_param: int = 7, time_mask_param: int = 10,
                 num_freq_masks: int = 1, num_time_masks: int = 1) -> np.ndarray:
    """Apply SpecAugment (frequency and time masking) to a spectrogram."""
    S = S.copy()
    n_mels, n_frames = S.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        S[f0 : f0 + f, :] = 0  # or the minimum dB value

    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_param + 1)
        t0 = np.random.randint(0, max(1, n_frames - t))
        S[:, t0 : t0 + t] = 0  # or the minimum dB value

    return S
```

### 8.4 Augmentation Strategy Summary

| Augmentation | Domain | Probability | Applied When |
|-------------|--------|------------|-------------|
| Time shift | Waveform | 50% | Before feature extraction |
| Noise injection | Waveform | 50% | Before feature extraction |
| Pitch perturbation | Waveform | 30% | Before feature extraction |
| Speed perturbation | Waveform | 30% | Before feature extraction |
| Random gain | Waveform | 50% | Before feature extraction |
| Frequency masking (SpecAugment) | Spectrogram | 50% | After feature extraction |
| Time masking (SpecAugment) | Spectrogram | 50% | After feature extraction |

**On-the-fly vs. pre-computed augmentation:**
- **On-the-fly (preferred):** Apply random augmentations during each training epoch. This means the model sees different variations every epoch, effectively creating an infinite training set. Implemented via a TensorFlow `tf.data.Dataset` pipeline or a Keras `Sequence` generator.
- **Pre-computed:** Generate N augmented copies of each training sample and save to disk. Simpler to implement but produces a fixed augmentation set. If used, aim for 5–10× expansion (4,850–9,700 augmented training samples).

### 8.5 Class-Specific Augmentation Intensity

To address class imbalance, apply MORE augmentation to minority classes. Specifically:

| Class Size Bucket | Pre-Computed Multiplier | On-the-fly Oversampling Factor |
|-------------------|------------------------|-------------------------------|
| Large (>150 samples, e.g., `normal_engine_idle`) | 3× | 1× (no oversampling) |
| Medium (75–150 samples) | 5× | 2× |
| Small (<75 samples, e.g., `dead_battery`) | 8× | 3× |

This differential augmentation effectively upsamples minority classes while also providing diversity through varied augmentations.

---

## 9. Step 7 — Feature Normalization

### 9.1 Rationale

Neural networks train more effectively when input features are centered and scaled. Without normalization, features with large absolute values (e.g., spectral centroid in Hz) dominate gradients, and features with small values are ignored.

### 9.2 Method: Per-Channel Standardization (Z-Score)

For each mel band (or MFCC coefficient), compute the mean and standard deviation across ALL frames of ALL training samples, then normalize:

```python
# Compute statistics on TRAINING SET ONLY
train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)  # mean per mel band
train_std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-6  # std per mel band

# Apply to all sets
X_train_norm = (X_train - train_mean) / train_std
X_val_norm = (X_val - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std
```

**Critical:** The mean and standard deviation are computed ONLY from the training set and then applied to validation and test sets. Using validation/test statistics would constitute data leakage.

### 9.3 Saving Normalization Statistics

The normalization statistics (mean and std per mel band) must be saved and reused:
- During evaluation (applied to validation/test features)
- During quantization (the representative dataset must use the same normalization)
- During on-device inference (the statistics are baked into the model's input preprocessing or applied as a fixed transform)

```python
np.savez('data/normalization_stats.npz', mean=train_mean, std=train_std)
```

### 9.4 On-Device Normalization

On the Arduino, applying floating-point normalization defeats the purpose of int8 quantization. Instead, the normalization is absorbed into the TFLite model's input quantization parameters during conversion. The TFLite converter computes the input quantization scale and zero-point from the representative dataset (which is already normalized), so the int8 input values inherently encode the normalized feature values. This means:

1. On-device: compute mel-spectrogram (in integer or float, depending on implementation)
2. The mel-spectrogram values are quantized to int8 using the scale/zero-point from the TFLite model
3. This quantization step implicitly handles the normalization, as long as the on-device feature values are in the same range as the training features

If on-device features are computed in a different numeric range than the training features (e.g., integer vs. float), we may need an explicit rescaling step. See Section 11 for the full discussion.

---

## 10. Label Encoding and Class Mappings

### 10.1 Tier 1 — Binary (2 classes)

| Integer Label | Class Name | Mapped Conditions |
|--------------|-----------|------------------|
| 0 | Normal | `normal_brakes`, `normal_engine_idle`, `normal_engine_startup` |
| 1 | Anomaly | All other conditions (including combined faults) |

**Training sample distribution after mapping:** Normal: 402 (29%), Anomaly: 984 (71%)
**Note:** This is ALSO imbalanced (in favor of anomaly). Class weighting is still needed.

### 10.2 Tier 2 — State-Aware (6 classes)

| Integer Label | Class Name | Mapped Conditions | Count |
|--------------|-----------|------------------|-------|
| 0 | Normal Braking | `normal_brakes` | 77 |
| 1 | Braking Fault | `worn_out_brakes` | 76 |
| 2 | Normal Idle | `normal_engine_idle` | 264 |
| 3 | Idle Fault | `low_oil`, `power_steering`, `serpentine_belt`, all 4 combined | 789 |
| 4 | Normal Start-Up | `normal_engine_startup` | 61 |
| 5 | Start-Up Fault | `bad_ignition`, `dead_battery` | 119 |

**Note:** "Idle Fault" (789 samples) is heavily dominant. "Normal Start-Up" (61 samples) is the smallest. Imbalance ratio: 789/61 = 12.9:1. Class weighting and oversampling of minority classes are essential.

### 10.3 Tier 3 — Full Fault Identification (9 classes, excluding combined)

| Integer Label | Class Name | Count |
|--------------|-----------|-------|
| 0 | Normal Braking | 77 |
| 1 | Worn Brakes | 76 |
| 2 | Normal Idle | 264 |
| 3 | Low Oil | 107 |
| 4 | Power Steering Fault | 129 |
| 5 | Serpentine Belt Fault | 116 |
| 6 | Normal Start-Up | 61 |
| 7 | Bad Ignition | 62 |
| 8 | Dead Battery | 57 |

**Total (excluding combined):** 949 files. Imbalance ratio: 264/57 = 4.63:1.

### 10.4 Class Weight Computation

For weighted loss functions, compute inverse-frequency weights:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
```

This assigns higher weights to under-represented classes, ensuring the model pays proportionally more attention to rare classes during training.

---

## 11. On-Device Feature Extraction Parity

### 11.1 The Training-Serving Skew Problem

A critical concern for deployed ML models is **training-serving skew** — when the features computed during training (on a PC using librosa/NumPy in float64/float32) differ from the features computed during inference (on the Cortex-M4 using CMSIS-DSP in fixed-point or reduced-precision float32). If the feature distributions differ significantly, the model's accuracy will degrade on-device even if it performs well on the PC test set.

### 11.2 Sources of Skew

| Source | Risk Level | Mitigation |
|--------|-----------|-----------|
| **FFT implementation differences** | Low | librosa uses scipy/numpy FFT (float64); CMSIS-DSP uses `arm_rfft_fast_f32` (float32) or `arm_rfft_q15` (fixed-point Q15). Float32 FFT on Cortex-M4 is nearly identical to float64. Q15 FFT introduces quantization noise but is usually acceptable. |
| **Mel filterbank construction** | Low-Medium | librosa's mel filterbank uses Slaney normalization by default. The on-device filterbank must use the EXACT same center frequencies and normalization. Simplest approach: pre-compute the mel filterbank matrix on PC and embed it as a constant array in the Arduino sketch. |
| **Log compression** | Low | `log()` on float32 is highly accurate. On fixed-point, a lookup table or piecewise linear approximation is used. Small differences are within the quantization noise floor. |
| **Window function** | Negligible | Hann window coefficients are pre-computed constants — identical on PC and device. |
| **Normalization statistics** | Medium | If normalization mean/std are computed in float64 on PC but applied in float32 or int on-device, there can be rounding differences. Mitigation: save normalization stats in float32 and use those same values on-device. |

### 11.3 Recommended Approach: Float32 Feature Extraction on Device

The Cortex-M4F has a hardware FPU that can perform float32 operations. The recommended approach is:

1. **On-device:** Compute the mel-spectrogram in float32 using CMSIS-DSP FFT and a pre-computed mel filterbank matrix (also stored in float32)
2. **Apply log compression** in float32
3. **Quantize the mel-spectrogram to int8** using the TFLite model's input scale/zero-point
4. **Feed int8 values to the TFLite Micro interpreter**

This approach minimizes training-serving skew because the float32 feature extraction on-device closely matches the float64 feature extraction on PC (the difference between float32 and float64 mel-spectrograms is negligible).

### 11.4 Validation Strategy

Before deploying on-device, validate feature parity:

1. Pick 10–20 representative audio clips from the test set
2. Compute their mel-spectrograms on PC using librosa (the training pipeline)
3. Compute their mel-spectrograms on the Arduino using CMSIS-DSP (or simulate the on-device pipeline in Python using float32 arithmetic with the same FFT size and mel filterbank)
4. Compare: compute the mean absolute error (MAE) and maximum absolute error between PC and on-device spectrograms
5. If MAE < 1% of the feature dynamic range, the skew is negligible

---

## 12. Preprocessing Outputs and Artifacts

The preprocessing pipeline produces the following artifacts, which should be saved to disk for reproducibility:

```
data/
├── splits/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   └── test_manifest.csv
├── features/
│   ├── mel_spectrograms/
│   │   ├── train.npz       # X_train, y_train (all tiers)
│   │   ├── val.npz
│   │   └── test.npz
│   └── mfcc/
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
├── classical_ml_features/
│   ├── train.npz            # 441-dim feature vectors
│   ├── val.npz
│   └── test.npz
├── normalization_stats.npz   # mean, std per mel band
└── preprocessing_config.json # all parameters (sample rate, n_fft, etc.)
```

### 12.1 `preprocessing_config.json`

This file records ALL preprocessing parameters for full reproducibility:

```json
{
    "target_sr": 16000,
    "target_duration_s": 1.5,
    "target_samples": 24000,
    "normalization": "peak",
    "mel_spectrogram": {
        "n_fft": 512,
        "hop_length": 256,
        "n_mels": 40,
        "fmin": 20,
        "fmax": 8000,
        "power": 2.0,
        "window": "hann"
    },
    "mfcc": {
        "n_mfcc": 13,
        "include_deltas": true
    },
    "split": {
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "random_state": 42,
        "stratify_by": "condition"
    },
    "augmentation": {
        "time_shift_max_ms": 100,
        "noise_snr_range_db": [5, 25],
        "pitch_shift_range_semitones": [-2, 2],
        "speed_perturb_range": [0.9, 1.1],
        "gain_range": [0.7, 1.3],
        "specaugment_freq_mask_param": 7,
        "specaugment_time_mask_param": 10
    }
}
```

---

## 13. Implementation Notes

### 13.1 Library Dependencies

```
librosa>=0.10.0          # Audio loading, feature extraction
soundfile>=0.12.0        # WAV file reading
numpy>=1.24.0            # Numerical operations
pandas>=2.0.0            # Data management
scikit-learn>=1.3.0      # Train/test split, classical ML, metrics
tensorflow>=2.15.0       # Model training, TFLite conversion
matplotlib>=3.7.0        # Visualization
```

### 13.2 Reproducibility

All random operations must be seeded:

```python
import numpy as np
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

### 13.3 Data Pipeline Performance

For training efficiency, the preprocessing pipeline should use `tf.data.Dataset` with:
- **Prefetching** (`dataset.prefetch(tf.data.AUTOTUNE)`) to overlap data loading with training
- **Parallel mapping** (`dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)`) for on-the-fly augmentation
- **Caching** (`dataset.cache()`) if the preprocessed features fit in memory (~15 MB for mel-spectrograms, easily fits)
- **Shuffling** (`dataset.shuffle(buffer_size=1000)`) with a large buffer for good randomization

### 13.4 Memory Considerations for Pre-Computed Features

Estimated sizes of pre-computed feature arrays (float32):
- Mel-spectrograms: 1,386 samples × 40 × 92 × 4 bytes = ~20 MB (trivially fits in PC memory)
- MFCCs: 1,386 samples × 92 × 13 × 4 bytes = ~6.6 MB
- Classical ML features: 1,386 samples × 441 × 4 bytes = ~2.4 MB

Total: ~29 MB — easily fits in memory even on modest hardware.
