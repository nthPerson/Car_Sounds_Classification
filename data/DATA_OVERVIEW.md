# Data Overview

This document describes all data files and directories used in the project. All audio is standardized to 16 kHz mono, 1.5 seconds (24,000 samples), float32.

---

## 1. Source Dataset

The raw audio source is the [Car Diagnostics Dataset](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset) (Kaggle), stored in `car_diagnostics_dataset/` (gitignored). It contains 1,386 WAV files across 3 operational states and 13 conditions:

| State | Conditions | Files |
|-------|-----------|-------|
| Braking | Normal Braking, Worn Brakes | 154 |
| Idle | Normal Idle, Low Oil, Power Steering Fault, Serpentine Belt Fault, + combined faults | 946 |
| Start-Up | Normal Start-Up, Bad Ignition, Dead Battery | 286 |

Original sample rates are 24 kHz, 44.1 kHz, or 48 kHz depending on the recording. All files are resampled to 16 kHz during preprocessing.

---

## 2. Train/Val/Test Splits

**Directory:** `data/splits/`

These CSV manifests define the 70/15/15 stratified split and are the **single source of truth** for which files belong to which split. They are committed to git and should never be regenerated unless intentionally changing the split.

| File | Samples | Description |
|------|---------|-------------|
| `train_manifest.csv` | 970 | Training set |
| `val_manifest.csv` | 208 | Validation set (early stopping, LR scheduling, model checkpointing) |
| `test_manifest.csv` | 208 | Held-out test set (final evaluation only) |

**Columns:** `file_path`, `filename`, `state`, `condition`, `sample_rate`, `duration_s`, `channels`, `tier1_label`, `tier1_name`, `tier2_label`, `tier2_name`, `tier3_label`, `tier3_name`

**Tier 2 class distribution (primary deployment target):**

| Class | Train | Val | Test |
|-------|-------|-----|------|
| Normal Braking | 54 | 11 | 12 |
| Braking Fault | 53 | 12 | 11 |
| Normal Idle | 185 | 39 | 40 |
| Idle Fault | 552 | 119 | 118 |
| Normal Start-Up | 43 | 9 | 9 |
| Start-Up Fault | 83 | 18 | 18 |

The training set has a 12.8x class imbalance (552:43). Augmented training variants address this (see Section 4).

**Split parameters:** `random_state=42`, stratified by condition.

---

## 3. Pre-Computed Features

Features are extracted from the preprocessed audio and stored as compressed NumPy archives. Each `.npz` file contains:

- `X` — feature array (float32)
- `y_tier1` — Tier 1 labels (0=Normal, 1=Anomaly)
- `y_tier2` — Tier 2 labels (0-5, six state-aware classes)
- `y_tier3` — Tier 3 labels (0-8 for nine fault classes, -1 for excluded combined-fault samples)
- `file_paths` — source file paths (object array)

Features are stored **un-normalized**. Z-score normalization is applied at training time using the appropriate normalization statistics file.

### 3.1 Mel-Spectrograms

**Directory:** `data/features/mel_spectrograms/`

Log-mel spectrograms used as input to M2 (2-D CNN) and M6 (DS-CNN).

| Parameter | Value |
|-----------|-------|
| n_fft | 512 |
| hop_length | 256 |
| n_mels | 40 |
| fmin | 20 Hz |
| fmax | 8,000 Hz |
| center | False |
| Output shape | (40, 92) per sample |

| File | Samples | Shape | Size | Description |
|------|---------|-------|------|-------------|
| `train.npz` | 970 | (970, 40, 92) | 11.9 MB | Original training features |
| `val.npz` | 208 | (208, 40, 92) | 2.5 MB | Validation features |
| `test.npz` | 208 | (208, 40, 92) | 2.6 MB | Test features |
| `train_augmented.npz` | 3,317 | (3317, 40, 92) | 39.9 MB | Condition B: synthetic augmentation + class balancing |
| `train_noise_augmented.npz` | 3,317 | (3317, 40, 92) | 40.6 MB | Condition C: synthetic aug + real-world noise injection |
| `train_combined.npz` | 5,664 | (5664, 40, 92) | 68.6 MB | Condition D: all of B + noise-augmented copies from C |

### 3.2 MFCCs

**Directory:** `data/features/mfcc/`

Mel-frequency cepstral coefficients used as input to M5 (1-D CNN).

| Parameter | Value |
|-----------|-------|
| n_mfcc | 13 |
| Same FFT/hop/mel settings as mel-spectrograms | |
| Output shape | (92, 13) per sample |

| File | Samples | Shape | Size | Description |
|------|---------|-------|------|-------------|
| `train.npz` | 970 | (970, 92, 13) | 4.1 MB | Original training features |
| `val.npz` | 208 | (208, 92, 13) | 0.9 MB | Validation features |
| `test.npz` | 208 | (208, 92, 13) | 0.9 MB | Test features |
| `train_augmented.npz` | 3,317 | (3317, 92, 13) | 13.9 MB | Condition B |
| `train_noise_augmented.npz` | 3,317 | (3317, 92, 13) | 14.1 MB | Condition C |
| `train_combined.npz` | 5,664 | (5664, 92, 13) | 23.9 MB | Condition D |

### 3.3 Classical ML Features

**Directory:** `data/classical_ml_features/`

441-dimensional hand-crafted feature vectors used for M1 (Random Forest, SVM). Each vector consists of 63 frame-level features (13 MFCCs + 13 delta MFCCs + 13 delta-delta MFCCs + spectral centroid, bandwidth, rolloff, 7 spectral contrast bands, RMS, ZCR, 12 chroma) aggregated with 7 statistics (mean, std, min, max, median, skewness, kurtosis).

| File | Samples | Shape | Size |
|------|---------|-------|------|
| `train.npz` | 970 | (970, 441) | 1.5 MB |
| `val.npz` | 208 | (208, 441) | 0.3 MB |
| `test.npz` | 208 | (208, 441) | 0.3 MB |

No augmented variants exist for classical ML features — classical models (RF, SVM) were trained on the original split only, using cross-validation.

---

## 4. Augmented Training Variants

Four training conditions exist for ablation study. All share the same validation and test sets. Augmentation is applied only to the training set.

| Condition | File Suffix | Samples | Class Balanced | Augmentation Applied |
|-----------|------------|---------|---------------|---------------------|
| **A** (Original) | `train.npz` | 970 | No (43-552 per class) | None |
| **B** (Synthetic) | `train_augmented.npz` | 3,317 | Yes (~530-581) | Waveform augmentation |
| **C** (Noise) | `train_noise_augmented.npz` | 3,317 | Yes (~530-581) | Waveform aug + real-world noise |
| **D** (Combined) | `train_combined.npz` | 5,664 | Inverted (552-1079) | B originals + B augmented + C augmented |

**Condition B and C waveform augmentations:** time shift (50%, +/-100ms), Gaussian noise (50%, SNR 5-25dB), pitch shift (30%, +/-2 semitones), speed perturbation (30%, 0.9-1.1x), random gain (50%, 0.7-1.3x). Class-aware multipliers bring minority classes up to ~550 samples each.

**Condition C additionally applies** real-world noise injection (100% of augmented samples) at SNR randomly drawn from {5, 10, 15, 20} dB, using operationally-appropriate noise from the noise bank.

**Condition D structure:** The first 3,317 samples are identical to Condition B (970 originals + 2,347 synthetic copies). The remaining 2,347 are the augmented-only copies from Condition C (excluding the 970 originals to avoid duplication).

---

## 5. Normalization Statistics

Per-channel mean and standard deviation computed from the **training set only** for z-score normalization at training time. Each condition has its own statistics because the augmented data shifts the feature distributions.

| File | Training Condition | Contents |
|------|--------------------|----------|
| `normalization_stats.npz` | A (original, 970 samples) | `mel_mean(40,)`, `mel_std(40,)`, `mfcc_mean(13,)`, `mfcc_std(13,)` |
| `normalization_stats_augmented.npz` | B (synthetic, 3,317) | Same keys |
| `normalization_stats_noise_augmented.npz` | C (noise, 3,317) | Same keys |
| `normalization_stats_combined.npz` | D (combined, 5,664) | Same keys |

---

## 6. Real-World Noise Data

### 6.1 Raw Noise Samples

**Directory:** `data/noise_samples/noise_samples/`

360 WAV files (16 kHz, 1.5s, int16) recorded with the Arduino Nano 33 BLE Sense Rev2's PDM microphone in and around a 2008 Volkswagen Jetta on March 22, 2026. Organized into 23 label-named subdirectories.

| Subdirectory | Samples | Recording Environment |
|-------------|---------|----------------------|
| `ambient_under_hood` | 23 | Engine off, under hood |
| `cold_engine_just_started_under_hood` | 20 | Cold start, under hood |
| `warm_engine_idle_under_hood` | 20 | Warm idle, under hood |
| `varying_rpm_1500-2500_under_hood` | 20 | Light revving, under hood |
| `engine_fan_on_under_hood` | 20 | Cooling fan cycling, under hood |
| `fan_speed_1_idle_under_hood` | 10 | HVAC low, under hood |
| `fan_speed_medium_idle_under_hood` | 10 | HVAC medium, under hood |
| `fan_speed_max_idle_under_hood` | 10 | HVAC max, under hood |
| `low_speed_residential_under_hood` | 20 | ~15-25 mph, under hood |
| `moderate_speed_city_under_hood` | 20 | ~30-45 mph, under hood |
| `rough_road_under_hood` | 20 | Rough surface, under hood |
| `interior_cabin_city_streets_windows_up` | 10 | City driving, cabin, windows up |
| `interior_cabin_city_windows_down` | 10 | City driving, cabin, window down |
| `interior_cabin_highway_speeds_windows_up` | 20 | Highway, cabin, windows up |
| `interior_cabin_highway_speeds_windows_down` | 20 | Highway, cabin, window down |
| `parking_lot_windows_up_ambient` | 20 | Parked, cabin, windows up |
| `parking_lot_windows_down_ambient` | 20 | Parked, cabin, window down |
| `quiet_residential_ambient` | 10 | Quiet residential, cabin |
| `stoplight_ambient_windows_down` | 7 | Stopped at light, window down |
| `busy_street_ambient` | 10 | Near busy street, cabin |
| `busy_highway_ambient` | 20 | Near highway, cabin |
| `windy_day_ambient` | 10 | Windy day, under hood |
| `windy_driving_city_streets` | 10 | City driving in wind, under hood |

**Also contains:** `collection_log.csv` (360 rows) — timestamp, label, filename, sample_rate, num_samples, duration_s, RMS energy for each recorded clip.

### 6.2 Noise Bank

**Directory:** `data/noise_bank/`

Preprocessed noise data used during augmentation. The 23 raw labels are grouped into 6 acoustic categories with operational-state relevance mappings.

| File | Size | Description |
|------|------|-------------|
| `noise_bank.npz` | 14.3 MB | 360 noise waveforms as float32 arrays, keyed by noise group (engine_mechanical, hvac, road_driving, cabin_driving, wind, ambient_stationary) |
| `noise_config.json` | 1 KB | Group definitions, per-group sample counts, operational-state-to-noise-group mapping, SNR levels {5, 10, 15, 20} dB |

**Noise group → operational state mapping:**

| Operational State | Relevant Noise Groups |
|------------------|----------------------|
| Braking | engine_mechanical, road_driving, cabin_driving, wind, hvac |
| Idle | engine_mechanical, hvac, ambient_stationary, wind |
| Startup | ambient_stationary, wind, hvac |

---

## 7. Configuration and Report Files

| File | Size | Description |
|------|------|-------------|
| `preprocessing_config.json` | 4.4 KB | Complete preprocessing parameters: sample rate, FFT settings, mel/MFCC params, augmentation settings, split ratios, tier definitions with class names. Generated by `src/run_preprocessing.py`. |
| `augmentation_report.json` | 0.7 KB | Condition B generation report: original size, augmented count, per-class multipliers, final distribution, techniques applied. Generated by `src/generate_augmented_data.py`. |
| `noise_augmentation_report.json` | 2.0 KB | Condition C generation report: same fields as above plus noise bank configuration (groups, counts, SNR levels, operational-state mapping). Generated by `notebooks/05_noise_augmentation.ipynb`. |

---

## 8. Taxonomy Tiers

All feature files contain labels for three classification tiers:

**Tier 1 — Binary (2 classes):** Normal (0) vs. Anomaly (1)

**Tier 2 — State-Aware (6 classes, primary deployment target):**

| Label | Class | Operational State |
|-------|-------|------------------|
| 0 | Normal Braking | Braking |
| 1 | Braking Fault | Braking |
| 2 | Normal Idle | Idle |
| 3 | Idle Fault | Idle |
| 4 | Normal Start-Up | Start-Up |
| 5 | Start-Up Fault | Start-Up |

**Tier 3 — Individual Faults (9 classes):**

| Label | Class | Operational State |
|-------|-------|------------------|
| 0 | Normal Braking | Braking |
| 1 | Worn Brakes | Braking |
| 2 | Normal Idle | Idle |
| 3 | Low Oil | Idle |
| 4 | Power Steering Fault | Idle |
| 5 | Serpentine Belt Fault | Idle |
| 6 | Normal Start-Up | Start-Up |
| 7 | Bad Ignition | Start-Up |
| 8 | Dead Battery | Start-Up |

Combined-fault samples (present in the source dataset) are excluded from Tier 3 (`tier3_label = -1`) but retained in Tiers 1 and 2 where they map to the appropriate aggregate class.

---

## 9. Directory Structure Summary

```
data/
├── splits/                              # Train/val/test manifests (git-tracked)
│   ├── train_manifest.csv                   970 samples
│   ├── val_manifest.csv                     208 samples
│   └── test_manifest.csv                    208 samples
├── features/
│   ├── mel_spectrograms/                # Log-mel spectrograms (40×92)
│   │   ├── train.npz                        Condition A (970)
│   │   ├── train_augmented.npz              Condition B (3,317)
│   │   ├── train_noise_augmented.npz        Condition C (3,317)
│   │   ├── train_combined.npz               Condition D (5,664)
│   │   ├── val.npz                          Validation (208)
│   │   └── test.npz                         Test (208)
│   └── mfcc/                            # MFCCs (92×13) — same variants as above
├── classical_ml_features/               # 441-dim feature vectors (train/val/test only)
├── noise_samples/noise_samples/         # Raw noise WAVs (23 categories, 360 files)
│   ├── ambient_under_hood/
│   ├── cold_engine_just_started_under_hood/
│   ├── ...                                  (23 subdirectories)
│   └── collection_log.csv
├── noise_bank/                          # Preprocessed noise for augmentation
│   ├── noise_bank.npz                       360 waveforms grouped by category
│   └── noise_config.json                    Group definitions and mapping
├── normalization_stats.npz              # Condition A normalization
├── normalization_stats_augmented.npz    # Condition B normalization
├── normalization_stats_noise_augmented.npz  # Condition C normalization
├── normalization_stats_combined.npz     # Condition D normalization
├── preprocessing_config.json            # All preprocessing parameters
├── augmentation_report.json             # Condition B generation report
└── noise_augmentation_report.json       # Condition C generation report
```
