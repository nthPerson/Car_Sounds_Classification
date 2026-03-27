# Data Sources — Report Assembly

## 1. Car Diagnostics Dataset

- **Source:** [Car Diagnostics Dataset on Kaggle](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset) by Malak Ragaie
- **Total files:** 1,386 WAV recordings
- **Operational states:** 3 (braking, idle, startup)
- **Conditions:** 13 total (normal + fault conditions per state)
  - Braking: normal_braking, worn_brakes
  - Idle: normal_engine_idle, low_oil_level, power_steering_fault, serpentine_belt_fault, combined faults (low_oil+power_steering, low_oil+serpentine, power_steering+serpentine, all_three)
  - Startup: normal_startup, bad_ignition, dead_battery
- **Raw sample rates:** Mixed (24 kHz, 44.1 kHz, 48 kHz)
- **Raw durations:** Braking/idle: 1.5s, Startup: 1.7s

## 2. Taxonomy Tiers

| Tier | Classes | Description | Use Case |
|------|---------|-------------|----------|
| Tier 1 | 2 | Normal vs Anomaly (binary) | Quick screening |
| **Tier 2** | **6** | **State-aware: Normal/Fault per operational state** | **Primary deployment target** |
| Tier 3 | 9 | Individual fault identification (excludes combined faults) | Detailed diagnosis |

### Tier 2 Class Definitions (Primary)

| Class | Label | Operational State | Conditions Included |
|-------|-------|-------------------|-------------------|
| 0 | Normal Braking | Braking | normal_braking |
| 1 | Braking Fault | Braking | worn_brakes |
| 2 | Normal Idle | Idle | normal_engine_idle |
| 3 | Idle Fault | Idle | low_oil, power_steering, serpentine_belt, combined faults |
| 4 | Normal Start-Up | Startup | normal_startup |
| 5 | Start-Up Fault | Startup | bad_ignition, dead_battery |

## 3. Train/Val/Test Split

| Set | Samples | Proportion | Purpose |
|-----|---------|-----------|---------|
| Train | 970 | 70% | Model training |
| Validation | 208 | 15% | Hyperparameter tuning, early stopping |
| Test | 208 | 15% | Final evaluation (used once) |

- **Stratification:** By condition (13-way), ensuring every condition appears in all sets
- **Random state:** 42 (reproducible)
- **Manifests:** `data/splits/{train,val,test}_manifest.csv`

### Tier 2 Class Distribution (Training Set)

| Class | Count | Percentage | Imbalance |
|-------|-------|-----------|-----------|
| Idle Fault | 552 | 56.9% | 12.8x vs smallest |
| Normal Idle | 185 | 19.1% | 4.3x |
| Start-Up Fault | 83 | 8.6% | 1.9x |
| Normal Braking | 54 | 5.6% | 1.3x |
| Braking Fault | 53 | 5.5% | 1.2x |
| Normal Start-Up | 43 | 4.4% | 1.0x (smallest) |

**Imbalance ratio:** 12.8x (552 Idle Fault : 43 Normal Start-Up). This severe imbalance motivated the data augmentation strategy.

## 4. Data Preprocessing

| Step | Details |
|------|---------|
| Resampling | All files to 16 kHz mono (matches on-device PDM mic) |
| Peak normalization | Divide by max absolute amplitude to [-1, 1] |
| Duration standardization | Truncate to 1.5s (24,000 samples); pad shorter clips with zeros |
| Startup handling | Center-crop 1.7s startup clips to 1.5s |

**Configuration saved to:** `data/preprocessing_config.json`

## 5. Feature Extraction

### 5.1 Log-Mel Spectrograms (for M2 2D-CNN and M6 DS-CNN)

| Parameter | Value |
|-----------|-------|
| n_fft | 512 |
| hop_length | 256 |
| n_mels | 40 |
| fmin | 20 Hz |
| fmax | 8,000 Hz |
| Output shape | (40, 92) per sample; expanded to (40, 92, 1) for 2D-CNN input |
| Log conversion | `librosa.power_to_db(S, ref=np.max)`, top_db=80 |
| Normalization | Per-band z-score (mean/std from training set only) |

### 5.2 MFCCs (for M5 1D-CNN)

| Parameter | Value |
|-----------|-------|
| n_mfcc | 13 |
| Same FFT/hop/mel params | As above |
| Output shape | (92, 13) |
| Normalization | Per-coefficient z-score |

### 5.3 Classical ML Features (for M1 RF/SVM)

- 63 frame-level features: 13 MFCCs + 13 delta MFCCs + 13 delta-delta MFCCs + spectral centroid, bandwidth, contrast (7 bands), rolloff, zero-crossing rate, RMS energy, chroma (12 bins)
- 7 aggregation statistics per feature: mean, std, min, max, median, skew, kurtosis
- Total: 63 x 7 = 441 dimensions
- Source: `src/features.py`

### 5.4 Normalization Statistics

- Computed from **training set only** (never val/test)
- Separate stats per training condition: `data/normalization_stats.npz` (original), `data/normalization_stats_augmented.npz` (Condition B), `data/normalization_stats_noise_augmented.npz` (Condition C), `data/normalization_stats_combined.npz` (Condition D)
- Applied identically to val and test sets

## 6. Data Augmentation

### 6.1 Training Conditions

| Condition | Samples | Augmentation | Class Balanced |
|-----------|---------|-------------|---------------|
| A (Original) | 970 | None | No (43-552/class) |
| B (Synthetic) | 3,317 | Waveform augmentation | Yes (~530-581/class) |
| C (Noise) | 3,317 | Waveform aug + real-world noise | Yes (~530-581/class) |
| D (Combined) | 5,664 | B + C augmented copies combined | Partially (552-1,079/class) |

### 6.2 Waveform Augmentations (Conditions B, C, D)

| Augmentation | Probability | Parameters |
|-------------|------------|------------|
| Time shift | 50% | +/-100 ms (1,600 samples) |
| Gaussian noise | 50% | SNR 5-25 dB |
| Pitch shift | 30% | +/-2 semitones |
| Speed perturbation | 30% | 0.9-1.1x rate |
| Random gain | 50% | 0.7-1.3x amplitude |

### 6.3 Class-Balancing Multipliers (target ~550/class)

| Class | Original | Multiplier | Final (B/C) |
|-------|----------|-----------|-------------|
| Normal Braking | 54 | 9x | 540 |
| Braking Fault | 53 | 9x | 530 |
| Normal Idle | 185 | 2x | 555 |
| Idle Fault | 552 | 0x | 552 |
| Normal Start-Up | 43 | 12x | 559 |
| Start-Up Fault | 83 | 6x | 581 |

### 6.4 Feature Files

| Condition | Mel-Spectrogram File | MFCC File |
|-----------|---------------------|-----------|
| A (Original) | `data/features/mel_spectrograms/train.npz` | `data/features/mfcc/train.npz` |
| B (Synthetic) | `train_augmented.npz` | `train_augmented.npz` |
| C (Noise) | `train_noise_augmented.npz` | `train_noise_augmented.npz` |
| D (Combined) | `train_combined.npz` | `train_combined.npz` |

## 7. Real-World Noise Bank

### 7.1 Collection Methodology

- **Hardware:** Arduino Nano 33 BLE Sense Rev2 (same as deployment) with on-board MP34DT06JTR PDM mic
- **Vehicle:** 2008 Volkswagen Jetta
- **Date:** March 22, 2026
- **Protocol:** Arduino captured 1.5s clips at 16 kHz, transmitted binary over serial to PC receiver script
- **Software:** `arduino/noise_collector/noise_collector.ino` (Arduino) + `src/collect_noise.py` (PC)
- **Total:** 360 samples across 23 categories (9.0 minutes of audio)

### 7.2 Noise Categories

| Group | Samples | Source Labels |
|-------|---------|--------------|
| Engine Mechanical | 103 | ambient_under_hood, cold_engine, warm_engine_idle, varying_rpm, engine_fan |
| HVAC | 30 | fan_speed_1, fan_speed_medium, fan_speed_max |
| Road/Driving | 60 | low_speed_residential, moderate_speed_city, rough_road |
| Cabin/Driving | 60 | interior_cabin (city/highway, windows up/down) |
| Wind | 20 | windy_day_ambient, windy_driving |
| Ambient/Stationary | 87 | parking_lot, quiet_residential, stoplight, busy_street, busy_highway |

### 7.3 Operational-State Mapping

| Operational State | Relevant Noise Groups |
|------------------|----------------------|
| Braking | Engine Mechanical, Road/Driving, Cabin/Driving, Wind, HVAC |
| Idle | Engine Mechanical, HVAC, Ambient/Stationary, Wind |
| Startup | Ambient/Stationary, Wind, HVAC |

### 7.4 Noise Injection Parameters

- **SNR levels:** {5, 10, 15, 20} dB (uniformly sampled)
- **Mixing formula:** `mixed = signal + scale * noise` where `scale = sqrt(signal_power / (noise_power * 10^(SNR/10)))`
- **Applied to:** 100% of augmented samples in Condition C

### 7.5 Quality Validation (all 360 passed)

- Sample rate: 16,000 Hz (all)
- Duration: 1.5s exactly (all)
- Clipping: None
- Silence: None (RMS > 0.001)

## Source Files
- `docs/project_spec/data_preprocessing_spec.md` — Full preprocessing specification
- `docs/noise_data_collection_and_augmentation.md` — Noise collection and augmentation methodology
- `data/DATA_OVERVIEW.md` — Complete data file inventory
- `data/preprocessing_config.json` — Preprocessing parameters
- `data/noise_bank/noise_config.json` — Noise bank configuration
- `data/noise_samples/noise_samples/collection_log.csv` — Collection metadata (360 entries)
- `notebooks/04_noise_analysis.ipynb` — Noise quality analysis
- `notebooks/05_noise_augmentation.ipynb` — Augmentation pipeline
