# Real-World Noise Data Collection and Augmentation

**Date:** March 22, 2026
**Related notebooks:** `notebooks/04_noise_analysis.ipynb`, `notebooks/05_noise_augmentation.ipynb`

---

## 1. Motivation

The Car Diagnostics Dataset was recorded under controlled conditions, but the deployment target --- an Arduino Nano 33 BLE Sense Rev2 with an on-board PDM microphone --- will operate in real-world automotive environments where engine sounds are mixed with road noise, wind, HVAC, and ambient environmental sound. This domain gap between training and deployment conditions risks degrading classifier performance on-device.

To bridge this gap, we collected a bank of real-world background noise samples using the same Arduino hardware and microphone that will be used for deployment, recorded in and around a real vehicle. These noise samples were then mixed into the training data at controlled signal-to-noise ratios (SNRs) to produce a noise-augmented training set.

This approach has two advantages over synthetic noise (e.g., Gaussian white noise):

1. **Spectral realism.** Real automotive noise is spectrally colored --- road noise concentrates energy at low frequencies, wind noise is broadband but non-stationary, and HVAC fan noise has harmonic structure. Gaussian noise is spectrally flat and does not teach the model to handle these patterns.

2. **Hardware fidelity.** Recording through the same PDM microphone captures the microphone's frequency response, noise floor, and quantization characteristics. This means the noise in training data has the same spectral signature as the noise the model will encounter during deployment.

---

## 2. Collection Hardware and Vehicle

| Component | Details |
|-----------|---------|
| **Microcontroller** | Arduino Nano 33 BLE Sense Rev2 (nRF52840, Cortex-M4F @ 64 MHz) |
| **Microphone** | On-board MP34DT06JTR MEMS PDM microphone |
| **Sample rate** | 16 kHz (matching training pipeline) |
| **Bit depth** | 16-bit signed integer (converted to float32 on PC) |
| **Clip duration** | 1.5 seconds (24,000 samples per clip) |
| **Vehicle** | 2008 Volkswagen Jetta |
| **Collection date** | March 22, 2026 |

The Arduino was connected to a laptop via USB. Audio was captured by the on-board PDM microphone, transmitted as binary data over serial at 115,200 baud, and saved as WAV files by a PC-side Python receiver script.

---

## 3. Collection Software

Two software components were developed for data collection:

### 3.1 Arduino Sketch (`arduino/noise_collector/noise_collector.ino`)

The sketch runs on the Arduino and provides an interactive serial interface. Commands are entered via the Arduino IDE Serial Monitor or the PC-side script:

- `label <name>` --- Set a descriptive label for subsequent recordings (e.g., `label warm_engine_idle_under_hood`)
- `record <N>` --- Capture N consecutive 1.5-second clips with ~2 second gaps
- `gain <value>` --- Set PDM microphone gain (10, 20, 40, or 60)
- `status` --- Display current label and gain settings

Each clip is transmitted using a binary transfer protocol with text-based header lines (label, sample rate, sample count) followed by raw int16 audio data and a checksum for integrity verification.

### 3.2 PC Receiver Script (`src/collect_noise.py`)

The Python script connects to the Arduino over serial, provides an interactive terminal for sending commands, and automatically receives and saves audio clips. It:

- Saves each clip as a 16 kHz, 16-bit WAV file organized by label into subdirectories
- Logs every recording to `collection_log.csv` with timestamp, label, filename, sample rate, sample count, duration, and RMS energy
- Supports cross-platform use (Linux, WSL, Windows)

---

## 4. Collection Protocol

Noise samples were collected across a range of conditions representative of real-world deployment scenarios. The Arduino was positioned either under the hood (near the engine bay) or inside the cabin, depending on the noise category.

### 4.1 Categories Collected

| # | Label | Samples | Mic Position | Engine State | Vehicle State | Description |
|---|-------|---------|-------------|-------------|--------------|-------------|
| 1 | `ambient_under_hood` | 23 | Under hood | Off | Parked | Baseline ambient with engine off |
| 2 | `cold_engine_just_started_under_hood` | 20 | Under hood | Cold start | Parked | Engine immediately after cold start |
| 3 | `warm_engine_idle_under_hood` | 20 | Under hood | Warm idle | Parked | Engine idling after reaching operating temperature |
| 4 | `varying_rpm_1500-2500_under_hood` | 20 | Under hood | Revving | Parked | Light throttle blips, 1500--2500 RPM |
| 5 | `engine_fan_on_under_hood` | 20 | Under hood | Warm idle | Parked | Radiator/cooling fan cycling on |
| 6 | `fan_speed_1_idle_under_hood` | 10 | Under hood | Warm idle | Parked | HVAC blower at lowest setting |
| 7 | `fan_speed_medium_idle_under_hood` | 10 | Under hood | Warm idle | Parked | HVAC blower at medium setting |
| 8 | `fan_speed_max_idle_under_hood` | 10 | Under hood | Warm idle | Parked | HVAC blower at maximum setting |
| 9 | `low_speed_residential_under_hood` | 20 | Under hood | Running | ~15--25 mph | Residential streets, low speed |
| 10 | `moderate_speed_city_under_hood` | 20 | Under hood | Running | ~30--45 mph | City driving, moderate speed |
| 11 | `rough_road_under_hood` | 20 | Under hood | Running | Variable | Uneven/rough road surface |
| 12 | `interior_cabin_city_streets_windows_up` | 10 | Cabin | Running | City | City driving, windows closed |
| 13 | `interior_cabin_city_windows_down` | 10 | Cabin | Running | City | City driving, driver window open |
| 14 | `interior_cabin_highway_speeds_windows_up` | 20 | Cabin | Running | Highway | Highway speeds (~55--70 mph), windows closed |
| 15 | `interior_cabin_highway_speeds_windows_down` | 20 | Cabin | Running | Highway | Highway speeds, driver window open |
| 16 | `parking_lot_windows_up_ambient` | 20 | Cabin | Off | Parked | Parking lot ambient, windows closed |
| 17 | `parking_lot_windows_down_ambient` | 20 | Cabin | Off | Parked | Parking lot ambient, window open |
| 18 | `quiet_residential_ambient` | 10 | Cabin | Off | Parked | Quiet residential street, engine off |
| 19 | `stoplight_ambient_windows_down` | 7 | Cabin | Idle | Stopped | Stopped at traffic light, window open |
| 20 | `busy_street_ambient` | 10 | Cabin | Off | Parked | Parked near busy street, passing traffic |
| 21 | `busy_highway_ambient` | 20 | Cabin | Off | Parked | Parked near highway, heavy traffic |
| 22 | `windy_day_ambient` | 10 | Under hood | Off | Parked | Windy conditions, engine off |
| 23 | `windy_driving_city_streets` | 10 | Under hood | Running | City | City driving in windy conditions |

**Total: 360 samples across 23 categories (9.0 minutes of audio).**

### 4.2 Collection Procedure

For each category, the operator:

1. Positioned the Arduino (under hood or in cabin) and confirmed stable serial connection
2. Set the label via the serial terminal (e.g., `label warm_engine_idle_under_hood`)
3. Established the target condition (e.g., started the engine, reached operating temperature, set HVAC to medium)
4. Issued a batch recording command (e.g., `record 20`) to capture consecutive clips
5. Verified the saved WAV files and RMS values in the collection log

Clips were recorded consecutively with ~2-second gaps between each 1.5-second capture.

---

## 5. Quality Validation

All 360 samples passed automated quality checks (performed in `notebooks/04_noise_analysis.ipynb`):

- **Sample rate:** All at 16,000 Hz
- **Duration:** All exactly 1.5 seconds (24,000 samples)
- **Clipping:** Zero samples exhibited clipping (no samples at +/-1.0)
- **Silence:** Zero samples were near-silent (all RMS > 0.001)
- **File integrity:** All WAV files loaded successfully

### 5.1 RMS Energy Profile

Noise samples had a mean RMS of 0.037 compared to the training data mean RMS of 0.210 --- approximately 5.6x lower energy. The natural (unscaled) SNR between training data and noise is approximately 11.9 dB, which falls within the target mixing range. During augmentation, noise is explicitly scaled to target SNR levels rather than mixed at natural amplitude.

The loudest noise category was `interior_cabin_highway_speeds_windows_down` (mean RMS 0.138), reflecting the high broadband energy from wind and road noise at highway speeds with an open window. The quietest was `quiet_residential_ambient` (mean RMS ~0.008).

### 5.2 Near-Duplicate Analysis

Pairwise waveform correlation analysis identified 466 near-duplicate pairs (correlation > 0.95), concentrated in three quiet, stationary categories:

- `parking_lot_windows_up_ambient` (190 pairs) --- very low-energy, nearly constant ambient hum
- `ambient_under_hood` (211 pairs) --- engine off, steady low-level background
- `quiet_residential_ambient` (45 pairs) --- near-silent environment

This is expected behavior: in near-silent conditions, consecutive 1.5-second clips capture essentially the same steady-state background. The dynamic categories (driving, engine running, wind) showed near-zero duplicates, confirming that these samples provide genuine acoustic variety.

No samples were removed, as the near-duplicates are low-energy and contribute minimal spectral variation when mixed at the target SNR levels.

---

## 6. Noise Grouping and Operational-State Mapping

The 23 fine-grained labels were organized into 6 broader noise groups based on acoustic characteristics and deployment relevance:

| Group | Source Labels | Samples | Characteristics |
|-------|-------------|---------|-----------------|
| **Engine Mechanical** | `ambient_under_hood`, `cold_engine_just_started_under_hood`, `warm_engine_idle_under_hood`, `varying_rpm_1500-2500_under_hood`, `engine_fan_on_under_hood` | 103 | Low-to-mid frequency mechanical hum, fan noise, combustion harmonics |
| **HVAC** | `fan_speed_1_idle_under_hood`, `fan_speed_medium_idle_under_hood`, `fan_speed_max_idle_under_hood` | 30 | Broadband fan noise with speed-dependent harmonic content |
| **Road/Driving** | `low_speed_residential_under_hood`, `moderate_speed_city_under_hood`, `rough_road_under_hood` | 60 | Tire-road contact noise, suspension, speed-dependent spectral profile |
| **Cabin/Driving** | `interior_cabin_city_streets_windows_up`, `interior_cabin_city_windows_down`, `interior_cabin_highway_speeds_windows_up`, `interior_cabin_highway_speeds_windows_down` | 60 | Filtered version of road/wind noise, cabin resonances, window-state dependent |
| **Wind** | `windy_day_ambient`, `windy_driving_city_streets` | 20 | Broadband, non-stationary turbulent noise |
| **Ambient/Stationary** | `parking_lot_windows_up_ambient`, `parking_lot_windows_down_ambient`, `quiet_residential_ambient`, `stoplight_ambient_windows_down`, `busy_street_ambient`, `busy_highway_ambient` | 87 | Environmental background: traffic, birds, distant road noise |

### 6.1 Operational-State Relevance Mapping

Not all noise types are equally relevant for all car diagnostic conditions. A car being diagnosed during braking will have road and wind noise present, while an idle diagnosis will not. The following mapping ensures that augmented training samples receive operationally-appropriate noise:

| Operational State | Training Classes | Relevant Noise Groups | Available Noise Samples |
|------------------|-----------------|----------------------|------------------------|
| **Braking** | Normal Braking, Braking Fault | Engine Mechanical, Road/Driving, Cabin/Driving, Wind, HVAC | 273 |
| **Idle** | Normal Idle, Idle Fault | Engine Mechanical, HVAC, Ambient/Stationary, Wind | 240 |
| **Startup** | Normal Start-Up, Start-Up Fault | Ambient/Stationary, Wind, HVAC | 137 |

This mapping is stored in `data/noise_bank/noise_config.json` and applied automatically during augmentation.

---

## 7. Augmentation Methodology

### 7.1 Noise-Augmented Training Set (Condition C)

Each augmented training sample is produced by a two-stage pipeline:

**Stage 1 --- Standard Waveform Augmentation.** The same augmentations used in the synthetic-only condition (Condition B), applied with the same probabilities:

| Augmentation | Probability | Parameters |
|-------------|------------|------------|
| Time shift | 50% | +/-100 ms (1,600 samples at 16 kHz) |
| Gaussian noise | 50% | SNR 5--25 dB |
| Pitch shift | 30% | +/-2 semitones |
| Speed perturbation | 30% | 0.9--1.1x rate |
| Random gain | 50% | 0.7--1.3x amplitude |

**Stage 2 --- Real-World Noise Injection.** Applied to 100% of augmented samples (always on):

1. Select the operational state for the sample's Tier 2 class (braking, idle, or startup)
2. Pool all noise samples from the relevant noise groups for that state
3. Select one noise sample uniformly at random from the pool
4. Select an SNR uniformly at random from {5, 10, 15, 20} dB
5. Scale the noise to achieve the target SNR relative to the signal power and mix additively

The mixing formula:

```
mixed = signal + scale * noise
scale = sqrt(signal_power / (noise_power * 10^(SNR_dB / 10)))
```

### 7.2 Class Balancing

The same class-balancing strategy as Condition B is applied: per-class augmentation multipliers bring each Tier 2 class to approximately 550 samples (matching the largest class, Idle Fault, at 552).

| Tier 2 Class | Original | Multiplier | Augmented Copies | Final Count |
|-------------|----------|-----------|-----------------|-------------|
| Normal Braking | 54 | 9x | 486 | 540 |
| Braking Fault | 53 | 9x | 477 | 530 |
| Normal Idle | 185 | 2x | 370 | 555 |
| Idle Fault | 552 | 0x | 0 | 552 |
| Normal Start-Up | 43 | 12x | 516 | 559 |
| Start-Up Fault | 83 | 6x | 498 | 581 |
| **Total** | **970** | | **2,347** | **3,317** |

### 7.3 Reproducibility

All random operations use `numpy.random.default_rng` with deterministic seeds derived from `random_state=42`. Each augmented copy receives a unique seed (`42 + sample_index * 100 + copy_index + 10000`), ensuring reproducibility while producing distinct augmentations. The seed offset of 10,000 ensures no overlap with the seeds used for the synthetic-only augmentation (Condition B).

---

## 8. Ablation Study Design

Three training conditions were produced to enable ablation analysis of the augmentation strategy:

| Condition | Training File | Samples | Augmentation | Class Balanced |
|-----------|--------------|---------|-------------|---------------|
| **A** (Baseline) | `train.npz` | 970 | None | No (43--552 per class) |
| **B** (Synthetic) | `train_augmented.npz` | 3,317 | Waveform augmentation | Yes (~540--581) |
| **C** (Noise) | `train_noise_augmented.npz` | 3,317 | Waveform aug + real-world noise | Yes (~540--581) |

Conditions B and C have identical sample counts and class distributions, differing only in whether real-world noise injection was applied. This allows direct comparison to isolate the effect of noise augmentation.

All three conditions use the same validation and test sets (unaugmented, 208 samples each) and the same model architectures and hyperparameters. Each condition has its own normalization statistics computed from its respective training set.

### 8.1 Expected Ablation Comparisons

- **A vs. B:** Effect of data augmentation and class balancing (already demonstrated in Phase 3 results)
- **A vs. C:** Combined effect of augmentation, class balancing, and noise injection
- **B vs. C:** Isolated effect of real-world noise injection (primary comparison of interest)

The B vs. C comparison is the most informative for evaluating whether real-world noise injection improves model robustness, since it controls for the effects of dataset expansion and class balancing.

---

## 9. Output Artifacts

| File | Description | Size |
|------|-------------|------|
| `data/noise_samples/noise_samples/` | Raw WAV files organized by label (23 subdirectories) | ~34 MB |
| `data/noise_samples/noise_samples/collection_log.csv` | Collection metadata (timestamp, label, filename, RMS) | 53 KB |
| `data/noise_bank/noise_bank.npz` | Preprocessed noise waveforms grouped by noise category | 14.3 MB |
| `data/noise_bank/noise_config.json` | Group definitions, relevance mapping, SNR levels | 1 KB |
| `data/features/mel_spectrograms/train_noise_augmented.npz` | Mel-spectrogram features (3,317 x 40 x 92) | ~41 MB |
| `data/features/mfcc/train_noise_augmented.npz` | MFCC features (3,317 x 92 x 13) | ~14 MB |
| `data/normalization_stats_noise_augmented.npz` | Per-channel mean/std from noise-augmented training set | 1.4 KB |
| `data/noise_augmentation_report.json` | Full augmentation configuration and class distributions | 2 KB |

---

## 10. Limitations and Considerations

1. **Single vehicle.** All noise was collected from one vehicle (2008 VW Jetta). Different vehicles have different engine, road, and cabin noise characteristics. The model's noise robustness may not fully generalize to vehicles with substantially different acoustic profiles (e.g., electric vehicles, trucks, newer models with better sound insulation).

2. **Single collection session.** All samples were collected on a single day (March 22, 2026). Weather, road conditions, and traffic patterns on that day influence the noise bank. Multiple collection sessions across different days and conditions would improve diversity.

3. **No rain or extreme weather.** The collection did not include rain, snow, or extreme temperature conditions, which can alter both the noise environment and the vehicle's acoustic behavior.

4. **Domain mismatch with training data.** The Car Diagnostics Dataset was recorded on different vehicles under different conditions. The noise bank captures the deployment environment rather than the training data's recording environment. This is intentional --- the goal is to make the model robust to deployment-time noise --- but it means the augmented training data combines sounds from two different acoustic domains.

5. **PDM microphone limitations.** The MP34DT06JTR PDM microphone has a relatively flat frequency response from 100 Hz to 10 kHz but rolls off below 100 Hz, which means very low-frequency engine and road vibrations may be underrepresented in the noise bank.
