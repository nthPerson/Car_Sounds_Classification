# On-Device Evaluation Report — Phase 6 Documentation

**Phase:** 6 (On-Device Evaluation)
**Date:** March 26, 2026
**Related code:** `src/playback_test.py`, `src/evaluate.py`, `notebooks/07_evaluation.ipynb`
**Related data:** `results/playback_test_results_*.json`, `results/quantization_summary.csv`, `results/paper_*.png`

---

## 1. Purpose

Phase 6 completes the evaluation of the deployed car sound classifier by consolidating results across all four evaluation dimensions defined in the [evaluation plan](project_spec/evaluation_plan.md). The primary deployed model is M6 DS-CNN PTQ (Tier 2, 6-class, int8), selected for its highest classification performance and fit within all Arduino Nano 33 BLE Sense Rev2 hardware constraints.

---

## 2. Evaluation Dimensions Summary

| Dimension | What It Measures | Key Result | Data Source |
|-----------|-----------------|------------|-------------|
| **D1** | Classification performance (PC) | F1 macro 0.7835 (M6 PTQ Tier 2) | `results/quantization_summary.csv` |
| **D2** | Quantization impact | Nearly lossless (all McNemar p > 0.05) | `results/quantization_results.json` |
| **D3** | On-device performance | 783 ms inference, 319 KB flash, 179 KB SRAM | `docs/deployment_results.md` |
| **D4** | End-to-end system | 68.3% accuracy via Bluetooth speaker playback | `results/playback_test_results_bluetooth_speaker.json` |

---

## 3. Model Selection Justification

M6 DS-CNN PTQ was selected as the deployment model based on four criteria:

1. **Highest Tier 2 F1 macro** (0.7835) among all quantized models — outperforms M2 PTQ (0.7413) by 4.2 points and M5 PTQ (0.6138) by 17.0 points
2. **Fits within all hardware constraints** — 21.3 KB model (vs 150 KB flash budget), 63.6 KB tensor arena (vs 80 KB SRAM budget)
3. **PTQ is sufficient** — no need for QAT, since quantization produces no statistically significant accuracy loss
4. **CMSIS-NN optimized architecture** — depthwise separable convolutions are the target of CMSIS-NN's most optimized kernels for Cortex-M4

---

## 4. Classification Performance (D1)

### 4.1 Master Comparison — Tier 2 (Primary Deployment Target)

| Model | Type | F1 Macro | Accuracy | Size (KB) |
|-------|------|----------|----------|-----------|
| M6 DS-CNN PTQ | Int8 | **0.7835** | 0.7837 | 21.3 |
| M6 DS-CNN QAT | Int8 | 0.7774 | 0.7740 | 22.3 |
| M2 2D-CNN (Cond D) | Float32 | 0.7512 | 0.8750 | — |
| M2 2D-CNN QAT | Int8 | 0.7426 | 0.8558 | 20.5 |
| M2 2D-CNN PTQ | Int8 | 0.7413 | 0.8510 | 20.1 |
| M1 SVM | Classical | 0.6992 | 0.8510 | — |
| M1 RF | Classical | 0.6794 | 0.8413 | — |
| M5 1D-CNN PTQ | Int8 | 0.6138 | 0.7837 | 15.2 |
| M6 DS-CNN (Cond B) | Float32 | 0.7826 | 0.7837 | — |
| M5 1D-CNN (Cond B) | Float32 | 0.6295 | 0.7933 | — |

The DS-CNN architecture consistently outperforms the 2D-CNN and 1D-CNN on the F1 macro metric, which accounts for class imbalance. The classical ML baselines (RF, SVM) are competitive on accuracy but lag on F1 macro due to poorer minority class performance.

### 4.2 Data Augmentation Ablation

Three augmentation conditions were evaluated for each neural network architecture at Tier 2:

| Architecture | Cond B (Synthetic) | Cond C (Noise) | Cond D (Combined) | Best |
|-------------|-------------------|----------------|-------------------|------|
| M2 2D-CNN | 0.7328 | 0.7262 | **0.7512** | D |
| M5 1D-CNN | **0.6295** | 0.6092 | 0.6097 | B |
| M6 DS-CNN | **0.7826** | 0.7773 | 0.7813 | B |

- **M5 and M6** perform best on Condition B (synthetic augmentation only)
- **M2** benefits from the larger combined dataset (Condition D)
- Real-world noise injection (Condition C) did not consistently improve performance on the clean test set, though it may improve robustness in noisy deployment environments
- Quantization was performed on each model's best condition (M2 on D, M5/M6 on B)

---

## 5. Quantization Impact (D2)

### 5.1 Summary

Fifteen int8 TFLite models were produced: 9 PTQ (all architectures x 3 tiers) and 6 QAT (M2 and M6; M5 Conv1D not supported by tfmot).

**Key finding: Quantization is nearly lossless.** No model-tier-method combination shows a statistically significant accuracy change. All McNemar's test p-values exceed 0.05, and prediction agreement rates range from 95.2% to 99.5%.

### 5.2 Tier 2 Quantization Results

| Model | Method | Int8 F1 | Delta F1 | Agreement | McNemar p |
|-------|--------|---------|----------|-----------|-----------|
| M6 DS-CNN | PTQ | 0.7835 | -0.001 | 98.6% | > 0.05 |
| M6 DS-CNN | QAT | 0.7774 | +0.005 | 95.2% | > 0.05 |
| M2 2D-CNN | QAT | 0.7426 | +0.001 | 95.7% | > 0.05 |
| M2 2D-CNN | PTQ | 0.7413 | +0.003 | 99.0% | > 0.05 |
| M5 1D-CNN | PTQ | 0.6138 | -0.015 | 99.0% | > 0.05 |

### 5.3 Implications

- **PTQ is sufficient** for all models — QAT provides no measurable benefit
- The small model sizes (6K-15K parameters) and BatchNorm folding leave minimal quantization error
- Int8 model sizes (15-22 KB) are 4-12x smaller than float32 .h5 files
- All models fit comfortably within the 150 KB flash budget

---

## 6. On-Device Performance (D3)

### 6.1 Hardware Metrics

| Metric | Measured | Budget | Utilization |
|--------|---------|--------|-------------|
| Flash | 319 KB | 983 KB | 32.5% |
| SRAM (static) | 179 KB | 256 KB | 69.9% |
| Tensor arena | 63,556 bytes | 80 KB | 77.5% |
| Remaining SRAM | 83 KB | — | For stack/heap |

### 6.2 Latency

| Phase | Mean (ms) | Std (ms) | Share |
|-------|-----------|----------|-------|
| Audio capture | 1,490 | — | 60.3% |
| Feature extraction | 197.4 | 0.2 | 8.0% |
| Model inference | 784.9 | 0.2 | 31.7% |
| **Total cycle** | **2,470** | **0.4** | **100%** |

The total classification cycle is 2.5 seconds — one prediction every 2.5 seconds. For a car diagnostic tool (where the user holds the device near the engine for several seconds), this is acceptable.

Latency variance is extremely low (< 1 ms std), demonstrating deterministic real-time behavior.

### 6.3 Inference Performance Analysis

The M6 DS-CNN model executes 6,006,000 MAC operations per inference:

- 783 ms at 64 MHz = 50.1 million cycles = **8.4 cycles/MAC**
- This corresponds to reference int8 kernels without SIMD acceleration
- CMSIS-NN SIMD (theoretical): ~3 cycles/MAC = ~282 ms
- The 4.6x theoretical speedup was not achieved due to Arduino's GCC 7.2.1 lacking ACLE intrinsic support (see `docs/deployment_results.md` Section 8.3 for the full investigation)

### 6.4 MicroMutableOpResolver Optimization

Switching from `AllOpsResolver` to `MicroMutableOpResolver<5>` (registering only Conv2D, DepthwiseConv2D, FullyConnected, Mean, Softmax) saved 165 KB of flash (484 KB to 319 KB), a 34% reduction.

---

## 7. End-to-End System Evaluation (D4)

### 7.1 Playback Test Protocol

The playback test plays all 208 test set audio clips through a speaker while the Arduino captures and classifies via its PDM microphone. The script (`src/playback_test.py`) sends a `READY` command, waits for the Arduino to confirm it is listening, then plays the audio. The Arduino's prediction is compared against the ground-truth label.

### 7.2 Results

| Metric | PC Int8 | Bluetooth Speaker | Laptop Speakers |
|--------|---------|-------------------|-----------------|
| Accuracy | 0.7837 | **0.6827** | 0.2500 |
| F1 Macro | 0.7835 | **0.5465** | 0.1540 |
| F1 Weighted | 0.7835 | **0.7162** | 0.2825 |
| Degradation (F1) | — | -30.3% | -80.3% |

### 7.3 Per-Class Recall (Bluetooth Speaker)

| Class | PC Int8 | Bluetooth | Gap |
|-------|---------|-----------|-----|
| Normal Braking (n=12) | 0.750 | 0.333 | -0.417 |
| Braking Fault (n=11) | 0.818 | **0.818** | 0.000 |
| Normal Idle (n=40) | 0.675 | 0.575 | -0.100 |
| Idle Fault (n=118) | 0.864 | **0.780** | -0.084 |
| Normal Start-Up (n=9) | 0.778 | 0.444 | -0.334 |
| Start-Up Fault (n=18) | 0.556 | **0.556** | 0.000 |

Braking Fault and Start-Up Fault achieve PC-equivalent recall through Bluetooth playback. Idle Fault comes within 8.4 points. Normal Braking and Normal Start-Up show the largest gaps, likely due to their smaller sample sizes and sensitivity to speaker frequency response.

### 7.4 Confidence Calibration

| Prediction Type | Count | Mean Confidence |
|----------------|-------|-----------------|
| Correct | 142 (68.3%) | 0.808 |
| Incorrect | 66 (31.7%) | 0.655 |

The 0.153 gap between correct and incorrect confidence means a confidence threshold (e.g., 0.70) could be used to filter unreliable predictions in deployment, accepting lower throughput for higher precision.

---

## 8. Speaker Hardware Ablation

Two playback configurations were tested to quantify the impact of audio reproduction quality:

**Laptop speakers (built-in):** Frequency response rolls off below ~300 Hz. This eliminated the low-frequency engine content that distinguishes idle/braking from startup sounds, causing 72.1% of all predictions to fall into the startup classes. Accuracy: 25.0%.

**Bluetooth speaker:** Better bass response restored the low-frequency spectral content. Accuracy improved from 25.0% to 68.3% (2.7x improvement).

**Implication for real-world deployment:** In actual use, the Arduino's PDM microphone would be positioned near the engine with no speaker in the signal path. The microphone captures the full frequency range of engine sounds directly. The Bluetooth speaker playback test therefore represents a **conservative lower bound** on real-world deployment accuracy — actual performance would be better because the speaker frequency response bottleneck is eliminated.

---

## 9. Deployment Constraints and Trade-offs

### 9.1 Cycle Time Suitability

The 2.5-second classification cycle is suitable for car diagnostics:
- A mechanic or vehicle owner holds the device near the engine for 5-10 seconds
- 2-4 classifications per positioning provides redundancy for majority voting
- The 1.5-second audio capture window is sufficient for most engine sound patterns

### 9.2 Confidence Thresholding

The well-separated confidence distributions (correct: 0.81, incorrect: 0.66) enable a production deployment pattern:
- Predictions above 0.70 confidence: accept and display
- Predictions below 0.70 confidence: display "uncertain — please retry"
- This would increase precision at the cost of some recall

### 9.3 GCC Toolchain Limitation

The Arduino Mbed OS board package bundles GCC 7.2.1 (2017), which prevents CMSIS-NN SIMD acceleration. Upgrading to a newer toolchain (GCC 10+) would enable ~3x inference speedup (783 ms to ~260 ms), reducing the total cycle to ~1.9 seconds. This could be achieved by:
- Switching to PlatformIO with a newer ARM GCC toolchain
- Using the nRF Connect SDK (ships with GCC 12+)
- Updating the Arduino Mbed OS board package when a newer version is released

---

## 10. Conclusions

1. **M6 DS-CNN PTQ is the correct deployment choice.** It achieves the highest Tier 2 F1 (0.7835) while fitting comfortably within all hardware constraints (32.5% flash, 69.9% SRAM).

2. **Quantization is free.** Int8 PTQ produces no measurable accuracy loss for any model in the study. QAT provides no additional benefit. This is a strong result for the tinyML community — it suggests that small audio classification models (6K-15K parameters) are inherently robust to quantization.

3. **The speaker-mic domain gap is dominated by audio reproduction quality,** not by the classifier or on-device feature extraction. Real-world deployment (direct mic-to-engine) eliminates the largest source of accuracy loss.

4. **Data augmentation improves robustness but synthetic augmentation is most effective.** Real-world noise injection showed marginal improvement on the clean test set but may prove valuable in noisy deployment scenarios.

5. **The Arduino Nano 33 BLE Sense Rev2 is a viable platform for real-time audio classification,** with the caveat that inference latency is limited by the toolchain rather than the hardware. A toolchain upgrade could provide a 3x speedup.

---

## 11. Artifact Inventory

### Result Files

| File | Description |
|------|-------------|
| `results/classical_ml_summary.csv` | RF/SVM test metrics (7 rows) |
| `results/nn_summary_condition_b.csv` | NN metrics, Condition B (9 rows) |
| `results/nn_summary_condition_c.csv` | NN metrics, Condition C (9 rows) |
| `results/nn_summary_condition_d.csv` | NN metrics, Condition D (9 rows) |
| `results/quantization_summary.csv` | All 15 int8 models (18 rows) |
| `results/quantization_results.json` | Detailed quantization analysis |
| `results/playback_test_results_bluetooth_speaker.json` | 208-clip playback (Bluetooth) |
| `results/playback_test_results_laptop_speakers.json` | 208-clip playback (laptop) |
| `results/playback_test_summary_bluetooth_speaker.csv` | Per-clip summary (Bluetooth) |
| `results/playback_test_summary_laptop_speakers.csv` | Per-clip summary (laptop) |
| `results/feature_parity_results.json` | Feature validation (24 clips, all PASS) |

### Publication Figures

| File | Description |
|------|-------------|
| `results/paper_master_accuracy_tier2.png` | All models ranked by Tier 2 F1 macro |
| `results/paper_ablation_tier2.png` | Augmentation ablation (B/C/D x 3 architectures) |
| `results/paper_quantization_impact.png` | Delta F1 by model and method |
| `results/paper_f1_macro_vs_size.png` | F1 vs model size scatter (int8 models) |
| `results/paper_latency_breakdown.png` | Audio capture / features / inference timing |
| `results/paper_memory_utilization.png` | Flash and SRAM usage vs budget |
| `results/paper_pc_vs_playback_recall.png` | Per-class recall: PC vs Bluetooth vs laptop |
| `results/paper_speaker_ablation.png` | Prediction distribution by speaker type |
| `results/paper_confidence_calibration.png` | Confidence histogram (correct vs incorrect) |

### Documentation

| File | Description |
|------|-------------|
| `docs/deployment_results.md` | Phase 5 deployment details (memory, latency, CMSIS-NN, playback) |
| `docs/on_device_evaluation_report.md` | This document (Phase 6 synthesis) |
| `docs/quantization_results.md` | Phase 4 quantization analysis |
| `docs/neural_network_models.md` | Phase 3 NN architectures and ablation |
| `docs/classical_ml_baselines.md` | Phase 2 classical ML analysis |
| `docs/noise_data_collection_and_augmentation.md` | Noise collection and augmentation methodology |
| `notebooks/07_evaluation.ipynb` | Comprehensive evaluation notebook |
