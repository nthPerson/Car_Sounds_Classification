# Automotive Audio Anomaly Classifier — Feasibility & Implementation Report

**Project:** Real-Time Car Sound Classification on Arduino Nano 33 BLE Sense Rev2  
**Dataset:** [Car Diagnostics Dataset (Kaggle)](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset)  
**Date:** March 2, 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset Analysis](#2-dataset-analysis)
   - 2.1 [Structure & Classes](#21-structure--classes)
   - 2.2 [Audio Properties](#22-audio-properties)
   - 2.3 [Class Balance](#23-class-balance)
   - 2.4 [Feature Separability](#24-feature-separability)
   - 2.5 [Dataset Limitations](#25-dataset-limitations)
3. [Target Hardware: Arduino Nano 33 BLE Sense Rev2](#3-target-hardware-arduino-nano-33-ble-sense-rev2)
   - 3.1 [Key Specifications](#31-key-specifications)
   - 3.2 [On-Board Microphone](#32-on-board-microphone)
   - 3.3 [Memory Budget for ML](#33-memory-budget-for-ml)
4. [Model Architecture Options](#4-model-architecture-options)
   - 4.1 [Option A — Compact CNN on Mel-Spectrogram](#41-option-a--compact-cnn-on-mel-spectrogram)
   - 4.2 [Option B — 1-D CNN on Raw/MFCC Features](#42-option-b--1-d-cnn-on-rawmfcc-features)
   - 4.3 [Option C — MicroNNs / DS-CNN (Depthwise Separable)](#43-option-c--micronns--ds-cnn-depthwise-separable)
   - 4.4 [Option D — Classical ML Baseline (for Comparison)](#44-option-d--classical-ml-baseline-for-comparison)
5. [Quantization Strategy](#5-quantization-strategy)
   - 5.1 [Full-Precision Baseline (Float32)](#51-full-precision-baseline-float32)
   - 5.2 [Post-Training Quantization (Int8)](#52-post-training-quantization-int8)
   - 5.3 [Quantization-Aware Training (QAT)](#53-quantization-aware-training-qat)
   - 5.4 [Expected Performance Impact](#54-expected-performance-impact)
6. [End-to-End Pipeline Design](#6-end-to-end-pipeline-design)
   - 6.1 [Training Pipeline (PC/Cloud)](#61-training-pipeline-pccloud)
   - 6.2 [Conversion & Deployment Pipeline](#62-conversion--deployment-pipeline)
   - 6.3 [On-Device Inference Loop](#63-on-device-inference-loop)
7. [Class Taxonomy Recommendations](#7-class-taxonomy-recommendations)
8. [Addressing Dataset Challenges](#8-addressing-dataset-challenges)
9. [Evaluation Plan: Full vs. Quantized](#9-evaluation-plan-full-vs-quantized)
10. [Alternative Toolchain: Edge Impulse](#10-alternative-toolchain-edge-impulse)
11. [Risk Assessment & Mitigations](#11-risk-assessment--mitigations)
12. [Recommended Implementation Roadmap](#12-recommended-implementation-roadmap)
13. [References](#13-references)

---

## 1. Executive Summary

This report evaluates the feasibility of training an audio classifier on the Car Diagnostics Dataset and deploying it in a quantized form on the **Arduino Nano 33 BLE Sense Rev2**. The key findings are:

- **Feasible with constraints.** The dataset provides 1,386 labeled WAV files across 3 operational states and 13 conditions. The short clip durations (1.5–1.7 s) and mono audio are well-suited for embedded inference windows.
- **Hardware is purpose-built for this task.** The Nano 33 BLE Sense Rev2 ships with an on-board PDM microphone (MP34DT06JTR), 256 KB SRAM, 1 MB flash, and a 64 MHz Arm Cortex-M4F — the reference platform for TensorFlow Lite for Microcontrollers (LiteRT).
- **Model must be very small.** After accounting for the runtime, PDM driver, and application code, the model should target **≤ 100–150 KB flash** and **≤ 50–80 KB SRAM** at inference time.
- **Int8 quantization is essential** to fit within these limits and is expected to reduce model size by ~4× with minimal accuracy degradation (typically < 2% loss).
- **Class count should be reduced** for the deployed model — a simplified 3–6 class taxonomy mapped from the 13 dataset conditions is recommended for reliable on-device performance.

---

## 2. Dataset Analysis

### 2.1 Structure & Classes

The dataset is organized into three operational states with a total of 13 conditions:

| Operational State | Condition | File Count |
|---|---|---|
| **Braking** | `normal_brakes` | 77 |
| | `worn_out_brakes` | 76 |
| **Idle** | `normal_engine_idle` | 264 |
| | `low_oil` | 107 |
| | `power_steering` | 129 |
| | `serpentine_belt` | 116 |
| | `combined/no oil_serpentine belt` | 107 |
| | `combined/power steering combined_no oil` | 107 |
| | `combined/power steering combined_no oil_serpentine belt` | 107 |
| | `combined/power steering combined_serpentine belt` | 116 |
| **Start-Up** | `normal_engine_startup` | 61 |
| | `bad_ignition` | 62 |
| | `dead_battery` | 57 |
| | **Total** | **1,386** |

The dataset is heavily weighted toward the **idle state** (1,053 files, 76% of the total), with braking (153 files, 11%) and start-up (180 files, 13%) being significantly smaller.

### 2.2 Audio Properties

| Property | Value |
|---|---|
| **Sample rates** | 24,000 Hz, 44,100 Hz, 48,000 Hz (mixed) |
| **Channels** | 1 (mono throughout) |
| **Duration — Braking** | 1.5 s (constant) |
| **Duration — Idle** | 1.5 s (constant) |
| **Duration — Start-Up** | 1.7 s (constant) |
| **File format** | WAV |

Key observations:
- **Mixed sample rates** across files will require resampling to a common rate during preprocessing. A target of **16 kHz** is recommended for TinyML deployment (matches the on-board mic's practical range and reduces computation).
- All files are **mono**, which is ideal — no channel-mixing needed.
- Short, fixed-length durations (1.5–1.7 s) align well with the inference window sizes used in embedded audio classification (typically 1–2 s).

### 2.3 Class Balance

The dataset exhibits significant class imbalance:
- `normal_engine_idle` has **264 files** — 3.4× to 4.6× more than the smallest classes.
- Start-up classes (`dead_battery`: 57, `normal_engine_startup`: 61) are the most under-represented.
- The four "combined" idle fault categories each have ~107–116 files but represent multi-fault scenarios.

**Implication:** Data augmentation (time-shift, noise injection, pitch perturbation, SpecAugment) and class-weighted loss functions will be necessary during training to prevent bias toward the majority class.

### 2.4 Feature Separability

The EDA notebook computes four lightweight features — RMS energy, zero-crossing rate (ZCR), spectral centroid, and spectral bandwidth — for every file. Box-plot analysis shows:

- **Braking classes** (normal vs. worn-out) show moderate separation on RMS energy and spectral bandwidth.
- **Idle classes** show the strongest separation, particularly between `normal_engine_idle` and fault conditions on spectral centroid and ZCR.
- **Start-up classes** show visible separation on RMS energy (`dead_battery` has notably lower energy than `normal_engine_startup` and `bad_ignition`).

These results indicate that even basic spectral features carry discriminative information, which is encouraging for more expressive representations like MFCCs and mel-spectrograms used in deep learning models.

### 2.5 Dataset Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **Small dataset** (1,386 total files) | Risk of overfitting, especially with deeper models | Data augmentation, transfer learning, regularization |
| **Mixed sample rates** (24k / 44.1k / 48k) | Inconsistent frequency resolution | Resample all to a common rate (16 kHz) during preprocessing |
| **Studio recordings** (likely) | Domain gap vs. real-world noisy environments | Noise augmentation during training; may need supplementary real-world recordings |
| **Limited vehicle diversity** | Model may not generalize across vehicle makes/models | Document limitation; consider as proof-of-concept scope |
| **Combined fault categories** | Multi-label nature adds complexity | Treat as separate classes or collapse into "multi-fault" |
| **Short clip length** (1.5–1.7 s) | May not capture transient events fully | Acceptable for embedded inference windows; consider overlapping windows |

---

## 3. Target Hardware: Arduino Nano 33 BLE Sense Rev2

### 3.1 Key Specifications

| Specification | Value |
|---|---|
| **Processor** | Nordic nRF52840 — Arm Cortex-M4F @ 64 MHz |
| **Flash** | 1 MB |
| **SRAM** | 256 KB |
| **Connectivity** | Bluetooth 5 Low Energy (NINA-B306 module) |
| **USB** | Micro-USB |
| **Operating voltage** | 3.3 V |
| **Dimensions** | 45 mm × 18 mm |
| **Weight** | 5 g |
| **OS** | Arm Mbed OS |
| **ML Framework Support** | TensorFlow Lite for Microcontrollers (LiteRT), Edge Impulse |

### 3.2 On-Board Microphone

| Property | Value |
|---|---|
| **Sensor** | ST MP34DT06JTR — MEMS omnidirectional digital microphone |
| **Interface** | PDM (Pulse Density Modulation) |
| **SNR** | 64 dB |
| **Sensitivity** | −26 dBFS ±1 dB |
| **AOP** | 122.5 dBSPL |
| **Arduino Library** | `PDM` (built-in) |
| **Practical sample rate** | Up to 16 kHz (after PDM decimation, commonly used at 16 kHz in TinyML examples) |

The on-board microphone is well-suited for capturing engine/vehicle sounds. The 64 dB SNR and 122.5 dBSPL acoustic overload point mean it can handle both quiet idle sounds and louder braking/start-up events without clipping.

**Important note:** The MP34DT06JTR is marked as **OBSOLETE** by ST. While existing boards are unaffected, future board revisions could use a different microphone. This does not affect the current project.

### 3.3 Memory Budget for ML

The 1 MB flash and 256 KB SRAM must be shared between:

| Component | Estimated Flash | Estimated SRAM |
|---|---|---|
| Mbed OS + Arduino core | ~250–350 KB | ~30–50 KB |
| PDM driver + audio buffer | ~10–20 KB | ~16–32 KB (double-buffered 1 s @ 16 kHz × 16-bit) |
| TFLite Micro runtime | ~50–80 KB | ~10–20 KB |
| Feature extraction (MFCC/mel) | ~10–20 KB | ~5–10 KB |
| Application logic + BLE stack | ~30–50 KB | ~10–20 KB |
| **Available for model weights** | **~100–200 KB** | **~50–100 KB** |

**Target budget for the quantized model:**
- **Flash (weights + model graph):** ≤ 100–150 KB
- **SRAM (tensor arena):** ≤ 50–80 KB

A float32 model of this size would correspond to ~25K–37K parameters. With int8 quantization, you can fit ~100K–150K parameters in the same flash budget — significantly more expressive.

---

## 4. Model Architecture Options

### 4.1 Option A — Compact CNN on Mel-Spectrogram

**Architecture:** Small 2-D CNN operating on a mel-spectrogram image (e.g., 32 × 32 or 40 × 25 mel-frequency × time bins).

```
Input: Mel-spectrogram [40 × T × 1]
→ Conv2D(16, 3×3) → BN → ReLU → MaxPool(2×2)
→ Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)
→ Conv2D(32, 3×3) → BN → ReLU → GlobalAvgPool
→ Dense(N_classes) → Softmax
```

| Metric | Estimate |
|---|---|
| Parameters | ~15K–30K |
| Int8 model size | ~20–40 KB |
| Tensor arena | ~20–40 KB |
| Inference latency | ~50–200 ms |

**Pros:** Proven architecture for audio classification; good accuracy; mel-spectrogram captures both time and frequency structure.  
**Cons:** Requires computing mel-spectrogram on-device (feasible but adds code complexity); 2-D convolutions are more compute-intensive than 1-D.

### 4.2 Option B — 1-D CNN on Raw/MFCC Features

**Architecture:** 1-D CNN operating on a time-series of MFCC frames.

```
Input: MFCC frames [T × 13]
→ Conv1D(16, 5) → BN → ReLU → MaxPool(2)
→ Conv1D(32, 3) → BN → ReLU → MaxPool(2)
→ Conv1D(32, 3) → BN → ReLU → GlobalAvgPool
→ Dense(N_classes) → Softmax
```

| Metric | Estimate |
|---|---|
| Parameters | ~8K–20K |
| Int8 model size | ~10–25 KB |
| Tensor arena | ~10–25 KB |
| Inference latency | ~20–100 ms |

**Pros:** Smaller and faster; MFCCs are a compact representation; well-suited for the Cortex-M4F.  
**Cons:** MFCCs discard some spectral detail; may be slightly less accurate than mel-spectrogram CNN.

### 4.3 Option C — MicroNNs / DS-CNN (Depthwise Separable)

**Architecture:** Depthwise separable CNN (DS-CNN), as used in the TensorFlow "micro speech" example for keyword spotting.

```
Input: Mel-spectrogram [40 × T × 1]
→ Conv2D(64, 10×4) → BN → ReLU
→ DepthwiseConv2D(3×3) → Conv2D(64, 1×1) → BN → ReLU  (×4 blocks)
→ AveragePool → Dense(N_classes) → Softmax
```

| Metric | Estimate |
|---|---|
| Parameters | ~20K–50K |
| Int8 model size | ~25–60 KB |
| Tensor arena | ~30–60 KB |
| Inference latency | ~100–300 ms |

**Pros:** State-of-the-art for keyword spotting on MCUs; depthwise separable convolutions are efficient.  
**Cons:** Larger than Options A/B; requires careful tuning to stay within memory.

### 4.4 Option D — Classical ML Baseline (for Comparison)

For the academic comparison aspect of your project, training a classical ML model (Random Forest, SVM) on hand-crafted features (MFCCs, spectral centroid, RMS, ZCR, etc.) on the PC is valuable as a performance baseline even though it isn't straightforward to deploy on-device.

---

## 5. Quantization Strategy

### 5.1 Full-Precision Baseline (Float32)

- Train the model in float32 using TensorFlow/Keras.
- Model size: ~4× larger than int8 equivalent.
- **Not directly deployable** on the Nano 33 BLE Sense (too large, no hardware FPU optimization in TFLite Micro for float32 on Cortex-M4).
- Serves as the **accuracy ceiling** for comparison.

### 5.2 Post-Training Quantization (Int8)

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen  # calibration set
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()
```

- **Size reduction:** ~4× (float32 → int8).
- **Accuracy impact:** Typically < 1–2% loss on audio classification tasks.
- **Latency improvement:** ~2–4× faster inference (CMSIS-NN optimized int8 kernels on Cortex-M4).
- **Requirement:** A **representative dataset** (100–500 samples) for calibration.

### 5.3 Quantization-Aware Training (QAT)

If post-training quantization causes unacceptable accuracy loss:

```python
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)
q_aware_model.compile(...)
q_aware_model.fit(...)
```

- Simulates quantization effects during training.
- Typically recovers 0.5–1% compared to post-training quantization.
- **Recommended** if your team wants the best possible quantized accuracy.

### 5.4 Expected Performance Impact

| Metric | Float32 | Post-Training Int8 | QAT Int8 |
|---|---|---|---|
| Model size | 1× (baseline) | ~0.25× | ~0.25× |
| Inference latency | 1× (baseline) | ~0.3–0.5× | ~0.3–0.5× |
| Accuracy | Baseline | −0.5–2% | −0–1% |
| Deployable on Nano 33? | ❌ Likely too large | ✅ Yes | ✅ Yes |

---

## 6. End-to-End Pipeline Design

### 6.1 Training Pipeline (PC/Cloud)

```
Raw WAV files (1,386 × 1.5–1.7 s)
  │
  ├─ Resample to 16 kHz mono
  ├─ Normalize amplitude
  ├─ Train/Val/Test split (stratified by condition, e.g., 70/15/15)
  │
  ├─ Feature extraction:
  │    ├─ Mel-spectrogram (40 mel bands, 25 ms window, 10 ms hop)
  │    └─ or MFCCs (13 coefficients, same windowing)
  │
  ├─ Data augmentation:
  │    ├─ Time shift (±100 ms)
  │    ├─ Background noise injection (SNR 5–20 dB)
  │    ├─ Pitch perturbation (±10%)
  │    └─ SpecAugment (frequency/time masking)
  │
  ├─ Model training (TensorFlow/Keras)
  │    ├─ Float32 baseline
  │    └─ Quantization-aware training variant
  │
  └─ Evaluation:
       ├─ Accuracy, F1-score (macro), confusion matrix
       └─ Per-class precision/recall
```

### 6.2 Conversion & Deployment Pipeline

```
Trained Keras model (.h5 / SavedModel)
  │
  ├─ TFLite Converter → .tflite (float32)     ← comparison artifact
  ├─ TFLite Converter → .tflite (int8 PTQ)    ← deployable
  ├─ TFLite Converter → .tflite (int8 QAT)    ← deployable
  │
  ├─ xxd -i model.tflite > model_data.h       ← C byte array
  │
  └─ Arduino sketch:
       ├─ #include "model_data.h"
       ├─ TFLite Micro interpreter setup
       ├─ PDM audio capture → feature extraction → inference
       └─ Output: predicted class + confidence → Serial / BLE
```

### 6.3 On-Device Inference Loop

```
┌─────────────────────────────────────────────┐
│              Arduino Nano 33 BLE            │
│                                             │
│  PDM Microphone                             │
│       │                                     │
│       ▼                                     │
│  Audio Buffer (1.5 s @ 16 kHz = 24,000     │
│  samples × 2 bytes = 48 KB)                │
│       │                                     │
│       ▼                                     │
│  Feature Extraction                         │
│  (Mel-spectrogram or MFCC, on-device)       │
│       │                                     │
│       ▼                                     │
│  TFLite Micro Interpreter                   │
│  (Int8 quantized model)                     │
│       │                                     │
│       ▼                                     │
│  Output: Class label + confidence           │
│       │                                     │
│       ├──► Serial Monitor (debug)           │
│       └──► BLE notification (production)    │
└─────────────────────────────────────────────┘
```

**Timing estimate** for a single inference cycle:
- Audio capture: 1.5 s (real-time)
- Feature extraction: ~20–50 ms
- Model inference: ~50–200 ms
- **Total latency:** ~1.6–1.8 s per classification (dominated by audio capture window)

For near-real-time operation, a **sliding window** with 50% overlap can produce a classification every ~0.75 s.

---

## 7. Class Taxonomy Recommendations

The full 13-class taxonomy is aggressive for a TinyML model with limited training data. We recommend the following tiered approach:

### Tier 1: Binary Anomaly Detection (simplest, most robust)

| Class | Mapped Conditions |
|---|---|
| **Normal** | `normal_brakes`, `normal_engine_idle`, `normal_engine_startup` |
| **Anomaly** | All fault conditions |

- Estimated accuracy: **85–95%** (int8 quantized)
- Smallest model, most reliable for deployment

### Tier 2: State-Aware Anomaly Detection (recommended)

| Class | Mapped Conditions |
|---|---|
| **Normal Braking** | `normal_brakes` |
| **Braking Fault** | `worn_out_brakes` |
| **Normal Idle** | `normal_engine_idle` |
| **Idle Fault** | `low_oil`, `power_steering`, `serpentine_belt`, all combined |
| **Normal Start-Up** | `normal_engine_startup` |
| **Start-Up Fault** | `bad_ignition`, `dead_battery` |

- 6 classes — good balance of utility and accuracy
- Estimated accuracy: **75–88%** (int8 quantized)

### Tier 3: Full Fault Identification (most ambitious)

All 13 classes (or 9, excluding combined faults). Achievable in float32 on PC, but challenging to maintain accuracy after quantization with only ~1,386 training samples.

**Recommendation:** Start with **Tier 2** as the deployment target, train and evaluate all three tiers for the academic comparison, and report the accuracy/size trade-offs.

---

## 8. Addressing Dataset Challenges

| Challenge | Recommended Approach |
|---|---|
| **Class imbalance** | (1) Class-weighted loss function: `class_weight` in `model.fit()`. (2) Data augmentation focused on minority classes. (3) SMOTE on extracted features for the classical ML baseline. |
| **Mixed sample rates** | Resample all files to 16 kHz at the start of the pipeline. Use `librosa.load(path, sr=16000)`. |
| **Small dataset size** | (1) Aggressive augmentation (5–10× expansion). (2) Transfer learning from a pre-trained audio model (e.g., YAMNet backbone), then fine-tune + convert. (3) Keep model small to avoid overfitting. |
| **Domain gap (studio vs. real-world)** | (1) Add real-world noise augmentation (road noise, wind, other vehicles). (2) If possible, record a small set of validation samples with the actual Nano mic in a real car. |
| **Combined fault classes** | Option A: Treat as separate multi-fault classes. Option B: Collapse all combined into a single "multi-fault" class. Option C: Exclude from classification and focus on single-fault identification. |

---

## 9. Evaluation Plan: Full vs. Quantized

For the master's project, the following comparison framework is recommended:

### Models to Compare

| Model ID | Architecture | Quantization | Deployable? |
|---|---|---|---|
| M1 | Classical ML (RF/SVM) | N/A | ❌ (PC only) |
| M2 | CNN (Tier 2, 6-class) | Float32 | ❌ (too large for MCU) |
| M3 | CNN (Tier 2, 6-class) | Int8 PTQ | ✅ |
| M4 | CNN (Tier 2, 6-class) | Int8 QAT | ✅ |
| M5 | DS-CNN (Tier 2, 6-class) | Int8 QAT | ✅ |

### Metrics to Report

| Metric | Where Measured |
|---|---|
| **Accuracy** (overall & per-class) | PC evaluation on held-out test set |
| **F1-score** (macro, to handle imbalance) | PC evaluation |
| **Confusion matrix** | PC evaluation |
| **Model size** (KB) | .tflite file size |
| **Inference latency** (ms) | On-device measurement |
| **Peak SRAM usage** (KB) | TFLite arena size + profiling |
| **Flash usage** (KB) | Arduino IDE compilation report |
| **Power consumption** (mW, if measurable) | On-device measurement |

### Suggested Reporting Table

| | M1 (RF) | M2 (F32) | M3 (Int8 PTQ) | M4 (Int8 QAT) | M5 (DS-CNN QAT) |
|---|---|---|---|---|---|
| Accuracy | — | — | — | — | — |
| F1 (macro) | — | — | — | — | — |
| Model size (KB) | N/A | — | — | — | — |
| Latency (ms) | N/A | N/A | — | — | — |
| Flash (KB) | N/A | N/A | — | — | — |
| SRAM (KB) | N/A | N/A | — | — | — |

---

## 10. Alternative Toolchain: Edge Impulse

[Edge Impulse](https://www.edgeimpulse.com/) provides a fully integrated, no-code/low-code platform that directly supports the Arduino Nano 33 BLE Sense (both Rev1 and Rev2). It is worth considering as a complementary or alternative approach:

### Advantages
- **End-to-end workflow:** Data upload → feature extraction (MFCC/MFE/spectrogram) → model training → on-device testing → Arduino library export — all in a web GUI.
- **Automatic quantization:** Generates int8 models optimized for the target board.
- **On-device testing:** Flash firmware to collect live data from the board and run inference tests.
- **Profiling:** Provides estimated latency, flash, and RAM usage before deployment.
- **Arduino library export:** Generates a `.zip` library you can directly include in the Arduino IDE.

### Disadvantages
- **Less control** over model architecture and quantization strategy.
- **Free tier limits** on dataset size and compute.
- Less suitable for detailed academic analysis of quantization effects.

### Recommendation
Use Edge Impulse as a **rapid prototyping baseline** to validate feasibility quickly, then implement the custom TensorFlow pipeline for the detailed quantization comparison that your team intends to investigate.

---

## 11. Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Model doesn't fit in flash | Medium | High | Start with smallest architecture (Option B); profile early; target ≤100 KB int8 |
| Tensor arena exceeds SRAM | Medium | High | Reduce input dimensions (fewer mel bins, shorter window); use `MicroInterpreter` arena sizing tool |
| Poor real-world accuracy (domain gap) | High | Medium | Augment with noise; validate with real recordings; communicate as limitation |
| Dataset too small for 13 classes | High | Medium | Use Tier 1 or Tier 2 taxonomy; heavy augmentation |
| Microphone frequency response limits feature quality | Low | Medium | Validate spectrograms captured on-device vs. dataset; 16 kHz is sufficient for engine sounds (dominant energy < 8 kHz) |
| Quantization causes severe accuracy drop | Low | Medium | Use QAT; mixed-precision quantization if needed |
| BLE latency for real-time alerts | Low | Low | BLE is only for reporting, not inference-critical path |

---

## 12. Recommended Implementation Roadmap

| Phase | Tasks | Duration (est.) |
|---|---|---|
| **Phase 1: Data Prep** | Resample to 16 kHz; stratified train/val/test split; augmentation pipeline; extract features (mel-spectrogram + MFCC) | 1 week |
| **Phase 2: Baseline Models** | Train classical ML (RF/SVM) on hand-crafted features; train float32 CNN (Options A & B) on all three class tiers | 1–2 weeks |
| **Phase 3: Quantization** | Post-training int8 quantization; quantization-aware training; evaluate accuracy vs. float32 baseline; model size analysis | 1 week |
| **Phase 4: Deployment** | Convert best int8 model to C array; write Arduino sketch (PDM capture → features → inference → output); flash to board | 1–2 weeks |
| **Phase 5: On-Device Eval** | Measure latency, flash/SRAM usage, inference accuracy with live audio playback; document environment setup | 1 week |
| **Phase 6: Comparison Report** | Compile full vs. quantized comparison tables; write up findings; prepare presentation | 1 week |

**Total estimated timeline: 6–8 weeks** (with parallel work across team members).

---

## 13. References

1. **Car Diagnostics Dataset** — Kaggle. https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset
2. **Arduino Nano 33 BLE Sense Rev2 — Datasheet** — Arduino. https://docs.arduino.cc/resources/datasheets/ABX00069-datasheet.pdf
3. **Arduino Nano 33 BLE Sense Rev2 — Tech Specs** — Arduino. https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2/
4. **LiteRT for Microcontrollers (TensorFlow Lite Micro)** — Google. https://ai.google.dev/edge/litert/microcontrollers
5. **Build and Convert Models for Microcontrollers** — Google. https://ai.google.dev/edge/litert/microcontrollers/build_convert
6. **MP34DT06JTR MEMS Microphone** — STMicroelectronics. https://www.st.com/en/mems-and-sensors/mp34dt06j.html
7. **Edge Impulse — Arduino Nano 33 BLE Sense** — Edge Impulse. https://docs.edgeimpulse.com/docs/edge-ai-hardware/mcu/arduino-nano-33-ble-sense
8. **TensorFlow Model Optimization Toolkit** — TensorFlow. https://www.tensorflow.org/model_optimization
9. **Micro Speech Example (keyword spotting)** — TFLite Micro. https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech
10. **CMSIS-NN** — Arm. https://github.com/ARM-software/CMSIS-NN (optimized neural network kernels for Cortex-M)
