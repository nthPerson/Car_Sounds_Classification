# Automotive Audio Anomaly Classification — Project Overview

**Project Title:** Real-Time Car Sound Classification with Quantized Neural Networks on Edge Hardware
**Course:** BDA 602
**Hardware Target:** Arduino Nano 33 BLE Sense Rev2
**Dataset:** [Car Diagnostics Dataset (Kaggle)](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset)
**Date:** March 10, 2026

---

## 1. Problem Statement

Modern vehicles produce characteristic audio signatures under different operating conditions. A healthy engine idling sounds distinctly different from one suffering low oil pressure; normal brake engagement produces a different acoustic profile than worn brake pads; and a clean ignition start-up is audibly distinct from a dead battery or faulty ignition system. Today, diagnosing these conditions requires either specialized OBD-II scanners, a trained mechanic's ear, or expensive sensor suites. There is an opportunity to build a low-cost, portable, always-on audio classifier that can detect common automotive faults in real time using only a microphone and a microcontroller.

This project investigates whether a machine learning model — trained on the Car Diagnostics Dataset (1,386 labeled WAV recordings across 13 conditions) — can be quantized from full-precision (float32) to integer (int8) representation and deployed on the **Arduino Nano 33 BLE Sense Rev2**, a constrained microcontroller with 1 MB flash, 256 KB SRAM, and a 64 MHz Arm Cortex-M4F processor.

The central research questions are:

1. **Can a neural network trained on this dataset achieve practically useful classification accuracy across multiple automotive fault conditions?**
2. **What is the quantitative impact of int8 quantization (both post-training and quantization-aware) on classification accuracy, model size, and inference latency compared to the full-precision baseline?**
3. **Can the resulting quantized model fit within the memory and compute constraints of the Arduino Nano 33 BLE Sense Rev2 while maintaining acceptable real-time inference performance?**

---

## 2. Project Goals

### 2.1 Primary Goals

| # | Goal | Success Criteria |
|---|------|-----------------|
| G1 | Build a complete ML pipeline for automotive audio classification | End-to-end pipeline from raw WAV files to trained model, with reproducible preprocessing, training, and evaluation |
| G2 | Train and compare multiple model architectures | At least one classical ML baseline and two neural network architectures trained and evaluated on the same data splits |
| G3 | Quantize neural network models to int8 | Both post-training quantization (PTQ) and quantization-aware training (QAT) applied and compared against float32 baselines |
| G4 | Deploy a quantized model on the Arduino Nano 33 BLE Sense Rev2 | At least one int8 model running on-device with real-time audio capture, on-device feature extraction, and inference output via Serial and/or BLE |
| G5 | Produce a comprehensive quantization impact analysis | Documented comparison of accuracy, model size, inference latency, and memory usage across float32, int8-PTQ, and int8-QAT variants |

### 2.2 Secondary Goals (Stretch)

| # | Goal | Notes |
|---|------|-------|
| S1 | Compare multiple class taxonomy tiers (2-class, 6-class, 9/13-class) | Analyze how reducing class count affects accuracy and deployability |
| S2 | Edge Impulse rapid prototype | Use Edge Impulse as an alternative toolchain for a quick feasibility check and compare against the custom TensorFlow pipeline |
| S3 | On-device validation with live audio playback | Play back dataset audio through a speaker and classify with the Nano's on-board microphone to measure real-world inference accuracy |
| S4 | BLE notification system | Transmit classification results over Bluetooth Low Energy to a companion app or dashboard |

---

## 3. Project Scope

### 3.1 In Scope

- **Data preprocessing:** Resampling, normalization, feature extraction (mel-spectrograms, MFCCs), data augmentation
- **Model development:** Classical ML baselines (Random Forest, SVM), compact CNNs (1-D and 2-D), depthwise separable CNNs
- **Quantization:** TensorFlow Lite post-training quantization (int8), quantization-aware training, model size and accuracy analysis
- **Deployment:** TFLite Micro on the Arduino Nano 33 BLE Sense Rev2, on-device audio capture via the PDM microphone, on-device feature extraction, inference, and result output
- **Evaluation:** Standard ML metrics (accuracy, F1, confusion matrices), quantization degradation analysis, on-device latency and memory profiling
- **Documentation:** Technical specification, experiment results, and a final report suitable for academic submission

### 3.2 Out of Scope

- **Collecting new audio data** — the project uses only the existing Kaggle dataset (supplemented by augmentation)
- **Multi-vehicle generalization** — the dataset likely represents a limited set of vehicles; generalizing across makes/models is not a goal
- **Production-grade reliability** — this is a proof-of-concept and academic study, not a product
- **Cloud inference or hybrid edge-cloud architectures** — all inference is on-device
- **Custom silicon or FPGA deployment** — the target is exclusively the Arduino Nano 33 BLE Sense Rev2

---

## 4. Non-Negotiable Constraints

These constraints are fixed and must be respected by all design decisions:

| Constraint | Detail |
|-----------|--------|
| **Target hardware** | Arduino Nano 33 BLE Sense Rev2 (Nordic nRF52840, Arm Cortex-M4F @ 64 MHz, 1 MB flash, 256 KB SRAM) |
| **On-board microphone** | ST MP34DT06JTR PDM MEMS microphone (64 dB SNR, practical sample rate up to 16 kHz after PDM decimation) |
| **ML framework for deployment** | TensorFlow Lite for Microcontrollers (LiteRT Micro) |
| **Quantization format** | Full-integer int8 quantization (required for CMSIS-NN optimized inference on Cortex-M4) |
| **Model flash budget** | ≤ 150 KB for the quantized model weights + graph (after accounting for OS, runtime, and application code) |
| **Model SRAM budget** | ≤ 80 KB for the TFLite tensor arena at inference time (after accounting for audio buffers, feature extraction workspace, and OS) |
| **Audio capture rate** | 16 kHz mono (matches the PDM mic's practical rate and minimizes on-device computation) |
| **Inference window** | 1.5 seconds of audio per classification (matching the dataset's clip duration) |

### 4.1 Memory Budget Breakdown

The 1 MB flash and 256 KB SRAM are shared among all system components. The following budget must be respected:

**Flash (1,024 KB total):**

| Component | Estimated Flash Usage |
|-----------|----------------------|
| Mbed OS + Arduino core + BLE stack | 250–350 KB |
| PDM driver | 10–20 KB |
| TFLite Micro runtime (with needed ops only) | 60–100 KB |
| Feature extraction code (FFT, mel filterbank, optional MFCC) | 15–30 KB |
| Application logic (main loop, Serial/BLE output) | 10–20 KB |
| **Remaining for model weights + graph** | **~150–250 KB** |

**SRAM (256 KB total):**

| Component | Estimated SRAM Usage |
|-----------|---------------------|
| Mbed OS + Arduino core + stack | 30–50 KB |
| PDM audio buffer (1.5 s × 16 kHz × 2 bytes, single-buffered) | ~48 KB |
| Feature extraction workspace (FFT scratch, mel output) | 5–15 KB |
| TFLite Micro runtime overhead | 10–20 KB |
| Application variables + BLE buffers | 5–15 KB |
| **Remaining for TFLite tensor arena (model activations)** | **~50–100 KB** |

> **Note:** The audio buffer is the single largest SRAM consumer after the OS. If memory becomes critically tight, reducing the inference window to 1.0 s (32 KB buffer) is an option — engine sounds are relatively stationary, so 1.0 s may still capture sufficient discriminative information. This would need to be validated experimentally.

---

## 5. Hardware Capabilities — Critical Details for Design Decisions

### 5.1 Processor: nRF52840 (Arm Cortex-M4F @ 64 MHz)

- **Hardware FPU:** The Cortex-M4**F** includes a single-precision hardware floating-point unit. This means float32 arithmetic IS supported in hardware — but only one operation at a time (scalar). The feasibility report's statement that there is "no hardware FPU optimization in TFLite Micro for float32" is imprecise. The real advantage of int8 quantization is that **CMSIS-NN provides SIMD-optimized int8 kernels** that pack four 8-bit values into a single 32-bit register and process them simultaneously, yielding 2–4× throughput improvement over scalar float32 FPU operations.
- **DSP extensions:** The Cortex-M4 includes single-cycle MAC (multiply-accumulate) and SIMD instructions that CMSIS-NN and CMSIS-DSP exploit for both inference and feature extraction (FFT, filtering).
- **No cache:** The Cortex-M4 has no data cache. Memory access patterns matter — sequential access to int8 arrays is significantly faster than strided access to float32 arrays due to reduced bus bandwidth.

### 5.2 On-Board Microphone: MP34DT06JTR

- **Interface:** PDM (Pulse Density Modulation) — outputs a 1-bit bitstream at a high clock rate (typically 1–3.25 MHz). The Arduino `PDM` library handles decimation filtering to produce 16-bit PCM samples at the desired sample rate.
- **Practical sample rate:** The `PDM` library on nRF52840 reliably supports 16 kHz output. Higher rates (e.g., 44.1 kHz) are theoretically possible but consume more CPU for decimation and are unnecessary for this application.
- **Frequency response:** The mic's frequency response extends to ~10 kHz with the PDM configuration used. At 16 kHz sample rate (Nyquist = 8 kHz), we capture the full range of engine sound fundamentals and dominant harmonics (typically below 5 kHz for idle, below 8 kHz for braking/startup events).
- **Sensitivity & SNR:** −26 dBFS sensitivity and 64 dB SNR are adequate for close-range engine sound capture. The 122.5 dBSPL acoustic overload point prevents clipping from loud engine events.

### 5.3 CMSIS-DSP and CMSIS-NN Libraries

These Arm-provided libraries are critical enablers for on-device feature extraction and inference:

- **CMSIS-DSP:** Provides optimized FFT (`arm_rfft_fast_f32`, `arm_rfft_q15`), filtering, and matrix operations. We will use this for computing the Short-Time Fourier Transform (STFT) and applying the mel filterbank on-device.
- **CMSIS-NN:** Provides optimized neural network kernels (convolution, depthwise convolution, fully connected, pooling, activation) for int8 inference. TFLite Micro automatically delegates to CMSIS-NN when available on Cortex-M targets.

---

## 6. Dataset Summary

The Car Diagnostics Dataset contains **1,386 WAV files** organized into three operational states:

| Operational State | Conditions | Files | % of Total |
|-------------------|-----------|-------|-----------|
| **Braking** | `normal_brakes` (77), `worn_out_brakes` (76) | 153 | 11.0% |
| **Idle** | `normal_engine_idle` (264), `low_oil` (107), `power_steering` (129), `serpentine_belt` (116), + 4 combined-fault categories (107–116 each) | 1,053 | 76.0% |
| **Start-Up** | `normal_engine_startup` (61), `bad_ignition` (62), `dead_battery` (57) | 180 | 13.0% |

**Audio properties:**
- Sample rates: Mixed (24 kHz, 44.1 kHz, 48 kHz) — must be resampled to 16 kHz
- Channels: 1 (mono throughout)
- Duration: 1.5 s (braking and idle), 1.7 s (start-up) — constant within each state
- Format: WAV (uncompressed PCM)

**Key challenges:**
- Significant class imbalance (264 vs. 57 files in extreme cases — 4.6:1 ratio)
- Small total dataset size (1,386 files) — high overfitting risk
- Likely studio/controlled recordings — domain gap vs. real-world noisy environments
- Mixed sample rates requiring standardization
- Combined-fault categories add multi-label complexity

Full dataset analysis is provided in [data_preprocessing_spec.md](data_preprocessing_spec.md).

---

## 7. Class Taxonomy Tiers

A key design decision is how many classes the deployed model should distinguish. More classes provide more diagnostic utility but require higher model capacity and more training data per class. We will train and evaluate three taxonomy tiers:

### Tier 1: Binary Anomaly Detection (2 classes)

| Class | Mapped Conditions |
|-------|------------------|
| **Normal** | `normal_brakes`, `normal_engine_idle`, `normal_engine_startup` |
| **Anomaly** | All fault conditions (including combined) |

- **Rationale:** Maximum robustness. With only 2 classes, even a small model can achieve high accuracy. The most practical "first alert" system — tells the user something is wrong, even if it can't say what.
- **Expected deployability:** Very high. Smallest model, lowest memory, fastest inference.

### Tier 2: State-Aware Anomaly Detection (6 classes) — **Primary deployment target**

| Class | Mapped Conditions |
|-------|------------------|
| **Normal Braking** | `normal_brakes` |
| **Braking Fault** | `worn_out_brakes` |
| **Normal Idle** | `normal_engine_idle` |
| **Idle Fault** | `low_oil`, `power_steering`, `serpentine_belt`, all combined-fault categories |
| **Normal Start-Up** | `normal_engine_startup` |
| **Start-Up Fault** | `bad_ignition`, `dead_battery` |

- **Rationale:** Good balance of diagnostic utility and feasibility. Distinguishes the operational state AND whether a fault is present, without requiring the model to identify the specific fault type. Collapsing all idle faults (including combined) into one class gives that class ~782 samples, which helps with imbalance.
- **Expected deployability:** High. 6 classes is well within the capacity of the target architectures.

### Tier 3: Full Fault Identification (9 classes, excluding combined faults)

| Class | Mapped Conditions |
|-------|------------------|
| **Normal Braking** | `normal_brakes` |
| **Worn Brakes** | `worn_out_brakes` |
| **Normal Idle** | `normal_engine_idle` |
| **Low Oil** | `low_oil` |
| **Power Steering Fault** | `power_steering` |
| **Serpentine Belt Fault** | `serpentine_belt` |
| **Normal Start-Up** | `normal_engine_startup` |
| **Bad Ignition** | `bad_ignition` |
| **Dead Battery** | `dead_battery` |

- **Rationale:** Maximum diagnostic specificity. Excludes the combined-fault categories to avoid multi-label ambiguity and keep the problem as single-label multi-class classification. Combined-fault samples (437 files) are excluded from this tier's training/evaluation.
- **Expected deployability:** Moderate. 9 classes with limited per-class samples (some as low as 57) makes this challenging, especially after quantization. May require the largest model variant.

### Decision on Combined-Fault Categories

The four combined-fault classes under the idle state represent vehicles with multiple simultaneous faults. These are problematic because:

1. They make single-label classification ambiguous (a "no oil + serpentine belt" sample could reasonably be labeled as either fault)
2. They increase the class count without adding clear diagnostic utility for the deployed model
3. Including them as separate classes dilutes per-class training data

**Our approach:**
- **Tier 1 & 2:** Combined faults are folded into "Anomaly" or "Idle Fault" — no ambiguity
- **Tier 3:** Combined faults are **excluded** from training and evaluation. They may optionally be used for robustness testing (does the model activate any fault class when presented with a multi-fault sample?)
- **Optional extension (Tier 3+):** If time permits, train a 13-class model including combined faults to report the full-taxonomy result for completeness

---

## 8. Technical Approach Summary

The project follows a four-stage pipeline:

### Stage 1: Data Preprocessing & Feature Extraction (PC)
- Resample all audio to 16 kHz mono
- Normalize amplitude (peak normalization or RMS normalization)
- Compute mel-spectrograms and/or MFCCs with windowing parameters matched to what will be computed on-device
- Apply data augmentation (time shift, noise injection, pitch perturbation, SpecAugment)
- Stratified train/validation/test split (70/15/15)
- Full specification: [data_preprocessing_spec.md](data_preprocessing_spec.md)

### Stage 2: Model Training & Selection (PC)
- Classical ML baselines (Random Forest, SVM) on hand-crafted feature vectors
- Compact 2-D CNN on mel-spectrograms
- 1-D CNN on MFCC sequences
- Depthwise separable CNN (DS-CNN) on mel-spectrograms
- All neural networks trained in float32 with TensorFlow/Keras
- Hyperparameter tuning via grid/random search with cross-validation
- Full specification: [model_architecture_spec.md](model_architecture_spec.md)

### Stage 3: Quantization & Conversion (PC)
- Post-training int8 quantization with representative dataset calibration
- Quantization-aware training with TensorFlow Model Optimization Toolkit
- Convert to TFLite format, then to C byte array for embedding in Arduino sketch
- Full specification: [quantization_deployment_spec.md](quantization_deployment_spec.md)

### Stage 4: On-Device Deployment & Evaluation (Arduino)
- Arduino sketch: PDM audio capture → on-device feature extraction → TFLite Micro inference → result output
- Measure inference latency, flash usage, SRAM usage
- Validate accuracy with audio playback tests
- Full specification: [quantization_deployment_spec.md](quantization_deployment_spec.md) and [evaluation_plan.md](evaluation_plan.md)

---

## 9. Document Index

This project's technical documentation is organized as follows:

| Document | Description |
|----------|-------------|
| [project_overview.md](project_overview.md) | This document — problem statement, goals, scope, constraints, and high-level approach |
| [deployment_feasibility_report.md](deployment_feasibility_report.md) | Original feasibility analysis — hardware specs, model options, quantization strategy, risk assessment |
| [data_preprocessing_spec.md](data_preprocessing_spec.md) | Dataset analysis, preprocessing pipeline, feature extraction parameters, augmentation strategy, data splitting |
| [model_architecture_spec.md](model_architecture_spec.md) | Model architectures, training configuration, hyperparameter search strategy, loss functions |
| [quantization_deployment_spec.md](quantization_deployment_spec.md) | Quantization methods, TFLite conversion, Arduino deployment, on-device inference loop |
| [evaluation_plan.md](evaluation_plan.md) | Evaluation methodology — ML metrics, quantization comparison, on-device benchmarking |
| [development_roadmap.md](development_roadmap.md) | Phased development plan with tasks, deliverables, dependencies, and milestones |

---

## 10. Key Terminology

| Term | Definition |
|------|-----------|
| **TFLite Micro** | TensorFlow Lite for Microcontrollers (also called LiteRT Micro) — a lightweight inference engine for running ML models on microcontrollers |
| **PTQ** | Post-Training Quantization — converting a trained float32 model to int8 after training, using a representative dataset for calibration |
| **QAT** | Quantization-Aware Training — inserting fake quantization nodes during training so the model learns to be robust to quantization effects |
| **CMSIS-NN** | Arm's Cortex Microcontroller Software Interface Standard — Neural Network library, providing optimized int8/int16 kernels for Cortex-M |
| **CMSIS-DSP** | Arm's DSP library for Cortex-M — provides optimized FFT, filtering, and matrix operations |
| **PDM** | Pulse Density Modulation — the digital interface used by the on-board MEMS microphone |
| **Mel-spectrogram** | A time-frequency representation of audio where the frequency axis is warped to the mel scale (approximating human pitch perception) |
| **MFCC** | Mel-Frequency Cepstral Coefficients — a compact representation derived from the mel-spectrogram via log compression and DCT |
| **Tensor arena** | The pre-allocated SRAM buffer used by TFLite Micro to store intermediate activations during inference |
| **SpecAugment** | A data augmentation technique that randomly masks contiguous blocks of frequency bands and/or time steps in a spectrogram |
| **DS-CNN** | Depthwise Separable Convolutional Neural Network — factorizes standard convolutions into depthwise and pointwise steps for efficiency |
