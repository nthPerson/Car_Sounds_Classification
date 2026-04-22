# Car Sound Classification on a Microcontroller

**BDA 602 — Real-Time Car Sound Classification with Quantized Neural Networks on the Arduino Nano 33 BLE Sense Rev2**

An end-to-end pipeline that trains a neural network to diagnose car engine faults from audio, quantizes it to 8-bit integers, and deploys it on a $30 microcontroller that classifies sounds in real time — with no cloud connection, no GPU, and all processing happening on-device.

---

## What This Project Does

1. **Listens** to 1.5-second clips of engine audio through the Arduino's on-board MEMS microphone
2. **Extracts** a log-mel spectrogram on-device using CMSIS-DSP optimized signal processing
3. **Classifies** the sound into one of six categories (three operational states x normal/fault) using a quantized depthwise separable CNN
4. **Reports** the prediction with a confidence score over Serial in under 2.5 seconds total

The complete system runs on an Arduino Nano 33 BLE Sense Rev2 (Nordic nRF52840, Cortex-M4F @ 64 MHz) with 1 MB of flash and 256 KB of SRAM — no external accelerator, no cloud inference, no phone required.

---

## Why It Matters

Diagnosing car problems by sound traditionally requires either an experienced mechanic or an expensive OBD-II scanner. A cheap, always-on acoustic classifier could democratize predictive maintenance for fleets, individual owners, and repair shops — and because all processing happens on-device, no audio ever leaves the sensor, making the approach privacy-preserving by design.

Beyond the practical application, this project is also a case study in **tinyML deployment**: it demonstrates how far modern neural networks can be compressed and optimized to fit on a constrained microcontroller, and documents the engineering trade-offs encountered along the way (quantization, toolchain limitations, speaker-microphone domain gaps, and more).

---

## Key Results

### Classification Performance (Tier 2, six-class state-aware task)

| Rank | Model | Type | Accuracy | F1 Macro | Size |
|------|-------|------|----------|----------|------|
| **1** | **M6 DS-CNN PTQ** (deployed) | **Int8** | **0.8702** | **0.7835** | **21.3 KB** |
| 2 | M6 DS-CNN QAT | Int8 | 0.8750 | 0.7774 | 22.3 KB |
| 3 | M2 2D-CNN | Float32 | 0.8750 | 0.7512 | 237 KB |
| 4 | SVM (baseline) | Classical | 0.8510 | 0.6992 | — |
| 5 | Random Forest (baseline) | Classical | 0.8413 | 0.6794 | — |

The deployed model is a depthwise separable CNN with 8,166 parameters. After post-training int8 quantization, it occupies 21.3 KB on disk — a 12x compression from its 246 KB float32 checkpoint — and beats both classical ML baselines by 8-10 F1 macro points.

### Quantization Is Nearly Lossless

Across all 15 quantized models evaluated (three architectures x three tiers x PTQ/QAT), **no combination shows a statistically significant accuracy change** after quantization (McNemar's test, all p > 0.05). For models at this scale (6K-15K parameters), PTQ is sufficient and QAT confers no measurable benefit.

### On-Device Performance (Arduino Nano 33 BLE Sense Rev2)

| Metric | Measured | Budget | Utilization |
|--------|----------|--------|-------------|
| Flash | 319 KB | 983 KB | 33% |
| SRAM (static) | 175 KB | 256 KB | 68% |
| Tensor arena | 63.6 KB | 80 KB | 77.5% |
| Audio capture | 1,490 ms | — | 60.3% of cycle |
| Feature extraction (CMSIS-DSP) | 197 ms | — | 8.0% of cycle |
| Model inference (TFLite Micro) | 785 ms | — | 31.7% of cycle |
| **Total cycle** | **2,472 ms** | — | one prediction every ~2.5 s |

### End-to-End Playback Evaluation

A speaker-to-microphone playback test on all 208 test clips evaluated the complete deployed system. Results were strongly influenced by playback hardware:

| Configuration | Accuracy | F1 Macro | Note |
|---------------|----------|----------|------|
| PC Int8 (baseline) | 0.8702 | 0.7835 | Reference |
| Bluetooth speaker | 0.6827 | 0.5465 | 30.3% F1 degradation |
| Laptop speakers | 0.2500 | 0.1540 | 80.3% degradation — rolloff below 300 Hz destroyed bass cues |

This speaker ablation study isolates playback quality as the dominant source of the on-device domain gap; the classifier itself transfers faithfully. In real deployment (microphone directly near the engine, no speaker in the signal path), performance should approach the PC baseline.

---

## Unique Contributions

- **Real-world noise corpus collected with the deployment hardware.** 360 background noise samples captured from a 2008 Volkswagen Jetta using the same Arduino Nano 33 BLE Sense Rev2 and PDM microphone that run inference — spanning engine, HVAC, road, wind, cabin, and ambient categories, with operational-state-aware mixing during augmentation.
- **Speaker hardware ablation.** Two playback configurations (laptop speakers and a Bluetooth speaker) isolate audio reproduction quality as the dominant factor in the observed domain gap, providing a reference point for what to expect in real deployment versus benchtop testing.
- **CMSIS-NN toolchain investigation.** A detailed diagnosis of a latent limitation in the Arduino Mbed OS board package: the bundled GCC 7.2.1 toolchain lacks the ACLE intrinsic support needed for CMSIS-NN SIMD acceleration, capping int8 inference at 8.4 cycles per MAC rather than the 3 cycles/MAC achievable with a modern toolchain.
- **Custom TFLite Micro build from source.** The archived `Arduino_TensorFlowLite` library was rebuilt from the [official tflite-micro repository](https://github.com/tensorflow/tflite-micro) with CMSIS-NN kernels enabled and packaged as a drop-in Arduino library, with all packaging gotchas documented.
- **Feature-parity validation.** A Python simulation of the on-device CMSIS-DSP feature extraction confirmed bit-level parity (cosine similarity 1.0000, 96.3% int8 exact match) with the training-time librosa pipeline before any hardware was flashed.

---

## Potential Applications

- **Predictive fleet maintenance.** Equip commercial vehicles with acoustic sensors that continuously monitor for fault signatures and flag issues before they cause breakdowns.
- **Consumer vehicle health monitoring.** A battery-powered device clipped under the hood could stream classification results over BLE to a smartphone app, supplementing or replacing dealership diagnostics for common faults.
- **Handheld mechanic diagnostic tool.** Point-and-classify: a mechanic holds the device near the engine for 5-10 seconds and receives a ranked list of likely fault conditions with confidence scores.
- **Privacy-preserving edge AI template.** Because no audio leaves the device, this approach is directly applicable to other domains with privacy-sensitive acoustic monitoring (home, medical, industrial) where cloud-based classification is undesirable.

---

## The Dataset

The [Car Diagnostics Dataset](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset) from Kaggle — 1,386 audio recordings across 13 engine conditions organized into three operational states (braking, idle, startup). The classification task is organized into three taxonomy tiers:

- **Tier 1 (2 classes):** Normal vs. Anomaly
- **Tier 2 (6 classes, primary target):** State-aware Normal/Fault (Normal Braking, Braking Fault, Normal Idle, Idle Fault, Normal Start-Up, Start-Up Fault)
- **Tier 3 (9 classes):** Individual fault identification

All experiments used a 70/15/15 train/val/test split (970/208/208 samples) with stratification by condition and `random_state=42`.

---

## How It Works (End-to-End Pipeline)

1. **Preprocessing (PC):** Resample to 16 kHz mono, peak-normalize amplitude, and crop/pad to exactly 1.5 seconds (24,000 samples).
2. **Feature extraction (PC + on-device):** Compute log-mel spectrograms (40 mel bands x 92 frames, n_fft=512, hop_length=256). The Python pipeline uses librosa; the Arduino pipeline replicates it using CMSIS-DSP with a sparse mel filterbank (95% sparse, 2.1 KB flash versus 40 KB dense).
3. **Data augmentation (PC):** Four training conditions compared in an ablation study (original; synthetic waveform augmentation with class balancing; real-world noise injection; combined). Synthetic augmentation + class balancing produced the single largest accuracy improvement (+0.206 F1 macro for M6).
4. **Model training (PC):** Six architectures trained — Random Forest, SVM, 2D-CNN, 1D-CNN, DS-CNN, plus int8 quantized variants (PTQ and QAT) of each neural network.
5. **Quantization (PC):** Post-training int8 quantization via TFLite and quantization-aware training via `tensorflow-model-optimization`. Evaluated with McNemar's test and prediction agreement analysis.
6. **Deployment (Arduino):** Best int8 model exported as a C byte array, compiled with the Arduino IDE using a custom TFLite Micro build, and flashed to the board. Validated via a 208-clip automated speaker-playback test.

---

## Getting Started

See [GETTING_STARTED.md](GETTING_STARTED.md) for environment setup, reproducing every phase from preprocessing to on-device evaluation, and running the automated playback test.

For a top-down understanding of the work, start with:

- [`docs/project_spec/project_overview.md`](docs/project_spec/project_overview.md) — Goals, scope, hardware constraints, taxonomy
- [`docs/project_spec/dev_log.md`](docs/project_spec/dev_log.md) — Chronological development history (newest entries first)
- [`docs/on_device_evaluation_report.md`](docs/on_device_evaluation_report.md) — Phase 6 synthesis (all evaluation dimensions)
- [`docs/report/`](docs/report/) — Final paper sections (prose) and assembly documents

---

## Repository Structure

```
Car_Sounds/
├── car_diagnostics_dataset/   # Raw WAV files (downloaded from Kaggle; gitignored)
├── data/                      # Preprocessed features, splits, augmented variants
│   ├── splits/                # Train/val/test manifests (source of truth)
│   ├── features/              # Mel-spectrograms and MFCCs for each condition
│   ├── noise_samples/         # 360 real-world noise clips from the Jetta
│   └── noise_bank/            # Grouped noise corpus used for augmentation
├── src/                       # Python source modules
│   ├── preprocessing.py       # Audio loading, normalization, cropping
│   ├── features.py            # Mel-spectrogram and MFCC extraction
│   ├── augmentation.py        # Waveform augmentations
│   ├── models.py              # Model architecture definitions (M2, M5, M6)
│   ├── train_classical.py     # RF and SVM training
│   ├── train_nn.py            # NN training with ablation support
│   ├── quantize_nn.py         # PTQ and QAT conversion + evaluation
│   ├── export_for_arduino.py  # C header generation for deployment
│   ├── validate_features.py   # Feature-parity validation (Python vs. Arduino)
│   ├── playback_test.py       # Automated speaker-playback evaluation
│   ├── collect_noise.py       # PC-side serial receiver for noise collection
│   └── evaluate.py            # Shared metric and plotting utilities
├── arduino/
│   ├── car_sound_classifier/  # Deployment sketch (9 files, ready to flash)
│   └── noise_collector/       # Noise-collection sketch used for the Jetta corpus
├── notebooks/                 # Jupyter notebooks for each development phase (01-07)
├── models/                    # Trained model checkpoints (gitignored)
├── results/                   # All result CSVs, JSONs, and publication figures
│   └── paper_*.png            # Nine publication-quality figures
├── docs/
│   ├── report/                # Final paper sections and assembly documents
│   ├── project_spec/          # Original project specifications and dev log
│   ├── deployment_results.md  # Phase 5 on-device deployment details
│   ├── quantization_results.md           # Phase 4 PTQ/QAT analysis
│   ├── neural_network_models.md          # Phase 3 NN training details
│   ├── classical_ml_baselines.md         # Phase 2 baseline analysis
│   ├── noise_data_collection_and_augmentation.md
│   └── on_device_evaluation_report.md    # Phase 6 synthesis
├── scripts/
│   └── verify_paper_numbers.py           # 73-check accuracy verification script
├── project_setup/             # Phase 0 artifacts (EDA notebook, mic test)
├── GETTING_STARTED.md         # Step-by-step setup and reproduction guide
└── CLAUDE.md                  # AI-assisted development instructions
```

---

## Hardware Target

| Component | Specification |
|-----------|---------------|
| Board | Arduino Nano 33 BLE Sense Rev2 |
| MCU | Nordic nRF52840 (Arm Cortex-M4F @ 64 MHz) |
| Flash | 1 MB (983 KB usable) |
| SRAM | 256 KB |
| Microphone | ST MP34DT06JTR PDM MEMS (on-board) |
| ML framework | TensorFlow Lite for Microcontrollers + CMSIS-NN |
| Quantization | Full-integer int8 (weights and activations) |

---

## Project Status

All eight development phases are complete — data preprocessing, classical ML baselines, neural network training with ablation, quantization, Arduino deployment, on-device evaluation, comprehensive analysis, and final report assembly. The classifier has been flashed to the target hardware, validated against the test set via automated playback, and all findings have been documented for academic submission.
