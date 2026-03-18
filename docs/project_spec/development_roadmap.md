# Development Roadmap

**Parent document:** [project_overview.md](project_overview.md)
**Date:** March 10, 2026

---

## Table of Contents

1. [Roadmap Overview](#1-roadmap-overview)
2. [Phase 0 — Project Setup](#2-phase-0--project-setup)
3. [Phase 1 — Data Preprocessing Pipeline](#3-phase-1--data-preprocessing-pipeline)
4. [Phase 2 — Baseline Model Training](#4-phase-2--baseline-model-training)
5. [Phase 3 — Neural Network Training](#5-phase-3--neural-network-training)
6. [Phase 4 — Quantization](#6-phase-4--quantization)
7. [Phase 5 — Arduino Deployment](#7-phase-5--arduino-deployment)
8. [Phase 6 — On-Device Evaluation](#8-phase-6--on-device-evaluation)
9. [Phase 7 — Final Report & Presentation](#9-phase-7--final-report--presentation)
10. [Dependency Graph](#10-dependency-graph)
11. [Risk-Adjusted Plan](#11-risk-adjusted-plan)
12. [Deliverables Summary](#12-deliverables-summary)

---

## 1. Roadmap Overview

The project is divided into 8 phases (0–7). Phases are sequential with some overlap — later phases depend on outputs from earlier phases, but some tasks can be parallelized across team members.

```
Phase 0: Project Setup                          [~2-3 days]
    │
    ▼
Phase 1: Data Preprocessing Pipeline            [~1 week]
    │
    ├────────────────────────┐
    ▼                        ▼
Phase 2: Classical ML     Phase 3: Neural Network Training   [~1-2 weeks]
Baselines                    │
    │                        │
    │                        ▼
    │                   Phase 4: Quantization                [~1 week]
    │                        │
    │                        ▼
    │                   Phase 5: Arduino Deployment          [~1-2 weeks]
    │                        │
    │                        ▼
    │                   Phase 6: On-Device Evaluation        [~1 week]
    │                        │
    ├────────────────────────┘
    ▼
Phase 7: Final Report & Presentation            [~1 week]
```

---

## 2. Phase 0 — Project Setup

**Goal:** Establish the development environment, repository structure, and team workflow.

### Tasks

| # | Task | Description | Output | Owner |
|---|------|-------------|--------|-------|
| 0.1 | Repository structure | Create standard directory layout (see below) | Directory structure | — |
| 0.2 | Python environment | Set up virtual environment with all dependencies (`requirements.txt`) | Working venv | — |
| 0.3 | Arduino IDE setup | Install board support, TFLite library, verify compilation | Blink sketch compiles and uploads | — |
| 0.4 | Mic test sketch | Write a minimal Arduino sketch that captures audio via PDM and prints samples to Serial | Working audio capture | — |
| 0.5 | Dataset verification | Run the EDA notebook, verify file counts match expectations | Verified dataset | — |

### Directory Structure

```
Car_Sounds/
├── car_diagnostics_dataset/    # Raw data (gitignored)
├── data/
│   ├── splits/                 # Train/val/test manifests
│   ├── features/               # Pre-computed features
│   └── normalization_stats.npz
├── docs/                       # Project documentation
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_neural_networks.ipynb
│   ├── 04_quantization.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── preprocessing.py        # Data preprocessing functions
│   ├── features.py             # Feature extraction functions
│   ├── augmentation.py         # Augmentation functions
│   ├── models.py               # Model architecture definitions
│   ├── train.py                # Training script
│   ├── quantize.py             # Quantization and conversion
│   └── evaluate.py             # Evaluation and metrics
├── arduino/
│   ├── mic_test/               # PDM microphone test sketch
│   ├── feature_test/           # On-device feature extraction test
│   └── car_sound_classifier/   # Final deployment sketch
│       ├── car_sound_classifier.ino
│       ├── model_data.h
│       ├── feature_extraction.h
│       ├── feature_extraction.cpp
│       ├── mel_filterbank.h
│       └── config.h
├── models/                     # Saved models (gitignored, except configs)
├── results/                    # Evaluation results, figures, tables
├── explore_dataset.ipynb       # Existing EDA notebook
├── requirements.txt
├── .gitignore
└── README.md
```

### Deliverables
- [ ] Working Python environment with all dependencies
- [ ] Arduino IDE configured and verified with the Nano 33 BLE Sense Rev2
- [ ] Verified that the PDM microphone captures audio on the board
- [ ] Repository directory structure created

---

## 3. Phase 1 — Data Preprocessing Pipeline

**Goal:** Implement the complete data preprocessing pipeline as specified in [data_preprocessing_spec.md](data_preprocessing_spec.md).

**Depends on:** Phase 0

### Tasks

| # | Task | Description | Spec Reference | Output |
|---|------|-------------|---------------|--------|
| 1.1 | Audio loading + resampling | Load all WAVs, resample to 16 kHz, verify | Spec §3 | Resampled audio arrays |
| 1.2 | Amplitude normalization | Implement peak normalization | Spec §4 | Normalized audio arrays |
| 1.3 | Duration standardization | Pad/center-crop to 1.5 s (24,000 samples) | Spec §5 | Fixed-length arrays |
| 1.4 | Train/val/test split | Stratified split (70/15/15), save manifests | Spec §6 | `splits/*.csv` |
| 1.5 | Mel-spectrogram extraction | Compute log-mel spectrograms (40×92) for all files | Spec §7.1 | `features/mel_spectrograms/*.npz` |
| 1.6 | MFCC extraction | Compute MFCCs (92×13) for all files | Spec §7.2 | `features/mfcc/*.npz` |
| 1.7 | Classical ML feature vectors | Extract 441-dim feature vectors | Spec §7.3 | `classical_ml_features/*.npz` |
| 1.8 | Data augmentation pipeline | Implement all waveform + spectrogram augmentations | Spec §8 | Augmentation functions |
| 1.9 | Feature normalization | Compute train-set mean/std, apply to all sets | Spec §9 | `normalization_stats.npz` |
| 1.10 | Label encoding | Create label mappings for all 3 tiers | Spec §10 | Label arrays in feature files |
| 1.11 | Config file | Save all preprocessing parameters | Spec §12.1 | `preprocessing_config.json` |
| 1.12 | Verification notebook | Notebook that visualizes sample spectrograms, verifies shapes, checks split balance | — | `notebooks/01_preprocessing.ipynb` |

### Deliverables
- [ ] Complete preprocessed dataset (mel-spectrograms, MFCCs, classical features) saved to disk
- [ ] Train/val/test split manifests
- [ ] Normalization statistics
- [ ] Preprocessing configuration file
- [ ] Verification notebook showing sample outputs and split statistics

### Acceptance Criteria
- All 1,386 files successfully loaded and resampled to 16 kHz
- All mel-spectrograms have shape (40, 92), all MFCCs have shape (92, 13)
- Train/val/test split ratios are correct (within 1% of target due to rounding)
- Every condition is represented in all three sets
- Normalization stats are computed from training set only

---

## 4. Phase 2 — Baseline Model Training

**Goal:** Train classical ML baselines (Random Forest, SVM) as specified in [model_architecture_spec.md](model_architecture_spec.md) §2.

**Depends on:** Phase 1 (needs feature vectors and split manifests)

### Tasks

| # | Task | Description | Output |
|---|------|-------------|--------|
| 2.1 | Random Forest — Tier 1 | Train + hyperparameter search (5-fold CV) | Best RF model (Tier 1) |
| 2.2 | Random Forest — Tier 2 | Train + hyperparameter search | Best RF model (Tier 2) |
| 2.3 | Random Forest — Tier 3 | Train + hyperparameter search | Best RF model (Tier 3) |
| 2.4 | SVM — Tier 1 | Train + hyperparameter search | Best SVM model (Tier 1) |
| 2.5 | SVM — Tier 2 | Train + hyperparameter search | Best SVM model (Tier 2) |
| 2.6 | SVM — Tier 3 | Train + hyperparameter search | Best SVM model (Tier 3) |
| 2.7 | Feature importance analysis | Extract and visualize top features from RF | Feature importance charts |
| 2.8 | Evaluate on test set | Compute accuracy, F1, confusion matrices for all models | Evaluation results |

**Parallelism:** Tasks 2.1–2.6 can be run in parallel. Phase 2 can overlap with Phase 3 (different team members can work on classical ML and neural networks simultaneously).

### Deliverables
- [ ] 6 trained classical ML models (3 RF + 3 SVM)
- [ ] Hyperparameter search results log
- [ ] Feature importance analysis
- [ ] Test set evaluation metrics for all models
- [ ] `notebooks/02_classical_ml.ipynb`

---

## 5. Phase 3 — Neural Network Training

**Goal:** Train all neural network architectures as specified in [model_architecture_spec.md](model_architecture_spec.md) §3–5.

**Depends on:** Phase 1 (needs preprocessed features)

### Tasks

| # | Task | Description | Output |
|---|------|-------------|--------|
| 3.1 | 2-D CNN — Tier 1 | Train compact 2-D CNN (float32) | `m2_cnn2d_float32/tier1_best.h5` |
| 3.2 | 2-D CNN — Tier 2 | Train compact 2-D CNN (float32) | `m2_cnn2d_float32/tier2_best.h5` |
| 3.3 | 2-D CNN — Tier 3 | Train compact 2-D CNN (float32) | `m2_cnn2d_float32/tier3_best.h5` |
| 3.4 | 1-D CNN — Tier 1 | Train 1-D CNN on MFCCs (float32) | `m5_cnn1d/tier1_best.h5` |
| 3.5 | 1-D CNN — Tier 2 | Train 1-D CNN on MFCCs (float32) | `m5_cnn1d/tier2_best.h5` |
| 3.6 | 1-D CNN — Tier 3 | Train 1-D CNN on MFCCs (float32) | `m5_cnn1d/tier3_best.h5` |
| 3.7 | DS-CNN — Tier 1 | Train DS-CNN (32ch, float32) | `m6_dscnn/tier1_best.h5` |
| 3.8 | DS-CNN — Tier 2 | Train DS-CNN (32ch, float32) | `m6_dscnn/tier2_best.h5` |
| 3.9 | DS-CNN — Tier 3 | Train DS-CNN (32ch, float32) | `m6_dscnn/tier3_best.h5` |
| 3.10 | Hyperparameter tuning | Tune learning rate, dropout, filters on Tier 2 models | Updated configs |
| 3.11 | Training curves | Plot and save loss/accuracy curves for all models | Training curve figures |
| 3.12 | Evaluate on test set | Compute metrics for all float32 models | Evaluation results |

### Deliverables
- [ ] 9 trained float32 neural network models (3 architectures × 3 tiers)
- [ ] Training curves for all models
- [ ] Test set evaluation metrics for all float32 models
- [ ] `notebooks/03_neural_networks.ipynb`
- [ ] Markdown docs/ file that documents models, hyperparameters, evaluation methods, and other details necessary for future research paper report. See docs/classical_ml_baselines.md for example structure.

---

## 6. Phase 4 — Quantization

**Goal:** Quantize all neural networks to int8 using both PTQ and QAT as specified in [quantization_deployment_spec.md](quantization_deployment_spec.md) §2–4.

**Depends on:** Phase 3 (needs trained float32 models)

### Tasks

| # | Task | Description | Output |
|---|------|-------------|--------|
| 4.1 | Representative dataset | Create calibration dataset (200 samples, stratified) | Calibration data generator |
| 4.2 | PTQ — 2-D CNN (all tiers) | Post-training int8 quantization of M2 | M3 `.tflite` files |
| 4.3 | QAT — 2-D CNN (all tiers) | Quantization-aware training fine-tuning of M2 → convert | M4 `.tflite` files |
| 4.4 | QAT — 1-D CNN (all tiers) | QAT fine-tuning + convert | M5 `.tflite` files |
| 4.5 | QAT — DS-CNN (all tiers) | QAT fine-tuning + convert | M6 `.tflite` files |
| 4.6 | PTQ — 1-D CNN (all tiers) | For comparison, also do PTQ on 1-D CNN | Additional `.tflite` files |
| 4.7 | PTQ — DS-CNN (all tiers) | For comparison, also do PTQ on DS-CNN | Additional `.tflite` files |
| 4.8 | Model size recording | Record file sizes for all float32 and int8 models | Size comparison table |
| 4.9 | TFLite evaluation | Evaluate all int8 models on test set using TFLite interpreter | Int8 accuracy metrics |
| 4.10 | Quantization impact analysis | Compute ΔAccuracy, ΔF1, agreement rates, confidence shifts | Quantization impact tables |
| 4.11 | Select deployment model | Choose the best int8 model for Arduino deployment (Tier 2) | Deployment candidate |

### Deliverables
- [ ] All quantized `.tflite` model files (PTQ and QAT variants)
- [ ] TFLite test set evaluation metrics
- [ ] Quantization impact analysis (tables and figures)
- [ ] Selected deployment model with justification
- [ ] `notebooks/04_quantization.ipynb`

---

## 7. Phase 5 — Arduino Deployment

**Goal:** Deploy the selected int8 model to the Arduino Nano 33 BLE Sense Rev2 as specified in [quantization_deployment_spec.md](quantization_deployment_spec.md) §5–12.

**Depends on:** Phase 4 (needs int8 `.tflite` model), Phase 0.3/0.4 (Arduino IDE + mic test)

### Tasks

| # | Task | Description | Output |
|---|------|-------------|--------|
| 5.1 | Model to C array | Convert best `.tflite` model to `model_data.h` using `xxd` | `model_data.h` |
| 5.2 | Mel filterbank export | Generate mel filterbank C header from Python | `mel_filterbank.h` |
| 5.3 | Feature extraction code | Implement on-device mel-spectrogram computation (FFT + mel filterbank + log) | `feature_extraction.h/.cpp` |
| 5.4 | Feature extraction test | Validate on-device features against Python features for 10+ audio clips | Feature parity report |
| 5.5 | TFLite Micro integration | Set up TFLite Micro interpreter, op resolver, tensor arena | Working inference on test input |
| 5.6 | Main inference loop | Integrate audio capture → features → inference → output | `car_sound_classifier.ino` |
| 5.7 | Serial output | Format and print classification results | Serial output working |
| 5.8 | Memory optimization | Profile and optimize SRAM usage if needed | Fits in 256 KB |
| 5.9 | End-to-end test | Run the full pipeline on the Arduino with audio input | Working end-to-end system |
| 5.10 | Deploy additional models | Repeat 5.1, 5.5–5.9 for other int8 models (for comparison) | Multiple deployable models |

### Critical Path Warning

Task 5.3 (on-device feature extraction) is the most technically challenging task in the entire project. It requires:
- Implementing FFT using CMSIS-DSP on the Cortex-M4
- Applying the mel filterbank in float32
- Matching the output closely enough to the Python pipeline

**Mitigation:** If custom feature extraction proves too difficult, use the Edge Impulse spectral features block as a reference implementation, or use a simpler feature representation (e.g., raw FFT magnitude bins without mel warping).

### Deliverables
- [ ] Complete Arduino sketch (`car_sound_classifier/`)
- [ ] Feature parity validation report
- [ ] Working end-to-end inference on the Arduino board
- [ ] Flash and SRAM usage report for each deployed model

---

## 8. Phase 6 — On-Device Evaluation

**Goal:** Benchmark and evaluate the deployed model(s) on the Arduino as specified in [evaluation_plan.md](evaluation_plan.md) §5–6.

**Depends on:** Phase 5 (needs deployed model on Arduino)

### Tasks

| # | Task | Description | Output |
|---|------|-------------|--------|
| 6.1 | Inference latency measurement | Run 100 inferences, record timing | Latency statistics |
| 6.2 | Feature extraction latency | Measure mel-spectrogram computation time | Feature timing |
| 6.3 | Tensor arena profiling | Determine minimum arena size for each model | Arena size report |
| 6.4 | Flash/SRAM recording | Record compilation reports for all deployed models | Memory usage table |
| 6.5 | Playback test setup | Prepare speaker, calibrate volume, set up test protocol | Test environment ready |
| 6.6 | Playback test execution | Run test set audio through speaker → Arduino classification | Playback predictions |
| 6.7 | Playback test evaluation | Compute accuracy, F1, confusion matrix from playback | Playback evaluation report |
| 6.8 | Compare PC vs. playback | Analyze accuracy gap between PC evaluation and live playback | Gap analysis |
| 6.9 | Robustness tests (stretch) | Test under noise, varying distance/volume | Robustness results |

### Deliverables
- [ ] On-device benchmarking results (latency, memory)
- [ ] Playback test results with comparison to PC evaluation
- [ ] Complete on-device evaluation report
- [ ] `notebooks/05_evaluation.ipynb` (consolidating all results)

---

## 9. Phase 7 — Final Report & Presentation

**Goal:** Compile all results into a comprehensive report and presentation.

**Depends on:** All previous phases

### Tasks

| # | Task | Description | Output |
|---|------|-------------|--------|
| 7.1 | Results compilation | Fill in all reporting templates from [evaluation_plan.md](evaluation_plan.md) | Complete results tables |
| 7.2 | Figure generation | Create all visualizations (confusion matrices, training curves, scatter plots) | Figures directory |
| 7.3 | Analysis writing | Write analysis and interpretation of results | Analysis sections |
| 7.4 | Limitations and future work | Document known limitations, lessons learned, future directions | Discussion section |
| 7.5 | Report assembly | Compile final report document | Final report |
| 7.6 | Presentation slides | Create presentation summarizing key findings | Slide deck |
| 7.7 | Code cleanup | Clean up notebooks and scripts for reproducibility | Clean codebase |
| 7.8 | README update | Update README with project description, setup instructions, usage | Updated README.md |

### Deliverables
- [ ] Final report (PDF)
- [ ] Presentation slides
- [ ] Clean, documented codebase
- [ ] Updated README with project description and instructions

---

## 10. Dependency Graph

```
Phase 0 ──────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
Phase 1 (Preprocessing)                  Phase 0.3-0.4 (Arduino Setup)
    │                                          │
    ├──────────────┐                           │
    ▼              ▼                           │
Phase 2         Phase 3 (NN Training)          │
(Classical ML)     │                           │
    │              ▼                           │
    │          Phase 4 (Quantization)          │
    │              │                           │
    │              ├───────────────────────────┘
    │              ▼
    │          Phase 5 (Arduino Deployment)
    │              │
    │              ▼
    │          Phase 6 (On-Device Evaluation)
    │              │
    ├──────────────┘
    ▼
Phase 7 (Report)
```

**Critical path:** 0 → 1 → 3 → 4 → 5 → 6 → 7

Phase 2 (classical ML) is OFF the critical path and can be done in parallel with Phases 3–5.

---

## 11. Risk-Adjusted Plan

### 11.1 Minimum Viable Deliverable

If time runs short, the minimum viable project is:

1. **Preprocessing pipeline** (Phase 1) — must be complete
2. **One classical ML baseline** (Random Forest, Tier 2) — provides a baseline
3. **One neural network** (2-D CNN, Tier 2, float32) — demonstrates the ML pipeline
4. **One quantized variant** (2-D CNN, Tier 2, int8 PTQ) — demonstrates quantization
5. **Arduino deployment** (the PTQ model) — demonstrates on-device inference
6. **Basic evaluation** (PC accuracy + on-device latency/memory) — demonstrates the comparison

This delivers on all primary goals (G1–G5) with reduced scope on the model matrix.

### 11.2 What to Cut First

If schedule pressure requires reducing scope, cut in this order (least impactful first):

1. **DS-CNN (M6)** — most complex architecture, similar results expected to 2-D CNN
2. **Tier 3 (9-class)** — most challenging tier, likely lowest accuracy, optional
3. **Playback tests** — nice-to-have but not essential for the quantization comparison
4. **SVM baselines** — Random Forest alone is a sufficient baseline
5. **Tier 1 (2-class)** — simplest tier, results are predictable (high accuracy)
6. **QAT** — PTQ alone demonstrates quantization; QAT is a refinement

### 11.3 Biggest Technical Risks

| Risk | Phase | Impact | Mitigation |
|------|-------|--------|-----------|
| On-device feature extraction doesn't match training features | Phase 5 | High — model accuracy drops significantly on-device | Pre-compute feature parity test early; use Edge Impulse as fallback |
| Tensor arena doesn't fit in SRAM | Phase 5 | High — model can't run | Start with M5 (1-D CNN, smallest arena); reduce mel bands if needed |
| Poor accuracy on small dataset | Phase 3 | Medium — models don't learn well | Heavy augmentation; consider transfer learning; start with Tier 2 |
| TFLite library version issues | Phase 5 | Medium — compilation errors | Pin library versions; use tflite-micro built from source |

---

## 12. Deliverables Summary

### Code Deliverables

| Deliverable | Location | Phase |
|------------|----------|-------|
| Preprocessing pipeline | `src/preprocessing.py`, `src/features.py`, `src/augmentation.py` | 1 |
| Model definitions | `src/models.py` | 3 |
| Training script | `src/train.py` | 3 |
| Quantization script | `src/quantize.py` | 4 |
| Evaluation script | `src/evaluate.py` | 2, 3, 4, 6 |
| Arduino mic test | `arduino/mic_test/` | 0 |
| Arduino feature test | `arduino/feature_test/` | 5 |
| Arduino classifier | `arduino/car_sound_classifier/` | 5 |

### Data Deliverables

| Deliverable | Location | Phase |
|------------|----------|-------|
| Split manifests | `data/splits/` | 1 |
| Preprocessed features | `data/features/` | 1 |
| Normalization stats | `data/normalization_stats.npz` | 1 |
| Preprocessing config | `data/preprocessing_config.json` | 1 |
| Trained models (float32) | `models/m2_*/`, `models/m5_*/`, `models/m6_*/` | 3 |
| Quantized models (int8) | `models/m3_*/`, `models/m4_*/`, `models/m5_*/`, `models/m6_*/` | 4 |
| Classical ML models | `models/m1_classical/` | 2 |

### Analysis Deliverables

| Deliverable | Location | Phase |
|------------|----------|-------|
| Preprocessing verification | `notebooks/01_preprocessing.ipynb` | 1 |
| Classical ML analysis | `notebooks/02_classical_ml.ipynb` | 2 |
| Neural network training analysis | `notebooks/03_neural_networks.ipynb` | 3 |
| Quantization analysis | `notebooks/04_quantization.ipynb` | 4 |
| Final evaluation compilation | `notebooks/05_evaluation.ipynb` | 6 |
| All figures and tables | `results/` | 7 |

### Documentation Deliverables

| Deliverable | Location | Phase |
|------------|----------|-------|
| Project overview | `docs/project_overview.md` | 0 |
| Data preprocessing spec | `docs/data_preprocessing_spec.md` | 0 |
| Model architecture spec | `docs/model_architecture_spec.md` | 0 |
| Quantization & deployment spec | `docs/quantization_deployment_spec.md` | 0 |
| Evaluation plan | `docs/evaluation_plan.md` | 0 |
| Development roadmap | `docs/development_roadmap.md` | 0 |
| Feasibility report (existing) | `docs/deployment_feasibility_report.md` | — |
| Final report | `report/final_report.pdf` | 7 |
| README | `README.md` | 7 |
