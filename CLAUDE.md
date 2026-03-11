# CLAUDE.md — Project Instructions for Claude

## Project Summary

This is a BDA 602 group project: **Real-Time Car Sound Classification with Quantized Neural Networks on the Arduino Nano 33 BLE Sense Rev2**. We train audio classifiers on the Car Diagnostics Dataset (1,386 WAV files, 13 conditions across 3 operational states), quantize them to int8, and deploy on a constrained microcontroller (1 MB flash, 256 KB SRAM, Cortex-M4F @ 64 MHz).

## Repository Layout

```
Car_Sounds/
├── car_diagnostics_dataset/   # Raw audio (gitignored, download from Kaggle)
├── data/
│   ├── splits/                # Train/val/test manifests (source of truth)
│   ├── features/              # Pre-computed mel-spectrograms, MFCCs (gitignored)
│   └── classical_ml_features/ # 441-dim feature vectors (gitignored)
├── docs/                      # All project documentation and specs
├── project_setup/             # Phase 0 artifacts (EDA notebook, mic test, dataset verification)
├── src/                       # Python source modules
├── notebooks/                 # Phase-specific Jupyter notebooks
├── arduino/                   # Arduino sketches for deployment
│   └── car_sound_classifier/  # Final deployment sketch
├── models/                    # Trained model artifacts (gitignored)
│   ├── m1_classical/          # RF/SVM baselines
│   ├── m2_cnn2d_float32/      # 2-D CNN float32 baseline
│   ├── m3_cnn2d_int8_ptq/     # 2-D CNN post-training quantized
│   ├── m4_cnn2d_int8_qat/     # 2-D CNN quantization-aware trained
│   ├── m5_cnn1d_int8_qat/     # 1-D CNN on MFCCs
│   └── m6_dscnn_int8_qat/     # Depthwise separable CNN
├── results/                   # Evaluation figures, tables, reports
├── requirements.txt
├── GETTING_STARTED.md         # Team guide: how to set up and run the project
└── CLAUDE.md                  # This file
```

## Key Technical Details

- **Target hardware:** Arduino Nano 33 BLE Sense Rev2 (non-negotiable)
- **Dataset:** [Car Diagnostics Dataset](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset)
- **3 taxonomy tiers:** Tier 1 (2-class binary), Tier 2 (6-class state-aware — primary deployment target), Tier 3 (9-class, excludes combined faults)
- **Mel-spectrogram params:** n_fft=512, hop=256, n_mels=40, fmin=20, fmax=8000 → shape (40, 92, 1)
- **MFCC params:** 13 coefficients, same FFT/hop/mel settings → shape (92, 13)
- **Audio:** All resampled to 16 kHz mono, duration standardized to 1.5 s (24,000 samples)
- **Train/val/test split:** 70/15/15, stratified by condition, random_state=42. Manifests in `data/splits/`
- **Model SRAM budget:** ≤ 80 KB tensor arena. **Model flash budget:** ≤ 150 KB
- **Quantization:** Full-integer int8 (both PTQ and QAT compared against float32 baseline)

## Documentation

All specifications live in `docs/`. Read these before making design decisions:

| Document | What it covers |
|----------|---------------|
| `project_overview.md` | Goals, scope, constraints, memory budgets, class taxonomy tiers |
| `data_preprocessing_spec.md` | Preprocessing pipeline, feature extraction, augmentation, splitting |
| `model_architecture_spec.md` | Model architectures (M1–M6), training config, hyperparameters |
| `quantization_deployment_spec.md` | PTQ/QAT, TFLite conversion, Arduino sketch, on-device inference |
| `evaluation_plan.md` | ML metrics, quantization impact analysis, on-device benchmarking |
| `development_roadmap.md` | 8 phases, task breakdown, dependencies, risk-adjusted plan |
| `deployment_feasibility_report.md` | Original feasibility study (hardware specs, risk assessment) |

## Commands

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run dataset verification and regenerate splits
python project_setup/verify_dataset_and_split.py

# Install dependencies
pip install -r requirements.txt
```

## Code Conventions

- Python source modules go in `src/`, notebooks in `notebooks/`
- Use random_state=42 for all randomized operations (splits, training, augmentation seeds)
- Split manifests in `data/splits/` are the single source of truth — never regenerate them unless intentionally changing the split
- Preprocessing config is recorded in `data/preprocessing_config.json` when generated
- Feature normalization statistics (mean/std) are computed from the training set ONLY
- Use `sparse_categorical_crossentropy` for all classification tiers (including binary) for code uniformity

## Getting Started Guide

`GETTING_STARTED.md` in the project root is the team-facing guide for setting up and running the project. **When completing a development phase that adds new runnable steps** (e.g., a training script, an evaluation command, a deployment procedure), **update `GETTING_STARTED.md`** so teammates always have accurate, up-to-date instructions. Keep the language simple and non-technical where possible.

## Development Log

**After making significant changes, update `docs/dev_log.md`.** Not every small edit needs a log entry — use judgment. Log entries should be added for:

- Completing a roadmap phase or major task
- Key experiment results (model accuracy, quantization impact, on-device measurements)
- Design decisions that deviate from the specs
- Bugs or issues that affect the team
- Changes to the data splits, preprocessing pipeline, or model architectures

This ensures team members can quickly catch up on what changed and why without reading every commit.
