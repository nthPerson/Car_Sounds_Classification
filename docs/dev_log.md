# Development Log

This log tracks major progress, decisions, and results across the project. Add new entries at the top so the most recent work is always first.

---

## 2026-03-17 — Phase 2 Complete: Classical ML Baselines

**What was done:**
- Implemented `src/evaluate.py` — reusable evaluation utilities (metrics, confusion matrices, ROC curves, reporting). Shared by Phase 2 and all later phases.
- Implemented `src/train_classical.py` — trains Random Forest (RandomizedSearchCV, n_iter=100) and SVM (GridSearchCV, Pipeline with StandardScaler) on all 3 taxonomy tiers with 5-fold stratified CV, f1_macro scoring.
- Created `notebooks/02_classical_ml.ipynb` — analysis notebook with comparison tables, confusion matrices, feature importance, ROC curves.
- Feature importance analysis: top-50 features identified from Tier 2 RF, retrained reduced-feature model.

**Test set results:**

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| RF Tier 1 | 0.8990 | 0.8724 | 0.8965 |
| SVM Tier 1 | 0.9327 | 0.9163 | 0.9316 |
| RF Tier 2 | 0.8413 | 0.6794 | 0.8272 |
| SVM Tier 2 | 0.8510 | 0.6992 | 0.8422 |
| RF Tier 3 | 0.7622 | 0.7019 | 0.7522 |
| SVM Tier 3 | 0.7552 | 0.7106 | 0.7494 |
| RF Tier 2 (top-50) | 0.8125 | 0.6724 | 0.8124 |

**Key observations:**
- SVM outperforms RF on Tiers 1–2; RF slightly better on Tier 3
- Tier 2 macro F1 is lower than accuracy suggests (~0.70 vs ~0.85) due to poor performance on minority classes — "Normal Start-Up" (9 test samples) is the hardest class for both models
- Top features are MFCC-related: delta2_mfcc_0_std, mfcc_0_std, delta_mfcc_0_min, followed by spectral centroid and RMS energy
- Reduced-feature RF (top-50) performs slightly worse, suggesting most of the 441 features contribute useful signal

**Design decisions:**
- Used RandomizedSearchCV (n_iter=100) for RF instead of exhaustive grid (648 combos) — practical with negligible accuracy loss
- SVM wrapped in Pipeline(StandardScaler, SVC) so scaler is part of the saved model artifact
- Merged train+val sets for CV-based training (1,178 samples for Tier 1/2, 806 for Tier 3) since classical ML uses CV for validation rather than a separate val set
- SVC(probability=True) enabled for ROC curves and future comparison analysis

**Output artifacts:**
- 6 trained models + 1 reduced-feature variant in `models/m1_classical/`
- Confusion matrices, ROC curves, feature importance chart in `results/`
- Search results and evaluation metrics as JSON in `models/m1_classical/`

**Status:** Phase 2 complete. Classical ML baselines established. Ready for Phase 3 (neural network training).

---

## 2026-03-11 — Phase 1 Complete: Data Preprocessing Pipeline

**What was done:**
- Implemented `src/preprocessing.py` — audio loading, resampling to 16 kHz, peak normalization, center-crop/pad to 1.5s (24,000 samples)
- Implemented `src/features.py` — log-mel spectrogram (40x92), MFCC (92x13), 441-dim classical ML feature extraction, per-channel normalization stats
- Implemented `src/augmentation.py` — 5 waveform augmentations (time shift, noise injection, pitch shift, speed perturbation, random gain) + SpecAugment (frequency/time masking), designed for on-the-fly use during training
- Implemented `src/run_preprocessing.py` — orchestrates the full pipeline, generates all feature files and config
- Created `notebooks/01_preprocessing.ipynb` — verification notebook with shape checks, visualizations, leakage checks

**Implementation detail — `center=False`:**
- All librosa STFT/mel/MFCC calls use `center=False` to produce the spec's expected 92 frames (not 94 from default `center=True`). This also matches on-device behavior where there is no padding.

**Output artifacts generated:**
- `data/features/mel_spectrograms/{train,val,test}.npz` — un-normalized log-mel spectrograms
- `data/features/mfcc/{train,val,test}.npz` — un-normalized MFCCs
- `data/classical_ml_features/{train,val,test}.npz` — 441-dim feature vectors
- `data/normalization_stats.npz` — per-mel-band and per-MFCC-coefficient mean/std from training set
- `data/preprocessing_config.json` — all parameters for reproducibility

**Feature shapes verified:**
| Feature | Train | Val | Test |
|---------|-------|-----|------|
| Mel-spectrogram | (970, 40, 92) | (208, 40, 92) | (208, 40, 92) |
| MFCC | (970, 92, 13) | (208, 92, 13) | (208, 92, 13) |
| Classical ML | (970, 441) | (208, 441) | (208, 441) |

**Design decisions:**
- Features stored un-normalized; normalization applied at training time after augmentation
- Tier 3 excluded samples (combined faults, label=-1) kept in arrays; filtered at training time
- Augmentation functions accept `np.random.Generator` for reproducibility
- Classical features: NaN from degenerate distributions replaced with 0.0

**Status:** Phase 1 complete. All preprocessing artifacts generated and verified. Ready for Phase 2 (classical ML baselines).

---

## 2026-03-11 — Phase 0 Complete: Project Setup

**What was done:**
- Created full repository directory structure (`src/`, `notebooks/`, `arduino/`, `data/`, `models/`, `results/`, `project_setup/`)
- Created `requirements.txt` with all Python dependencies (librosa, tensorflow, tensorflow-model-optimization, scikit-learn, etc.)
- Updated `.gitignore` to exclude large artifacts (datasets, models, pre-computed features) while tracking split manifests and code
- Wrote PDM microphone test sketch (`project_setup/mic_test/mic_test.ino`) — captures 1.5s at 16 kHz, prints audio statistics
- Wrote dataset verification + split generation script (`project_setup/verify_dataset_and_split.py`)
- Ran verification: all 1,386 files present, all 13 condition counts match, all mono, sample rates 24k/44.1k/48k as expected
- Generated stratified train/val/test splits (970/208/208) with label mappings for all 3 taxonomy tiers, saved to `data/splits/`
- Moved EDA notebook to `project_setup/explore_dataset.ipynb`

**Mic test results (Arduino Nano 33 BLE Sense Rev2):**
- PDM microphone initializes and captures at 16 kHz successfully
- Raw samples show a positive DC offset (~800), which is normal for the nRF52840 PDM config and won't affect mel-spectrogram features (DC energy is below our 20 Hz fmin cutoff)
- Microphone responds to sound with reasonable dynamic range

**Split summary (Tier 2 — primary target):**

| Class | Train | Val | Test |
|-------|-------|-----|------|
| Normal Braking | 54 | 11 | 12 |
| Braking Fault | 53 | 12 | 11 |
| Normal Idle | 185 | 39 | 40 |
| Idle Fault | 552 | 119 | 118 |
| Normal Start-Up | 43 | 9 | 9 |
| Start-Up Fault | 83 | 18 | 18 |

**Status:** Phase 0 complete. Ready to begin Phase 1 (data preprocessing pipeline).

---

## 2026-03-10 — Project Documentation Created

**What was done:**
- Wrote 6 specification documents in `docs/`:
  - `project_overview.md` — problem statement, goals, scope, hardware constraints, memory budgets, class taxonomy tiers
  - `data_preprocessing_spec.md` — full preprocessing pipeline: resampling, normalization, feature extraction (mel-spectrogram 40×92, MFCC 92×13, 441-dim classical ML vectors), augmentation (7 techniques), splitting strategy
  - `model_architecture_spec.md` — 6 model variants (M1–M6) with layer-by-layer parameter counts, SRAM feasibility analysis, training config, hyperparameter strategy
  - `quantization_deployment_spec.md` — PTQ/QAT methods, TFLite conversion, Arduino sketch architecture, on-device feature extraction via CMSIS-DSP, memory layout
  - `evaluation_plan.md` — 4 evaluation dimensions, reporting templates, statistical tests
  - `development_roadmap.md` — 8 phases with task breakdowns, dependency graph, risk-adjusted plan

**Key design decisions documented:**
- Tier 2 (6-class state-aware) is the primary deployment target
- Combined-fault classes excluded from Tier 3 (folded into Tier 1/2 aggregates)
- DS-CNN reduced to 32 channels (64-channel variant exceeds SRAM budget)
- Audio standardized to 1.5s (center-crop startup clips from 1.7s)
- On-device feature extraction in float32 using CMSIS-DSP, then quantize input to int8

**Status:** Documentation complete. Provides the technical basis for all development work.
