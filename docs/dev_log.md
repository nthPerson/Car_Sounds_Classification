# Development Log

This log tracks major progress, decisions, and results across the project. Add new entries at the top so the most recent work is always first.

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
