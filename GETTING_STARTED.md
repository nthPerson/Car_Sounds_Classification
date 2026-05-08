# Getting Started

Welcome! This guide walks you through setting up and running the Car Sound Classification project. Follow each section in order.

## Important: Branch Policy

**Do not run scripts that generate files (preprocessing, training, etc.) or create new files while on the `main` branch.** Always create a Git branch for your work first. This prevents merge conflicts and keeps `main` clean.

```bash
# Create a branch before doing any work
git checkout -b your-branch-name
# ... or just use VScode's Source Control UI

# When your work is ready, push and open a pull request to merge into main
git push -u origin your-branch-name
# ... or just use VScode's Source Control UI
```

If you are unfamiliar with Git branching, ask a teammate for help before running any scripts.

## Project Documentation

All of the project's technical specifications, architecture decisions, and planning documents live in the `docs/` folder. **Read these to understand the project goals, what has been done, and what still needs to be built:**

| Document | What you'll learn |
|----------|-------------------|
| `docs/project_overview.md` | Project goals, scope, hardware constraints, and class taxonomy |
| `docs/data_preprocessing_spec.md` | How audio data is processed and what features are extracted |
| `docs/model_architecture_spec.md` | The six model variants we are building and how they differ |
| `docs/quantization_deployment_spec.md` | How models get converted for the Arduino and deployed on-device |
| `docs/evaluation_plan.md` | How we measure success (metrics, benchmarks, tests) |
| `docs/development_roadmap.md` | The full 8-phase plan with task breakdowns and dependencies |
| `docs/dev_log.md` | Running log of completed work, results, and decisions (start here to catch up) |

## Prerequisites

- **Python 3.10+** installed on your machine
- **Git** for cloning the repository
- A Kaggle account (to download the dataset)

## 1. Clone the Repository

```bash
git clone <repo-url>
cd Car_Sounds
```

## 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Download the Dataset

1. Go to the [Car Diagnostics Dataset on Kaggle](https://www.kaggle.com/datasets/malakragaie/car-diagnostics-dataset).
2. Download and unzip it so that the audio files are inside `car_diagnostics_dataset/` at the project root. Replace the spaces with underscores in the name of the directory.

Your folder should look like:

```
Car_Sounds/
├── car_diagnostics_dataset/   # <-- WAV files go here
├── data/
├── src/
└── ...
```

## 5. Verify the Dataset and Generate Splits

This script checks that all 1,386 audio files are present and generates the train/val/test split manifests (saved to `data/splits/`).

```bash
python project_setup/verify_dataset_and_split.py
```

> **Note:** The split manifests are already committed to the repo. Only re-run this if you intentionally want to regenerate them, and do not run it on the main branch.

## 6. Run the Preprocessing Pipeline (Phase 1)

This extracts mel-spectrograms, MFCCs, and classical ML feature vectors from all audio files.

```bash
python src/run_preprocessing.py
```

**What it produces:**

| Output | Location |
|--------|----------|
| Mel-spectrograms | `data/features/mel_spectrograms/{train,val,test}.npz` |
| MFCCs | `data/features/mfcc/{train,val,test}.npz` |
| Classical ML features | `data/classical_ml_features/{train,val,test}.npz` |
| Normalization stats | `data/normalization_stats.npz` |
| Config record | `data/preprocessing_config.json` |

## 7. Train Classical ML Baselines (Phase 2)

This trains Random Forest and SVM models on all 3 taxonomy tiers using the pre-computed 441-dim feature vectors.

```bash
python src/train_classical.py
```

**What it produces:**

| Output | Location |
|--------|----------|
| Trained models (6 total) | `models/m1_classical/{rf,svm}_tier{1,2,3}.joblib` |
| RF retrained on top-50 features | `models/m1_classical/rf_tier2_top50.joblib` |
| Hyperparameter search results | `models/m1_classical/search_results.json` |
| Evaluation metrics | `models/m1_classical/evaluation_results.json` |
| Confusion matrices | `results/cm_*.png` |
| Feature importance chart | `results/feature_importance_tier2.png` |
| ROC curves (Tier 1) | `results/roc_*.png` |
| Summary table | `results/classical_ml_summary.csv` |

To explore the results interactively, open `notebooks/02_classical_ml.ipynb`.

## 8. Generate Augmented Training Data

This expands the training set from 970 → 3,317 samples using waveform augmentation (time shift, noise, pitch shift, speed perturbation, random gain) with class-aware balancing to mitigate the 12.8x class imbalance.

```bash
python src/generate_augmented_data.py
```

Runtime: ~20 seconds. Run this once before training neural networks.

## 9. Train Neural Network Models (Phase 3)

This trains three neural network architectures (2-D CNN, 1-D CNN, DS-CNN) on all 3 taxonomy tiers. By default, trains on noise-augmented data and reuses existing hyperparameter search results.

```bash
python src/train_nn.py
```

Requires TensorFlow 2.18 with GPU support. Runtime: ~6 minutes on GPU (no HP search), ~22 minutes with HP search.

**Training data variant:** Edit `AUGMENTATION_VARIANT` at the top of `src/train_nn.py`:
- `"combined"` — All augmented data together (5,664 samples, default)
- `"noise"` — Real-world noise augmentation only (3,317 samples)
- `"standard"` — Synthetic waveform augmentation only (3,317 samples)
- `"none"` — Original data, no augmentation (970 samples)

Set `SKIP_HP_SEARCH = False` to re-run the hyperparameter search (adds ~15 min).

**What it produces:**

| Output | Location |
|--------|----------|
| 9 trained float32 models | `models/m2_cnn2d_float32/`, `models/m5_cnn1d_int8_qat/`, `models/m6_dscnn_int8_qat/` |
| Hyperparameter search results | `models/hyperparameter_search.json` |
| Training curves | `results/training_curves_m{2,5,6}_tier{1,2,3}.png` |
| Confusion matrices | `results/cm_m{2,5,6}_tier_{1,2,3}[_norm].png` |
| ROC curves (Tier 1) | `results/roc_m{2,5,6}_tier_1.png` |
| Summary table | `results/nn_summary.csv` |

To explore the results interactively, open `notebooks/03_neural_networks.ipynb`.

## 10. Collect Background Noise Samples (Optional)

Collect real-world background noise using the Arduino's microphone for more realistic data augmentation. Requires the Arduino Nano 33 BLE Sense Rev2 connected via USB.

**Step 1:** Upload the sketch `arduino/noise_collector/noise_collector.ino` to the Arduino using the Arduino IDE.

**Step 2:** Run the PC-side receiver script:

```bash
pip install pyserial    # if not already installed
python src/collect_noise.py --port /dev/ttyACM0 --output data/noise_samples/
# On Windows, use: --port COM3 (check Device Manager for the correct port)
```

**Step 3:** In the terminal, type commands to control the Arduino:

```
label idle_parking_lot    # Set the label
record 10                 # Record 10 consecutive 1.5s clips
label road_highway        # Change label
record 20                 # Record more clips
```

WAV files are saved automatically to `data/noise_samples/{label}/`.

---

## Step 10: Quantize Neural Networks to Int8 (Phase 4)

Quantize the float32 models to int8 TFLite for Arduino deployment. This script retrains each architecture on its best training condition, then converts via PTQ and QAT.

```bash
python src/quantize_nn.py
```

**Runtime:** ~14 minutes on GPU.

**What it produces:**

| File | Description |
|------|-------------|
| `models/m3_cnn2d_int8_ptq/tier{1,2,3}.tflite` | M2 Post-Training Quantized models |
| `models/m4_cnn2d_int8_qat/tier{1,2,3}.tflite` | M2 Quantization-Aware Trained models |
| `models/m5_cnn1d_int8_qat/tier{1,2,3}_ptq.tflite` | M5 PTQ models (QAT not supported for Conv1D) |
| `models/m6_dscnn_int8_qat/tier{1,2,3}_ptq.tflite` | M6 PTQ models |
| `models/m6_dscnn_int8_qat/tier{1,2,3}.tflite` | M6 QAT models |
| `results/quantization_results.json` | Full evaluation metrics for all int8 models |
| `results/quantization_summary.csv` | Summary comparison table |

The script configures which training condition each architecture uses (M2: Condition D, M5/M6: Condition B). These can be changed by editing `MODEL_CONDITIONS` at the top of `src/quantize_nn.py`.

**Note:** Requires `tensorflow-model-optimization` and `tf_keras` packages. The script sets `TF_USE_LEGACY_KERAS=1` automatically for compatibility.

---

## Step 11: Deploy to Arduino (Phase 5)

### 11a: Generate C headers

Export the selected model and feature extraction assets as C headers:

```bash
python src/export_for_arduino.py --model m6 --tier 2 --method ptq
```

This generates 4 files in `arduino/car_sound_classifier/`: `model_data.h`, `mel_filterbank.h`, `normalization.h`, `hann_window.h`.

To switch models, change `--model` to `m2` or `m5` (and `--method` to `qat` if desired).

### 11b: Validate feature parity

Before flashing, verify that the on-device feature extraction algorithm matches the Python training pipeline:

```bash
python src/validate_features.py
```

All clips should show cosine similarity >= 0.99 and int8 MAE <= 2.0.

### 11c: Compile and flash

1. Open `arduino/car_sound_classifier/car_sound_classifier.ino` in Arduino IDE
2. Install **Arduino Mbed OS Nano Boards** via Boards Manager
3. Install **Arduino_TensorFlowLite** via Library Manager
4. Select Board: **Arduino Nano 33 BLE**
5. Click Upload (Ctrl+U)
6. Open Serial Monitor at **115200 baud** to see classification results

### 11d: Run playback evaluation

Systematic evaluation using the test set played through a speaker:

```bash
python src/playback_test.py --port /dev/ttyS3 --num-clips 208 --delay 3.0
```

Use `--port COM3` on Windows. Place the Arduino near the speaker. Results are saved to `results/playback_test_results.json`.

### 11e: Compare Python vs Arduino mel-spectrograms

For slideshow / report figures, capture side-by-side spectrograms — librosa from the WAV file vs the spectrogram the Arduino actually computes on-device:

```bash
python src/compare_mel_spectrograms.py --port /dev/ttyS3
```

Press **SPACE** to capture each clip, **s** to skip, **q** to quit. For each clip the script saves `<clip>_python.png`, `<clip>_arduino.png`, and `<clip>_side_by_side.png` under `results/mel_comparison/`. Requires the latest `car_sound_classifier.ino` (which adds the `DUMP` serial command).

---

## Step 12: Review Comprehensive Evaluation (Phase 6)

The evaluation notebook consolidates results from all project phases across all 4 evaluation dimensions (classification performance, quantization impact, on-device benchmarking, and end-to-end playback test).

```bash
jupyter notebook notebooks/07_evaluation.ipynb
```

This notebook loads all pre-computed results from `results/` and generates master comparison tables and publication-quality figures. No model training or hardware is required.

**What it produces:**

| Output | Location |
|--------|----------|
| Master accuracy comparison | Displayed in notebook |
| Ablation study summary | Displayed in notebook |
| Publication figures (9 total) | `results/paper_*.png` |
| PC vs playback analysis | Displayed in notebook |

See also `docs/on_device_evaluation_report.md` for the complete Phase 6 synthesis report.

---

## Step 13: Report Assembly (Phase 7)

The report assembly documents in `docs/report/` collect all information, context, and references needed to write each section of the final research paper:

```
docs/report/
  01_abstract.md              Key numbers and findings summary
  02_introduction.md          Problem context, research questions
  03_data_sources.md          Dataset, preprocessing, augmentation, noise bank
  04_methodology.md           Models, training, quantization, deployment methods
  05_analysis.md              Interpretation of experimental observations
  06_results.md               Complete tables, figures, and quantitative data
  07_discussion_and_industry.md   Discussion, applications, ethics
  08_conclusion_and_future_work.md  Answers to research questions, limitations, future work
```

Each document contains bullet points, tables, file references, and figure paths — not polished prose. Use these as the basis for writing the final report.

---

## What's Next

All phases (0-7) are complete. The `docs/report/` directory contains the assembled material for writing the final research paper. Check `docs/project_spec/dev_log.md` for the full project history.

## Troubleshooting

- **`ModuleNotFoundError`** — Make sure your virtual environment is activated (`source .venv/bin/activate`).
- **Dataset verification fails** — Confirm the `car_diagnostics_dataset/` folder is in the project root and contains all the WAV files directly (not nested in an extra subfolder).
- **Out of memory during preprocessing** — Close other applications. The pipeline processes all 1,386 files and peak memory usage can reach ~2 GB.

## Quick Reference

| Task | Command |
|------|---------|
| Activate environment | `source .venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Verify dataset & splits | `python project_setup/verify_dataset_and_split.py` |
| Run preprocessing | `python src/run_preprocessing.py` |
| Train classical ML baselines | `python src/train_classical.py` |
| Generate augmented data | `python src/generate_augmented_data.py` |
| Train neural networks | `python src/train_nn.py` |
| Quantize to int8 | `python src/quantize_nn.py` |
| Export for Arduino | `python src/export_for_arduino.py --model m6 --tier 2 --method ptq` |
| Validate features | `python src/validate_features.py` |
| Run playback test | `python src/playback_test.py --port /dev/ttyS3` |
| Compare mel spectrograms (Py vs Arduino) | `python src/compare_mel_spectrograms.py --port /dev/ttyS3` |
| Review evaluation | `jupyter notebook notebooks/07_evaluation.ipynb` |
