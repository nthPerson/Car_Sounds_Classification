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

## What's Next

Later phases (model training, quantization, deployment) will be added to this guide as they are completed. Check `docs/dev_log.md` for the latest progress.

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
