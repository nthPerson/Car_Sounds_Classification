# Development Log

This log tracks major progress, decisions, and results across the project. Add new entries at the top so the most recent work is always first.

---

## 2026-04-28 — Mel-spectrogram comparison tool (Python vs Arduino)

**What was done:**
- Added a `DUMP` serial command to `arduino/car_sound_classifier/car_sound_classifier.ino`. It runs the normal capture-classify cycle but also streams the raw 40×92 log-mel spectrogram (in dB, post `power_to_db(ref=np.max, top_db=80)`, before z-score normalization) over Serial as `MELDUMP_START|...` / `MELDUMP_END` framed CSV. The existing `READY` / `AUTO` / `STOP` commands are unchanged.
- Added `src/compare_mel_spectrograms.py`. For each clip it plays the WAV through the speaker, reads the on-device spectrogram via the new `DUMP` command, computes the librosa-equivalent spectrogram from the original WAV, and saves three PNGs: `<clip>_python.png`, `<clip>_arduino.png`, `<clip>_side_by_side.png` (default output dir: `results/mel_comparison/`). UX mirrors `presentation_demo.py` — keyboard-driven (SPACE/s/q) with a live preview window. Reuses the helpers from `presentation_demo.py` for serial / playback so the two tools stay in sync.
- Both spectrograms are plotted on a shared dB color scale (`vmin=-80, vmax=0`), so the side-by-side view shows the genuine acoustic differences caused by speaker / room / PDM-mic transduction.

**Why:** Needed for slideshow figures comparing the "ideal" spectrogram (from the WAV file) against the spectrogram the Arduino actually computes from speaker-played audio captured on-device — directly visualizes the speaker-ablation story.

**Status:** Tool ready to run. Re-flash the sketch to pick up the new `DUMP` command before using `compare_mel_spectrograms.py`.

---

## 2026-03-26 — Phase 7: Final Report Assembly

**What was done:**
- Created 8 report assembly documents in `docs/report/` covering all paper sections: Abstract, Introduction, Data Sources, Methodology, Analysis, Results, Discussion & Industry Applications, Conclusion & Future Work
- Each document collects key numbers, bullet points, file references, figure paths, and analysis summaries for its section
- Ran 4-agent parallel validation pass cross-referencing all documents against actual result files
- Fixed errors identified by validation: 3 accuracy values in master table (M6 float32, M6 QAT, M5 float32), Tier 3 RF vs SVM comparison direction, tensor arena budget clarification
- Added missing discussion topics: M2 accuracy-F1 paradox, training instability analysis, QAT agreement paradox
- Added missing limitations: hyperparameter search scope, TF version constraint, training instability

**Report assembly documents:**
- `docs/report/01_abstract.md` — Key numbers for ~250-word abstract
- `docs/report/02_introduction.md` — Problem context, research questions, evaluation framework
- `docs/report/03_data_sources.md` — Dataset, preprocessing, augmentation, noise bank
- `docs/report/04_methodology.md` — All methods, architectures, training configs
- `docs/report/05_analysis.md` — Interpretation of all experimental observations
- `docs/report/06_results.md` — Complete quantitative tables and figure index
- `docs/report/07_discussion_and_industry.md` — Findings discussion, applications, ethics
- `docs/report/08_conclusion_and_future_work.md` — Research question answers, 15 limitations, 12 future work items

**Status:** Phase 7 complete. Report assembly provides comprehensive basis for writing the final research paper.

---

## 2026-03-26 — Phase 6 Complete: On-Device Evaluation

**What was done:**
- Created comprehensive evaluation notebook (`notebooks/07_evaluation.ipynb`) consolidating all 4 evaluation dimensions (D1-D4) with 9 publication-quality figures
- Created on-device evaluation report (`docs/on_device_evaluation_report.md`) synthesizing all Phase 6 findings
- Ran data verification pass: 18/18 checks passed across all result files
- Updated `GETTING_STARTED.md` with Phase 6 evaluation section

**Key results (M6 DS-CNN PTQ, Tier 2):**
- PC Int8: F1 macro 0.7835, accuracy 78.4%
- Bluetooth playback: F1 macro 0.5465, accuracy 68.3% (30.3% degradation)
- Quantization: nearly lossless (all McNemar p > 0.05, PTQ sufficient)
- On-device: 783 ms inference, 197 ms features, 2,470 ms total cycle
- Memory: 319 KB flash (33%), 179 KB SRAM (68%)

**Publication figures generated:** `results/paper_*.png` (9 figures covering master accuracy, ablation, quantization impact, accuracy vs size, latency breakdown, memory utilization, PC vs playback recall, speaker ablation, confidence calibration)

**Status:** Phase 6 complete. All evaluation dimensions documented. Ready for Phase 7 (Final Report & Presentation).

---

## 2026-03-25 — Phase 5: Playback Test Complete

**What was done:**
- Ran the automated playback evaluation (`src/playback_test.py`) with all 208 test clips on two playback configurations
- Fixed synchronization: modified Arduino sketch to use triggered mode (`READY`/`AUTO`/`STOP` commands) instead of continuous classification, and updated playback script to wait for "Listening..." before playing audio
- Fixed `KeyError: 'report_str'` in playback script (correct key: `classification_report_str`)

**Results — Speaker Hardware Ablation:**

| Metric | Laptop Speakers | Bluetooth Speaker | PC Int8 |
|--------|----------------|-------------------|---------|
| Accuracy | 0.2500 | **0.6827** | 0.7837 |
| F1 Macro | 0.1540 | **0.5465** | 0.7835 |
| Degradation | -80.3% | **-30.3%** | — |

- Laptop speakers failed catastrophically: 72% of predictions were startup classes due to missing bass response (speakers roll off below ~300 Hz, losing engine idle/braking low-frequency content)
- Bluetooth speaker restored performance to 68.3% accuracy with 30.3% F1 degradation from PC evaluation
- Braking Fault achieved PC-equivalent recall (81.8%) through Bluetooth speaker
- Confidence well-calibrated: correct=0.81, incorrect=0.66

**Key finding:** Audio reproduction quality is the dominant factor in playback test performance. In real deployment (mic near actual engine), there is no speaker in the signal path, so the Bluetooth results represent a conservative lower bound.

**Status:** Phase 5 complete. All four evaluation dimensions now have data. Ready for Phase 6 (Evaluation & Reporting).

---

## 2026-03-25 — Phase 5: On-Device Testing and CMSIS-NN Optimization Investigation

**What was done:**
- Successfully compiled and uploaded the car sound classifier sketch to the Arduino Nano 33 BLE Sense Rev2
- Measured actual resource usage: 319 KB flash (33%), 175 KB SRAM (68%), 64 KB tensor arena (63,556 bytes used)
- Switched from `AllOpsResolver` to `MicroMutableOpResolver<5>` — saved 165 KB flash (484 KB → 319 KB)
- Increased tensor arena from 50 KB to 64 KB after `AllocateTensors()` reported needing 59,456 bytes
- Built TFLite Micro from source with CMSIS-NN optimized kernels for Cortex-M4 SIMD acceleration
- Packaged as Arduino-compatible library (`TensorFlowLite_CMSIS_NN`), resolving include path conflicts, duplicate symbol issues, ACLE intrinsic compatibility, and `CMSIS_NN` preprocessor guards

**Measured on-device performance:**
- Audio capture: 1,490 ms (fixed, 1.5s @ 16 kHz)
- Feature extraction: ~197 ms (CMSIS-DSP FFT mel-spectrogram)
- Model inference: ~783 ms (M6 DS-CNN int8, 6M MACs, 8.4 cycles/MAC)
- Total cycle: ~2,470 ms (one classification every 2.5 seconds)

**CMSIS-NN SIMD investigation result:**
- Built CMSIS-NN from tflite-micro source — compiled successfully on Arduino
- Achieved only 5% speedup (828 ms → 783 ms) instead of expected 4.6x
- Root cause: Arduino's bundled GCC 7.2.1 lacks ACLE intrinsic support for `__sxtb16`, `__smlabb`, `__smlatt` — CMSIS-NN selects SIMD code paths but compiler cannot emit SIMD instructions
- Pre-compiled `.a` approach (GCC 14.3.1 with real SIMD) failed due to C++ ABI incompatibility
- Conclusion: Full CMSIS-NN SIMD requires GCC 10+ (not available in Arduino Mbed OS board package v4.5.0)

**Status:** Deployment complete and functional. Classifier runs continuously. Ready for playback test evaluation.

---

## 2026-03-24 — Phase 5: Arduino Deployment (Code Complete)

**What was done:**
- Created `src/export_for_arduino.py` — generates all C headers from Python (model bytes, sparse mel filterbank, normalization stats, Hann window). Extracts quantization params and TFLite ops from the model.
- Created `src/validate_features.py` — feature parity validation comparing simulated on-device algorithm (manual FFT + sparse mel filterbank + global-max log + z-score) against librosa training pipeline
- Created Arduino sketch in `arduino/car_sound_classifier/` (8 files):
  - `car_sound_classifier.ino` — main sketch with PDM capture, feature extraction, TFLite Micro inference, Serial output
  - `feature_extraction.cpp/.h` — CMSIS-DSP mel-spectrogram computation (5 phases: peak norm, frame FFT, mel energy, log conversion, z-score)
  - `config.h` — all compile-time constants
  - 4 generated headers: `model_data.h` (M6 PTQ, 21.3 KB), `mel_filterbank.h` (sparse, 490 weights), `normalization.h`, `hann_window.h`
- Created `src/playback_test.py` — automated playback evaluation: plays test set audio through speaker, captures Arduino predictions over Serial, computes full evaluation metrics
- Created `docs/deployment_results.md` — deployment documentation

**Feature parity validation results (24 clips, all PASS):**
- Cosine similarity: 1.0000 (perfect match between simulated on-device and librosa)
- Int8 exact match rate: 96.3% (3.7% differ by rounding at quantization boundaries)
- Mean absolute error (int8): 0.04 (acceptance threshold: <= 2.0)

**Key design decisions:**
- Sparse mel filterbank reduces flash from 40.2 KB to 2.1 KB (95.2% sparse)
- `ref=np.max` log normalization makes pipeline scale-invariant, eliminating CMSIS-DSP vs numpy FFT scaling concerns
- Machine-readable `RESULT|...` output format enables automated evaluation by `playback_test.py`
- AllOpsResolver used for initial deployment (simplicity); switch to MicroMutableOpResolver for production

**Status:** Code complete. Awaiting Arduino hardware for compilation, flashing, and playback test. Feature parity validated in Python — on-device accuracy should closely match PC int8 evaluation (F1=0.7835).

---

## 2026-03-24 — Phase 4 Complete: Quantization (PTQ & QAT)

**What was done:**
- Implemented `src/quantize_nn.py` — full quantization pipeline: retrain float32 models under tf_keras, PTQ conversion, QAT fine-tuning, int8 TFLite evaluation, quantization impact analysis, visualizations
- Retrained M5/M6 on Condition B (their best training condition) and M2 on Condition D under legacy Keras (tf_keras) for .h5 compatibility with `tensorflow-model-optimization`
- Produced 15 int8 TFLite models: 9 PTQ (all architectures) + 6 QAT (M2 and M6; M5 Conv1D not supported by tfmot)
- Computed full quantization impact: accuracy/F1 deltas, prediction agreement rates, McNemar's significance tests, confidence distribution analysis
- Created `docs/quantization_results.md` — research-paper documentation of methods, results, and analysis

**Key results (Tier 2 — primary deployment target):**

| Model | Method | Int8 F1 Macro | Size (KB) | Delta F1 | Agreement |
|-------|--------|--------------|-----------|----------|-----------|
| M6 DS-CNN | PTQ | **0.7835** | 21.3 | -0.001 | 98.6% |
| M6 DS-CNN | QAT | 0.7774 | 22.3 | +0.005 | 95.2% |
| M2 2-D CNN | QAT | 0.7426 | 20.5 | +0.001 | 95.7% |
| M2 2-D CNN | PTQ | 0.7413 | 20.1 | +0.003 | 99.0% |
| M5 1-D CNN | PTQ | 0.6138 | 15.2 | -0.015 | 99.0% |

**Key findings:**
- **Quantization is nearly lossless.** No model-tier-method combination shows a statistically significant accuracy change (all McNemar p > 0.05). Int8 models perform as well as float32 on this task.
- **PTQ is sufficient.** PTQ performs as well as or slightly better than QAT across all architectures. The small model sizes (6K-15K parameters) and BatchNorm folding leave minimal quantization error.
- **M6 DS-CNN PTQ is the recommended deployment model.** Highest Tier 2 F1 (0.7835), 21.3 KB size, estimated ~35-45 KB tensor arena — all well within hardware budgets.
- **M5 QAT is not possible.** The `tfmot` library (v0.8.0) does not support Conv1D layers. M5 uses PTQ only, which is adequate given PTQ's strong performance.
- **Model sizes:** M2: 20 KB, M5: 15 KB, M6: 21 KB — all far under the 150 KB flash budget. The 4-12x compression from float32 .h5 exceeds the theoretical 4x because .h5 includes optimizer state and metadata.

**Technical notes:**
- Required `tf_keras==2.18.0` and `TF_USE_LEGACY_KERAS=1` environment variable for `tensorflow-model-optimization` compatibility with TF 2.18
- TF 2.21 cuDNN autotuner issue persists — TF 2.18 remains required for GPU training
- QAT used 10x lower learning rate than original training (M2: 0.0002, M6: 0.0002) with 30 max epochs and patience=10 early stopping

**Status:** Phase 4 complete. 15 int8 TFLite models produced. M6 DS-CNN PTQ selected as primary deployment candidate. Ready for Phase 5 (Arduino deployment).

---

## 2026-03-22 — Real-World Noise Data Collection + Noise-Augmented Training (Condition C)

**What was done:**
- Created Arduino sketch (`arduino/noise_collector/noise_collector.ino`) and PC receiver script (`src/collect_noise.py`) for data collection
- Collected 360 real-world background noise samples using the Arduino Nano 33 BLE Sense Rev2 PDM microphone in/around a 2008 VW Jetta (23 categories: engine idle, road noise, HVAC, wind, ambient, cabin driving)
- Built noise analysis notebook (`notebooks/04_noise_analysis.ipynb`) — validated quality, grouped into 6 noise categories with operational-state-aware mapping
- Built noise augmentation notebook (`notebooks/05_noise_augmentation.ipynb`) — generated `train_noise_augmented.npz` (3,317 samples with real-world noise at SNR 5-20 dB)
- Modified `src/train_nn.py` — replaced `USE_AUGMENTED_DATA` flag with `AUGMENTATION_VARIANT` ("none"/"standard"/"noise") and added `SKIP_HP_SEARCH` to reuse existing hyperparameter configs
- Trained all 9 NN models on noise-augmented data (Condition C), reusing Condition B hyperparameters
- Archived Condition B results for ablation comparison

**Three-condition ablation results (F1 Macro):**

| Model | Tier | Cond. A (Original) | Cond. B (Synthetic) | Cond. C (Noise) | B→C Δ |
|-------|------|-------------------|--------------------|--------------------|-------|
| M2 (2-D CNN) | 1 | 0.868 | 0.898 | 0.861 | −0.037 |
| M2 (2-D CNN) | 2 | 0.666 | 0.732 | **0.734** | +0.002 |
| M2 (2-D CNN) | 3 | 0.715 | 0.717 | **0.722** | +0.005 |
| M5 (1-D CNN) | 1 | 0.875 | 0.864 | 0.874 | +0.010 |
| M5 (1-D CNN) | 2 | 0.616 | 0.638 | 0.596 | −0.042 |
| M5 (1-D CNN) | 3 | 0.616 | 0.657 | 0.612 | −0.045 |
| M6 (DS-CNN) | 1 | 0.858 | 0.900 | 0.890 | −0.010 |
| M6 (DS-CNN) | 2 | 0.571 | 0.777 | 0.739 | −0.038 |
| M6 (DS-CNN) | 3 | 0.600 | 0.725 | **0.762** | +0.037 |

**Key observations:**
- B→C deltas are small (±0.01-0.05) on the clean test set, which is expected — noise injection improves robustness to deployment-time noise, not clean-test accuracy
- M6 Tier 3 shows the clearest improvement (+0.037), suggesting noise augmentation helps the most complex classification task
- M5 (1-D CNN on MFCCs) shows consistent degradation with noise injection — MFCCs may already discard the frequency bands where real-world noise provides useful regularization
- Condition C models are the production models for Phase 4 quantization, chosen for expected on-device robustness over marginal clean-test-set differences

**Documentation created:**
- `docs/noise_data_collection_and_augmentation.md` — complete documentation of noise collection methodology, hardware, protocol, augmentation approach, and limitations

**Status:** Phase 3 retrained with noise-augmented data (Condition C). Continued with Condition D (combined).

---

## 2026-03-22 — Combined Augmentation Training (Condition D)

**What was done:**
- Generated combined training features (`train_combined.npz`): 5,664 samples = all of Condition B (970 originals + 2,347 synthetic) + 2,347 noise-augmented copies from Condition C
- Added `"combined"` option to `AUGMENTATION_VARIANT` in `src/train_nn.py`
- Trained all 9 NN models on combined data, reusing Condition B hyperparameters
- Updated ablation study to 4 conditions (A, B, C, D)

**Combined dataset class distribution (Tier 2):**

| Class | Count | % |
|-------|-------|---|
| Normal Braking | 1,026 | 18.1% |
| Braking Fault | 1,007 | 17.8% |
| Normal Idle | 925 | 16.3% |
| Idle Fault | 552 | 9.7% |
| Normal Start-Up | 1,075 | 19.0% |
| Start-Up Fault | 1,079 | 19.1% |

Note: class imbalance is inverted — Idle Fault (no augmentation) is now the smallest class. `compute_class_weight('balanced')` compensates during training.

**Key results (F1 Macro, Condition D vs best prior):**

| Model | Tier 1 | Tier 2 | Tier 3 |
|-------|--------|--------|--------|
| M2 (2-D CNN) | 0.898 (=B) | **0.751** (best, +0.019 vs B) | **0.764** (best, +0.047 vs B) |
| M5 (1-D CNN) | 0.858 (B better) | 0.625 (B better) | 0.651 (B better) |
| M6 (DS-CNN) | **0.909** (best) | 0.742 (B better) | 0.686 (C better) |

**Key observations:**
- M2 is the clear winner from Condition D — achieves its best scores on Tiers 2 (+0.019) and 3 (+0.047), suggesting the 2-D CNN architecture benefits most from data volume
- M6 achieves its best Tier 1 score (0.909) but doesn't improve on Tiers 2/3, possibly due to the inverted class imbalance
- M5 (MFCC-based) does not benefit from combined data — Condition B remains best
- No single condition is universally best; the optimal choice depends on architecture and tier

**Phase 4 quantization candidates (Tier 2):**
1. M6 + Condition B (F1=0.777) — best Tier 2 overall
2. M2 + Condition D (F1=0.751) — best M2 score
3. M6 + Condition D (F1=0.742)

**Status:** All 4 ablation conditions complete (A, B, C, D). Ready to select best model+condition for Phase 4 (quantization).

---

## 2026-03-18 — Waveform Augmentation + Phase 3 Retrain

**What was done:**
- Implemented `src/generate_augmented_data.py` — pre-computed waveform augmentation with class-aware balancing
- Expanded training set from 970 → 3,317 samples using 5 waveform augmentation techniques (time shift, noise injection, pitch shift, speed perturbation, random gain)
- Class-balanced augmentation: minority classes received higher multipliers (up to 12x for Normal Start-Up) to target ~550 samples per Tier 2 class
- Retrained all 9 NN models with the augmented + balanced dataset
- Updated `src/train_nn.py` with `USE_AUGMENTED_DATA` flag and support for augmented normalization stats

**Augmentation multipliers (Tier 2):**

| Class | Original | Multiplier | Final |
|-------|----------|-----------|-------|
| Normal Braking | 54 | 9x | 540 |
| Braking Fault | 53 | 9x | 530 |
| Normal Idle | 185 | 2x | 555 |
| Idle Fault | 552 | 0x | 552 |
| Normal Start-Up | 43 | 12x | 559 |
| Start-Up Fault | 83 | 6x | 581 |

**Results comparison (F1 Macro, before → after augmentation):**

| Model | Tier 1 | Tier 2 | Tier 3 |
|-------|--------|--------|--------|
| M2 (2-D CNN) | 0.868 → **0.898** | 0.666 → **0.732** | 0.715 → **0.717** |
| M5 (1-D CNN) | 0.875 → 0.864 | 0.616 → **0.638** | 0.616 → **0.657** |
| M6 (DS-CNN) | 0.858 → **0.900** | 0.571 → **0.777** | 0.600 → **0.725** |
| Best Classical (SVM) | 0.916 | 0.699 | 0.711 |

**Key observations:**
- M6 (DS-CNN) showed the largest improvement — Tier 2 jumped from 0.571 to 0.777 (+0.206), confirming DS-CNN is highly sensitive to dataset size and class balance
- M6 now achieves the best Tier 2 F1 macro (0.777), exceeding all classical ML baselines (SVM: 0.699)
- All three NN architectures now exceed classical baselines on Tier 2 and Tier 3
- M6's training stability issue at higher learning rates was resolved — can now train at LR=0.002 with augmented data
- With augmented data, the hyperparameter search selected no SpecAugment for M2 and M6 (waveform augmentation provides sufficient regularization), but default SpecAugment for M5 (MFCCs)

**Status:** Phase 3 retrained with augmented data. Models significantly improved. Ready for Phase 4 (quantization).

---

## 2026-03-18 — Phase 3 Complete: Neural Network Training

**What was done:**
- Implemented `src/models.py` — three architecture definitions: M2 (compact 2-D CNN, ~14.6K params), M5 (1-D CNN on MFCCs, ~6.2K params), M6 (DS-CNN 32-ch, ~8.2K params)
- Implemented `src/train_nn.py` — full training orchestrator with data loading, on-the-fly SpecAugment via custom Keras Sequence, structured hyperparameter search on Tier 2, training with early stopping and LR scheduling, test set evaluation
- Created `notebooks/03_neural_networks.ipynb` — analysis notebook comparing NNs to classical baselines
- Created `docs/neural_network_models.md` — research-paper documentation with architecture details, search results, and analysis
- Required downgrade to TensorFlow 2.18.0 for GPU compatibility with RTX 2000 Ada (TF 2.21 cuDNN autotuner failures)

**Hyperparameter search results (Tier 2):**

| Architecture | Best LR | Best Dropout | Best Augmentation |
|-------------|---------|-------------|-------------------|
| M2 (2-D CNN) | 0.002 | 0.3 | Aggressive SpecAugment (F=10, T=15) |
| M5 (1-D CNN) | 0.002 | 0.2 | None |
| M6 (DS-CNN) | 0.0005 | N/A | Default SpecAugment |

**Test set results:**

| Model | Tier 1 (F1 Macro) | Tier 2 (F1 Macro) | Tier 3 (F1 Macro) |
|-------|------------------|------------------|------------------|
| M2 (2-D CNN) | 0.8679 | **0.6658** | **0.7150** |
| M5 (1-D CNN) | **0.8748** | 0.6164 | 0.6163 |
| M6 (DS-CNN) | 0.8580 | 0.5706 | 0.5997 |
| Best Classical (SVM) | 0.9163 | 0.6992 | 0.7106 |

**Key observations:**
- NNs did not exceed classical ML baselines on Tiers 1–2, but M2 slightly exceeded on Tier 3 (0.715 vs 0.711)
- With only 970 training samples and SpecAugment-only augmentation, classical hand-crafted features remain competitive
- M2 (2-D CNN) is the best NN architecture across all tiers and the primary quantization candidate for Phase 4
- M6 (DS-CNN) struggled significantly — training instability at LR ≥ 0.001, lowest accuracy across all tiers
- "Normal Start-Up" recall improved from 11–22% (classical ML) to 55.6% (M2 NN), though still the weakest class
- M5 (1-D CNN) is the safest deployment candidate due to small tensor arena (~20 KB) despite lower accuracy

**Design decisions:**
- SpecAugment only (no waveform augmentation) for speed; waveform aug can be added later if accuracy is insufficient
- Hyperparameter search on Tier 2 first, then train all tiers with best config per architecture
- TensorFlow 2.18.0 used instead of 2.21.0 due to cuDNN backend incompatibility with Ada Lovelace GPUs

**Status:** Phase 3 complete. 9 float32 models trained and evaluated. Ready for Phase 4 (quantization).

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
