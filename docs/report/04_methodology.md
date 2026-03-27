# Methodology — Report Assembly

## 4.1 Classical ML Baselines (M1)

### Random Forest
- **Implementation:** scikit-learn `RandomForestClassifier`
- **Hyperparameter search:** `RandomizedSearchCV` with 5-fold stratified CV, n_iter=100
  - `n_estimators`: [100, 200, 500]
  - `max_depth`: [10, 20, 30, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ["sqrt", "log2", 0.3]
  - `class_weight`: ["balanced", "balanced_subsample"]
- **Best params (Tier 2):** See `models/m1_classical/search_results.json`
- **Feature input:** 441-dimensional hand-crafted vectors
- **Feature scaling:** StandardScaler (fit on training set only, applied to all sets)

### SVM
- **Implementation:** scikit-learn `SVC`, two kernel configurations searched
- **Hyperparameter search:** `GridSearchCV` with 5-fold stratified CV
  - RBF kernel: `C` [0.01, 0.1, 1, 10, 100], `gamma` ["scale", "auto", 0.001, 0.01, 0.1]
  - Linear kernel: `C` [0.01, 0.1, 1, 10, 100]
- **Feature scaling:** StandardScaler in pipeline (fit on train only)
- **Best params (Tier 2):** See `models/m1_classical/search_results.json`

### Feature Importance Analysis
- RF `feature_importances_` for Tier 2
- Top features: delta2_mfcc_0, mfcc_0, spectral_centroid
- Top-50 feature subset experiment: retrain RF Tier 2 with same hyperparameters using only 50 most important features. Saved as `models/m1_classical/rf_tier2_top50.joblib`.
- Source: `src/train_classical.py`, `docs/classical_ml_baselines.md`

## 4.2 Neural Network Architectures

### M2 — 2D Compact CNN (14,566 params)

Input: (40, 92, 1) log-mel spectrogram

| Layer | Output Shape | Params | Details |
|-------|-------------|--------|---------|
| Conv2D | (40, 92, 16) | 160 | 3x3 kernel, ReLU, same padding |
| BatchNorm | (40, 92, 16) | 64 | |
| MaxPool2D | (20, 46, 16) | 0 | 2x2 |
| Conv2D | (20, 46, 32) | 4,640 | 3x3 kernel, ReLU |
| BatchNorm | (20, 46, 32) | 128 | |
| MaxPool2D | (10, 23, 32) | 0 | 2x2 |
| Conv2D | (10, 23, 32) | 9,248 | 3x3 kernel, ReLU |
| BatchNorm | (10, 23, 32) | 128 | |
| GlobalAvgPool2D | (32,) | 0 | |
| Dropout | (32,) | 0 | rate=0.3 (from HP search) |
| Dense | (6,) | 198 | softmax |

### M5 — 1D CNN on MFCCs (6,246 params)

Input: (92, 13) MFCCs

| Layer | Output Shape | Params | Details |
|-------|-------------|--------|---------|
| Conv1D | (92, 16) | 1,056 | kernel_size=5, ReLU |
| BatchNorm | (92, 16) | 64 | |
| MaxPool1D | (46, 16) | 0 | pool_size=2 |
| Conv1D | (46, 32) | 1,568 | kernel_size=3, ReLU |
| BatchNorm | (46, 32) | 128 | |
| MaxPool1D | (23, 32) | 0 | pool_size=2 |
| Conv1D | (23, 32) | 3,104 | kernel_size=3, ReLU |
| BatchNorm | (23, 32) | 128 | |
| GlobalAvgPool1D | (32,) | 0 | |
| Dropout | (32,) | 0 | rate from HP search |
| Dense | (6,) | 198 | softmax |

### M6 — Depthwise Separable CNN (8,166 params)

Input: (40, 92, 1) log-mel spectrogram

| Layer | Output Shape | Params | Details |
|-------|-------------|--------|---------|
| Conv2D | (20, 46, 32) | 1,312 | 4x10 kernel, stride=2, ReLU |
| BatchNorm | (20, 46, 32) | 128 | |
| DepthwiseConv2D | (20, 46, 32) | 320 | 3x3, same padding |
| BatchNorm | (20, 46, 32) | 128 | |
| Conv2D (pointwise) | (20, 46, 32) | 1,056 | 1x1 kernel |
| BatchNorm | (20, 46, 32) | 128 | |
| *(repeat DS block 3 more times)* | | | Same structure |
| GlobalAvgPool2D | (32,) | 0 | |
| Dropout | (32,) | 0 | rate from HP search |
| Dense | (6,) | 198 | softmax |

**Rationale for DS-CNN:** Depthwise separable convolutions factor standard convolutions into depthwise (spatial) + pointwise (channel mixing), reducing parameters and MACs while maintaining representational capacity. CMSIS-NN has optimized kernels for depthwise convolutions on Cortex-M4.

### Training Configuration (all NNs)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | sparse_categorical_crossentropy |
| Class weighting | sklearn `compute_class_weight('balanced')` |
| Early stopping | Monitor val_loss, patience=15, restore_best_weights=True |
| ReduceLROnPlateau | factor=0.5, patience=5, min_lr=1e-6 |
| Max epochs | 100 |
| Batch size | 32 |
| Validation | Held-out validation set (208 samples) |
| Random state | 42 |

### Hyperparameter Search
- **Method:** Grid search over learning rate and dropout (Tier 2 only, then applied to all tiers)
- **Learning rates:** {0.0005, 0.001, 0.002} (3 values)
- **Dropout rates:** M2: {0.2, 0.3, 0.5}; M5: {0.2, 0.3, 0.5}; M6: no dropout sweep (fixed)
- **Selection criterion:** Best validation accuracy on Tier 2
- **Note:** Tier 1 and Tier 3 models use the same hyperparameters selected for Tier 2 (no tier-specific optimization)
- **Results saved:** `models/hyperparameter_search.json`
- Source: `src/train_nn.py`, `docs/neural_network_models.md`

## 4.3 Data Augmentation Strategy

### Waveform Augmentations
Applied at the audio waveform level before feature extraction:

| Augmentation | Probability | Parameters | Rationale |
|-------------|------------|------------|-----------|
| Time shift | 50% | +/-100 ms | Temporal invariance |
| Gaussian noise | 50% | SNR 5-25 dB | Noise robustness |
| Pitch shift | 30% | +/-2 semitones | Frequency invariance |
| Speed perturbation | 30% | 0.9-1.1x | Duration/tempo invariance |
| Random gain | 50% | 0.7-1.3x | Amplitude invariance |

### Class Balancing
- Target: ~550 samples per Tier 2 class (matching largest class, Idle Fault at 552)
- Per-class multipliers: 0x (Idle Fault) to 12x (Normal Start-Up)
- Each augmented copy uses a unique random seed for reproducibility

### Real-World Noise Injection (Condition C)
- Applied to 100% of augmented samples (on top of waveform augmentation)
- Operational-state-aware noise selection (see Data Sources Section 7.3)
- SNR uniformly sampled from {5, 10, 15, 20} dB
- Source: `src/augmentation.py`, `docs/noise_data_collection_and_augmentation.md`

### Ablation Design
- Conditions B, C, D compared on all 3 architectures x 3 tiers
- Same hyperparameters used across conditions (from Condition B search)
- Same val/test sets (unaugmented) for fair comparison
- Source: `src/train_nn.py` (`AUGMENTATION_VARIANT` flag)

## 4.4 Quantization

### Post-Training Quantization (PTQ)
- **Method:** TFLite `TFLiteConverter` with full-integer int8 quantization
- **Representative dataset:** 200 stratified samples from training set
- **Input/output type:** int8 (full integer, no float fallback)
- **Representative dataset:** 200 samples stratified from training set
- **Calibration:** Min-max range estimation from representative dataset

### Quantization-Aware Training (QAT)
- **Library:** `tensorflow-model-optimization` (tfmot) v0.8.0
- **Method:** `tfmot.quantization.keras.quantize_model()` applied to float32 model
- **Fine-tuning:** 0.1x original learning rate, 30 max epochs, early stopping (patience 10)
- **Limitation:** Conv1D not supported by tfmot — M5 uses PTQ only
- **Environment:** `TF_USE_LEGACY_KERAS=1` required for tfmot compatibility with TF 2.18

### Quantization Evaluation
- **PC evaluation:** TFLite interpreter runs int8 model on full test set
- **Metrics:** Accuracy, F1 macro/weighted, per-class precision/recall
- **Impact analysis:** ΔAccuracy, ΔF1 macro, prediction agreement rate, McNemar's test (p-value for statistical significance), confidence distribution shift
- Source: `src/quantize_nn.py`, `docs/quantization_results.md`

## 4.5 On-Device Deployment

### Hardware Configuration
- Board: Arduino Nano 33 BLE Sense Rev2 (nRF52840)
- MCU: ARM Cortex-M4F @ 64 MHz
- Flash: 983 KB usable (of 1,024 KB)
- SRAM: 256 KB
- Microphone: MP34DT06JTR PDM MEMS (on-board), 16 kHz sampling

### On-Device Feature Extraction Pipeline
Five-phase algorithm implemented in `arduino/car_sound_classifier/feature_extraction.cpp`:

1. **Phase A — Peak normalization:** Find max |sample| across 24,000 int16 samples
2. **Phase B — Frame-by-frame FFT:** 92 frames x [extract 512 samples, Hann window, `arm_rfft_fast_f32()`, power spectrum, sparse mel filterbank (40 bands)]
3. **Phase C — Global-max log conversion:** `10 * log10(energy / max_energy)`, clamp to -80 dB
4. **Phase D — Z-score normalization:** Per-band using training set mean/std
5. **Phase E — Int8 quantization:** `int8 = clamp(round(float / scale) + zero_point, -128, 127)`

### Sparse Mel Filterbank
- 40 filters, 490 non-zero weights (of 40 x 257 = 10,280 possible)
- 95.2% sparsity — reduces flash from 40.2 KB (dense) to 2.1 KB (sparse)
- Generated by `src/export_for_arduino.py` from `librosa.filters.mel()`

### TFLite Micro Integration
- `MicroMutableOpResolver<5>`: Conv2D, DepthwiseConv2D, FullyConnected, Mean, Softmax
- Tensor arena: 65,536 bytes (64 KB), 63,556 bytes used
- Library: TFLite Micro built from source with CMSIS-NN (`TensorFlowLite_CMSIS_NN`)

### TFLite Micro Build from Source
- Official tflite-micro repository, `create_tflm_tree.py` with `OPTIMIZED_KERNEL_DIR=cmsis_nn`
- Packaged as Arduino library with include path fixes (flatbuffers, gemmlowp, ruy, kissfft)
- `#define CMSIS_NN` injected into 14 kernel headers
- ACLE intrinsic compatibility shims for GCC 7.2.1
- Source: `docs/deployment_results.md` Section 8.3, 10.5

### Feature Parity Validation
- Simulated on-device algorithm in Python (no librosa) compared against librosa pipeline
- 24 test clips (4 per Tier 2 class)
- Acceptance criteria: cosine similarity >= 0.99, int8 MAE <= 2.0
- Source: `src/validate_features.py`, `results/feature_parity_results.json`

## 4.6 Evaluation Framework

### D1: Classification Performance (PC)
- Metrics: accuracy, F1 macro, F1 weighted, per-class precision/recall/F1
- Confusion matrices (raw and normalized) for all model-tier combinations
- ROC curves and AUC for Tier 1 (binary)
- Computed via `src/evaluate.py`

### D2: Quantization Impact
- ΔAccuracy = float32_accuracy - int8_accuracy
- ΔF1 macro = float32_f1 - int8_f1
- Prediction agreement rate = % of test samples with same prediction
- McNemar's test: statistical significance of accuracy difference (p-value)
- Confidence distribution shift: mean softmax confidence change

### D3: On-Device Benchmarking
- Inference latency: `micros()` timer around `interpreter->Invoke()`
- Feature extraction latency: `micros()` timer around mel-spectrogram pipeline
- Flash/SRAM: Arduino IDE compilation output
- Tensor arena: `interpreter->arena_used_bytes()`

### D4: End-to-End Playback Test
- Protocol: `src/playback_test.py` sends READY, waits for "Listening...", plays audio, captures RESULT
- 208 test clips, 2 speaker configurations (laptop + Bluetooth)
- Metrics: same as D1, plus latency statistics and speaker ablation analysis
- Source: `docs/project_spec/evaluation_plan.md`, `src/playback_test.py`
