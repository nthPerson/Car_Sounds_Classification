# 3. Methodology

This section describes the experimental pipeline used to train, quantize, deploy, and evaluate audio classification models for real-time automotive fault detection on a microcontroller. The methodology spans five stages: classical machine learning baselines, neural network architecture design and training, data augmentation, integer quantization for edge deployment, and a four-dimensional evaluation framework that captures both PC-based accuracy and real-world on-device performance.

## 3.1 Classical ML Baselines

Two classical machine learning algorithms were trained as non-deployable baselines to establish reference performance using hand-crafted features.

**Random Forest.** A scikit-learn `RandomForestClassifier` was tuned via `RandomizedSearchCV` with 5-fold stratified cross-validation over 100 iterations. The search space covered `n_estimators` in {100, 200, 500}, `max_depth` in {10, 20, 30, None}, `min_samples_split` in {2, 5, 10}, `min_samples_leaf` in {1, 2, 4}, `max_features` in {"sqrt", "log2", 0.3}, and `class_weight` in {"balanced", "balanced_subsample"}.

**Support Vector Machine.** A scikit-learn `SVC` was tuned via `GridSearchCV` with 5-fold stratified cross-validation. Two kernel configurations were searched: an RBF kernel with `C` in {0.01, 0.1, 1, 10, 100} and `gamma` in {"scale", "auto", 0.001, 0.01, 0.1}, and a linear kernel with the same `C` range.

Both classifiers received 441-dimensional feature vectors as input. These vectors were constructed by computing 63 frame-level audio descriptors—13 MFCCs, 13 delta MFCCs, 13 delta-delta MFCCs, spectral centroid, spectral bandwidth, 7-band spectral contrast, spectral rolloff, zero-crossing rate, RMS energy, and 12 chroma bins—and then aggregating each descriptor across time using seven statistics (mean, standard deviation, minimum, maximum, median, skewness, and kurtosis). A `StandardScaler` was fit on the training set only and applied to all splits. Feature importance was assessed using the Random Forest's Gini-based `feature_importances_` attribute, and a reduced-feature experiment retrained the Random Forest on only the top 50 most important features to quantify redundancy in the 441-dimensional space.

## 3.2 Neural Network Architectures

Three convolutional neural network architectures were designed for deployment on the Arduino Nano 33 BLE Sense Rev2, balancing classification capacity against the microcontroller's constraints of 150 KB model flash and 80 KB tensor arena SRAM. Each architecture was trained across all three taxonomy tiers (Tier 1: 2-class binary; Tier 2: 6-class state-aware; Tier 3: 9-class individual faults). Parameter counts reported below correspond to the Tier 2 configuration.

### 3.2.1 M2 — Compact 2-D CNN (14,566 parameters)

The primary architecture is a three-block 2-D CNN operating on log-mel spectrograms of shape (40, 92, 1). Each block consists of a Conv2D layer with 3×3 kernels and same padding, followed by batch normalization, ReLU activation, and 2×2 max pooling; the third block omits max pooling and feeds directly into global average pooling. The three convolutional layers use 16, 32, and 32 filters respectively. Global average pooling is followed by a dropout layer (rate = 0.3, determined by hyperparameter search), and a dense softmax output layer. This architecture was chosen as the primary candidate for its proven effectiveness in audio classification tasks while remaining compact enough for microcontroller deployment.

### 3.2.2 M5 — 1-D CNN on MFCCs (6,246 parameters)

A lighter alternative operating on MFCC frames of shape (92, 13). The architecture mirrors M2's three-block structure but uses 1-D convolutions along the time axis with kernel sizes of 5, 3, and 3 respectively. Channel counts follow the same 16→32→32 progression, with batch normalization, ReLU, and 1-D max pooling (pool size 2) after each convolutional layer. Global average pooling precedes dropout (rate = 0.2) and a dense softmax layer. This architecture offers the smallest tensor arena footprint (estimated 15–25 KB) and serves as a guaranteed-to-fit fallback.

### 3.2.3 M6 — Depthwise Separable CNN (8,166 parameters)

This architecture follows the DS-CNN-S design from the ARM ML Zoo, adapted to the project's input dimensions. It uses a 32-channel configuration (revised from an initial 64-channel design that would have exceeded the 80 KB SRAM budget). The network begins with a Conv2D layer using a 4×10 kernel with stride (2, 2) to reduce the spatial dimensions of the (40, 92, 1) input to (20, 46, 32). This is followed by four depthwise separable (DS) blocks, each consisting of a 3×3 depthwise convolution (same padding), batch normalization, ReLU, a 1×1 pointwise convolution, batch normalization, and ReLU. The DS blocks maintain the (20, 46, 32) spatial dimensions throughout. Global average pooling and a dense softmax layer complete the classifier. Depthwise separable convolutions factor standard convolutions into spatial (depthwise) and channel-mixing (pointwise) operations, reducing both parameters and multiply-accumulate operations (MACs) while maintaining representational capacity. Critically, CMSIS-NN provides optimized kernels for depthwise convolutions on Cortex-M4, making this architecture a strong candidate for hardware-accelerated inference.

### 3.2.4 Training Configuration

All neural networks shared a common training configuration. The Adam optimizer was used with sparse categorical cross-entropy loss. Class weights were computed using scikit-learn's `compute_class_weight('balanced')` to mitigate the severe class imbalance in the dataset (12.8:1 ratio between the largest and smallest Tier 2 classes). Training ran for up to 100 epochs with early stopping on validation loss (patience = 15, restoring best weights) and a ReduceLROnPlateau scheduler (factor = 0.5, patience = 5, minimum learning rate = 1×10⁻⁶). The batch size was 32 and all random seeds were fixed at 42 for reproducibility. A held-out validation set of 208 unaugmented samples was used for all intermediate decisions; the test set of 208 samples was reserved for final reporting only.

### 3.2.5 Hyperparameter Search

Hyperparameters were tuned on Tier 2 (the primary deployment target) and then applied to Tiers 1 and 3 without tier-specific optimization. For each architecture, a structured grid search was conducted sequentially over learning rate ({0.0005, 0.001, 0.002}) and dropout rate ({0.2, 0.3, 0.5}), selecting configurations by best validation accuracy. The optimal settings were: M2 at learning rate 0.002 with dropout 0.2 and no augmentation; M5 at learning rate 0.001 with dropout 0.3 and default augmentation; and M6 at learning rate 0.002 with no dropout sweep (fixed architecture) and no augmentation. Notably, M6 exhibited training instability on the original imbalanced dataset (validation accuracy collapsed to approximately 23% at learning rates ≥ 0.001), but trained stably after data augmentation and class balancing were applied.

## 3.3 Data Augmentation Strategy

Four training conditions were constructed for an ablation study to isolate the contributions of synthetic augmentation, class balancing, and real-world noise injection. All conditions shared the same unaugmented validation and test sets (208 samples each), the same model architectures, and the same hyperparameters.

**Condition A (Original).** The unmodified training set of 970 samples with its original class imbalance (43 to 552 samples per Tier 2 class).

**Condition B (Synthetic Augmentation + Class Balancing).** Five waveform-domain augmentations were applied stochastically to generate additional training samples: time shift (50% probability, ±100 ms), additive Gaussian noise (50%, SNR 5–25 dB), pitch shift (30%, ±2 semitones), speed perturbation (30%, 0.9–1.1× rate), and random gain (50%, 0.7–1.3× amplitude). Class-aware augmentation multipliers brought each Tier 2 class to approximately 550 samples, matching the size of the largest class (Idle Fault at 552). Per-class multipliers ranged from 0× (Idle Fault, no augmentation needed) to 12× (Normal Start-Up, the most underrepresented class with only 43 original samples). The resulting training set contained 3,317 samples with an imbalance ratio reduced from 12.8:1 to 1.1:1.

**Condition C (Noise-Augmented).** The same waveform augmentation pipeline as Condition B, with an additional stage of real-world noise injection applied to 100% of augmented samples. Noise was drawn from a bank of 360 recordings collected with the Arduino Nano 33 BLE Sense Rev2's on-board PDM microphone in and around a 2008 Volkswagen Jetta. The noise bank was organized into six groups (engine mechanical, HVAC, road/driving, cabin/driving, wind, and ambient/stationary) totaling 23 subcategories. Noise injection was operationally state-aware: braking samples received road, driving, and engine noise; idle samples received engine, HVAC, and ambient noise; and startup samples received ambient, wind, and HVAC noise. The signal-to-noise ratio (SNR) for each sample was uniformly drawn from {5, 10, 15, 20} dB, and mixing followed the standard additive formula: mixed = signal + scale × noise, where the scale factor was computed as √(signal_power / (noise_power × 10^(SNR/10))). The resulting training set was also 3,317 samples with the same class distribution as Condition B, enabling a controlled comparison of the isolated effect of noise injection.

**Condition D (Combined).** The union of Condition B's full dataset (970 originals plus 2,347 synthetic copies) and Condition C's 2,347 noise-augmented copies (excluding the original 970 to avoid duplication), yielding 5,664 training samples. This condition inverted the original class imbalance—Idle Fault (552 samples, no augmentation) became the smallest class—but `compute_class_weight('balanced')` compensated during training.

Each architecture was trained under its best-performing condition for subsequent quantization: M2 under Condition D, and M5 and M6 under Condition B. All random operations used deterministic seeds derived from a base random state of 42 to ensure full reproducibility.

## 3.4 Quantization

All neural network models were quantized to int8 representation using two methods to investigate the accuracy–efficiency trade-off central to the project's research questions.

### 3.4.1 Post-Training Quantization (PTQ)

PTQ was applied using the TensorFlow Lite converter with full-integer int8 quantization. A representative dataset of 200 stratified samples from the training set was used to calibrate activation quantization ranges via min-max estimation. The converter was configured with `supported_ops` restricted to `TFLITE_BUILTINS_INT8` (no float32 fallback), and both input and output types were set to int8. This produced models where all weights, activations, and I/O tensors operated in 8-bit integer arithmetic.

### 3.4.2 Quantization-Aware Training (QAT)

QAT was performed using the `tensorflow-model-optimization` (tfmot) library v0.8.0. The `quantize_model()` function was applied to the float32 Keras model to insert fake-quantization nodes that simulate int8 rounding during forward passes, allowing the model to adapt its weights to quantization effects during fine-tuning. QAT fine-tuning used a learning rate of 0.1× the original training rate (M2: 0.0002, M6: 0.0002), with a maximum of 30 epochs, early stopping (patience = 10), and ReduceLROnPlateau scheduling. After fine-tuning, the QAT model was converted to int8 TFLite using the same converter settings as PTQ. A notable limitation was that the tfmot library (v0.8.0) does not support Conv1D layers, so M5 could only be quantized via PTQ.

### 3.4.3 Technical Environment

Quantization required TensorFlow 2.18.0 and the legacy Keras backend (`tf_keras==2.18.0` with the `TF_USE_LEGACY_KERAS=1` environment variable) due to compatibility constraints between tfmot and the TensorFlow 2.18 API. Float32 models were retrained under this legacy backend to produce `.h5` files compatible with the quantization pipeline. TensorFlow 2.21 was not viable due to a cuDNN autotuner issue on Ada Lovelace GPUs.

In total, 15 int8 TFLite models were produced: 9 PTQ models (3 architectures × 3 tiers) and 6 QAT models (M2 and M6 only, across 3 tiers).

## 3.5 On-Device Deployment

### 3.5.1 Hardware Platform

The deployment target was the Arduino Nano 33 BLE Sense Rev2, featuring a Nordic nRF52840 system-on-chip with an ARM Cortex-M4F processor running at 64 MHz, 1,024 KB of flash memory (983 KB usable), 256 KB of SRAM, and an on-board ST MP34DT06JTR PDM MEMS microphone sampling at 16 kHz.

### 3.5.2 Model Export

The selected deployment model (M6 DS-CNN PTQ, Tier 2) was exported to C header files using a custom script (`src/export_for_arduino.py`). This script generated four headers: model weights as a byte array (21.3 KB), a sparse mel filterbank (490 non-zero weights out of 10,280 possible entries, achieving 95.2% sparsity and reducing flash usage from 40.2 KB to 2.1 KB), per-band normalization statistics (training set mean and standard deviation for z-score normalization), and a 512-point Hann window for FFT windowing.

### 3.5.3 On-Device Feature Extraction

The on-device feature extraction pipeline was implemented in C++ (`feature_extraction.cpp`) and executed a five-phase algorithm to transform raw PDM audio into a quantized int8 mel-spectrogram input tensor:

1. **Peak normalization:** The maximum absolute sample value across all 24,000 int16 samples (1.5 seconds at 16 kHz) was identified to establish a reference amplitude.
2. **Frame-by-frame FFT:** The audio was divided into 92 overlapping frames (512-sample window, 256-sample hop). Each frame was windowed with a Hann function and transformed via `arm_rfft_fast_f32()` from the CMSIS-DSP library, followed by power spectrum computation and application of the sparse mel filterbank (40 bands).
3. **Global-max log conversion:** Mel energies were converted to log scale using `10 × log₁₀(energy / max_energy)`, with values clamped to −80 dB. The global-maximum reference made the pipeline scale-invariant, eliminating concerns about FFT magnitude scaling differences between the CMSIS-DSP and librosa implementations.
4. **Z-score normalization:** Per-band normalization using the training set's mean and standard deviation.
5. **Int8 quantization:** Normalized float values were mapped to int8 using the TFLite model's input quantization parameters (scale and zero point), with clamping to the [−128, 127] range.

### 3.5.4 TFLite Micro Runtime

The TFLite Micro runtime was built from source from the official `tflite-micro` repository with CMSIS-NN optimized kernels, using the `create_tflm_tree.py` tool with `OPTIMIZED_KERNEL_DIR=cmsis_nn` and `TARGET_ARCH=cortex-m4+fp`. The generated source tree was packaged as an Arduino-compatible library (`TensorFlowLite_CMSIS_NN`) with include path fixes for flatbuffers, gemmlowp, ruy, and kissfft dependencies. Additional modifications included injecting `#define CMSIS_NN` into 14 kernel headers and adding ACLE intrinsic compatibility shims for the Arduino toolchain's GCC 7.2.1 compiler. A `MicroMutableOpResolver<5>` was used, registering only the five operations required by the model: Conv2D, DepthwiseConv2D, FullyConnected, Mean (for global average pooling), and Softmax. The tensor arena was allocated at 65,536 bytes (64 KB), of which 63,556 bytes were used.

### 3.5.5 Feature Parity Validation

To ensure the on-device feature extraction pipeline produced outputs equivalent to the Python training pipeline, a feature parity validation was conducted. A simulated on-device algorithm was implemented in Python (without librosa) and compared against the librosa-based training pipeline across 24 test clips (4 per Tier 2 class). Acceptance criteria required cosine similarity ≥ 0.99 and int8 mean absolute error ≤ 2.0. All 24 clips passed, achieving cosine similarity of 1.0000 and an int8 exact match rate of 96.3%, with the 3.7% of differing values attributable to rounding at quantization boundaries.

## 3.6 Evaluation Framework

The evaluation framework was organized into four dimensions, designed to capture both the traditional machine learning performance and the deployment-specific characteristics of the system.

### 3.6.1 Dimension 1 — Classification Performance (PC Evaluation)

All models were evaluated on the held-out test set (208 samples for Tiers 1 and 2; 143 samples for Tier 3) using standard classification metrics: overall accuracy, macro-averaged F1 score, weighted-averaged F1 score, and per-class precision, recall, and F1 scores. Confusion matrices (both raw counts and row-normalized) were generated for all model–tier combinations. For Tier 1 (binary classification), ROC curves and area under the curve (AUC) were additionally computed. The macro F1 score served as the primary ranking metric because it accounts for class imbalance by weighting all classes equally, unlike overall accuracy which can be inflated by strong majority-class performance.

### 3.6.2 Dimension 2 — Quantization Impact Analysis

This dimension constitutes the central academic contribution of the project. For each quantized model, four metrics were computed to characterize the effect of int8 quantization relative to the float32 baseline: the change in accuracy (ΔAccuracy), the change in macro F1 score (ΔF1), the prediction agreement rate (the percentage of test samples receiving the same predicted label from both the float32 and int8 models), and the statistical significance of any accuracy difference as assessed by McNemar's test. A McNemar p-value exceeding 0.05 indicates that the float32 and int8 models do not differ significantly in their error patterns. Additionally, confidence distribution shifts were analyzed by comparing the mean and standard deviation of the maximum softmax probability between float32 and int8 predictions.

### 3.6.3 Dimension 3 — On-Device Performance Benchmarking

Hardware performance was measured directly on the Arduino Nano 33 BLE Sense Rev2. Flash and SRAM utilization were read from the Arduino IDE compilation output. The tensor arena size was determined empirically via `interpreter->arena_used_bytes()`. Latency was measured using `micros()` timers around each pipeline phase: audio capture (fixed at 1,490 ms for 1.5 seconds of PDM recording), feature extraction (mel-spectrogram computation), and model inference (`interpreter->Invoke()`). Multiple inference cycles were averaged to compute mean and standard deviation of latency.

### 3.6.4 Dimension 4 — End-to-End System Evaluation (Playback Test)

The most realistic evaluation available without an actual vehicle involved playing back the 208 test-set audio clips through external speakers and allowing the Arduino's on-board PDM microphone to capture and classify them. An automated evaluation script (`src/playback_test.py`) coordinated the process: it sent a READY command over serial, waited for the Arduino to report "Listening...", played the audio clip through the speaker, then captured the Arduino's RESULT output over serial. The same classification metrics as Dimension 1 were computed against the ground-truth labels, and the difference between PC and playback accuracy quantified the domain gap introduced by the speaker–microphone signal chain and ambient noise.

A speaker hardware ablation study was conducted using two playback configurations—built-in laptop speakers and an external Bluetooth speaker—to characterize the effect of audio reproduction quality on end-to-end system accuracy. Both configurations used the same 208 test clips, the same Arduino placement, and the same automated evaluation protocol.

*[Figure placeholder: `results/paper_master_accuracy_tier2.png` — Master comparison of all models on Tier 2 F1 macro]*

*[Figure placeholder: `results/paper_ablation_tier2.png` — Data augmentation ablation showing Conditions B, C, D across architectures]*

*[Figure placeholder: `results/paper_quantization_impact_tier2.png` — Float32 vs. int8 F1 delta and agreement rates]*

*[Figure placeholder: `results/paper_f1_macro_vs_size_tier2.png` — Accuracy–size trade-off across all model variants]*

*[Figure placeholder: `results/paper_latency_breakdown.png` — On-device latency breakdown by pipeline phase]*

*[Figure placeholder: `results/paper_memory_utilization.png` — Flash and SRAM utilization relative to hardware budgets]*

*[Figure placeholder: `results/paper_pc_vs_playback_recall.png` — Per-class recall comparison between PC and Bluetooth playback]*

*[Figure placeholder: `results/paper_speaker_ablation.png` — Speaker hardware ablation: laptop vs. Bluetooth vs. PC reference]*

*[Figure placeholder: `results/paper_confidence_calibration.png` — Confidence calibration analysis]*
