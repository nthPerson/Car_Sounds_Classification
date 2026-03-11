# Model Architecture Specification

**Parent document:** [project_overview.md](project_overview.md)
**Depends on:** [data_preprocessing_spec.md](data_preprocessing_spec.md)
**Date:** March 10, 2026

---

## Table of Contents

1. [Overview of Models to Train](#1-overview-of-models-to-train)
2. [Model M1 — Classical ML Baselines](#2-model-m1--classical-ml-baselines)
3. [Model M2/M3/M4 — Compact 2-D CNN (Mel-Spectrogram)](#3-model-m2m3m4--compact-2-d-cnn-mel-spectrogram)
4. [Model M5 — 1-D CNN (MFCC)](#4-model-m5--1-d-cnn-mfcc)
5. [Model M6 — Depthwise Separable CNN (DS-CNN)](#5-model-m6--depthwise-separable-cnn-ds-cnn)
6. [Training Configuration (Common to All Neural Networks)](#6-training-configuration-common-to-all-neural-networks)
7. [Hyperparameter Search Strategy](#7-hyperparameter-search-strategy)
8. [Loss Functions and Class Weighting](#8-loss-functions-and-class-weighting)
9. [Regularization Strategy](#9-regularization-strategy)
10. [Model Size Estimation and Memory Feasibility](#10-model-size-estimation-and-memory-feasibility)
11. [Training Infrastructure](#11-training-infrastructure)

---

## 1. Overview of Models to Train

The project trains a matrix of models across two dimensions: **architecture** and **class taxonomy tier**. Every neural network architecture is trained on all three taxonomy tiers (Tier 1: 2-class, Tier 2: 6-class, Tier 3: 9-class) to study how class count affects accuracy and deployability.

### 1.1 Model Matrix

| Model ID | Architecture | Input Features | Taxonomy Tiers | Deployable on Nano 33? |
|----------|-------------|----------------|---------------|------------------------|
| **M1** | Random Forest + SVM | Hand-crafted feature vectors (441-dim) | Tier 1, 2, 3 | No (PC baseline only) |
| **M2** | Compact 2-D CNN (float32) | Log-mel spectrogram (40×92×1) | Tier 1, 2, 3 | No (accuracy ceiling) |
| **M3** | Compact 2-D CNN (int8 PTQ) | Log-mel spectrogram (40×92×1) | Tier 1, 2, 3 | **Yes** |
| **M4** | Compact 2-D CNN (int8 QAT) | Log-mel spectrogram (40×92×1) | Tier 1, 2, 3 | **Yes** |
| **M5** | 1-D CNN (int8 QAT) | MFCC frames (92×13) | Tier 1, 2, 3 | **Yes** |
| **M6** | DS-CNN (int8 QAT) | Log-mel spectrogram (40×92×1) | Tier 1, 2, 3 | **Yes** (if fits) |

**Total model variants:** 6 architectures × 3 tiers = **18 trained models** (plus quantized variants of each neural network).

### 1.2 Why These Architectures

- **M1 (Classical ML):** Establishes a non-deep-learning baseline. Shows what is achievable with engineered features and traditional algorithms. Not deployable on-device, but valuable for academic comparison.
- **M2/M3/M4 (2-D CNN):** The primary architecture. A 2-D CNN on mel-spectrograms is the most proven approach for audio classification. M2 is the float32 baseline; M3 and M4 are the quantized deployable variants (PTQ and QAT respectively). Comparing M2→M3 and M2→M4 directly measures quantization impact.
- **M5 (1-D CNN):** A lighter alternative that operates on MFCC features. Uses fewer parameters and less memory, making it the fallback if the 2-D CNN doesn't fit on-device. Also tests whether the more compact MFCC representation sacrifices accuracy.
- **M6 (DS-CNN):** Depthwise separable convolutions are the state-of-the-art for keyword spotting on microcontrollers (the "micro_speech" reference architecture). Tests whether the efficiency gains of depthwise separable convolutions help on this task.

---

## 2. Model M1 — Classical ML Baselines

### 2.1 Purpose

Classical ML models serve as a **floor for acceptable performance**. If a neural network can't beat a Random Forest on the same task, something is wrong with the neural network pipeline. They also provide interpretability (feature importances) that helps understand the dataset.

### 2.2 Input Features

The 441-dimensional hand-crafted feature vector described in [data_preprocessing_spec.md](data_preprocessing_spec.md) Section 7.3:
- 63 frame-level features (MFCCs + deltas, spectral features, temporal features)
- 7 aggregation statistics per feature (mean, std, min, max, median, skewness, kurtosis)
- Total: 63 × 7 = 441 features per clip

### 2.3 Algorithms

#### 2.3.1 Random Forest

| Parameter | Search Range | Rationale |
|-----------|-------------|-----------|
| `n_estimators` | [100, 200, 500] | More trees reduce variance but with diminishing returns |
| `max_depth` | [None, 10, 20, 30] | None allows full growth; shallower trees reduce overfitting |
| `min_samples_split` | [2, 5, 10] | Higher values prevent learning noise |
| `min_samples_leaf` | [1, 2, 4] | Higher values provide regularization |
| `max_features` | ['sqrt', 'log2', 0.3] | Controls feature subsampling per tree |
| `class_weight` | ['balanced', 'balanced_subsample'] | Addresses class imbalance |

**Feature selection:** After fitting, extract the top-50 features by importance. Retrain on the reduced feature set to check if accuracy improves (removing noisy features often helps with small datasets).

#### 2.3.2 Support Vector Machine (SVM)

| Parameter | Search Range | Rationale |
|-----------|-------------|-----------|
| `kernel` | ['rbf', 'linear'] | RBF is the default for non-linear problems; linear is simpler |
| `C` | [0.01, 0.1, 1, 10, 100] | Regularization parameter — lower C = more regularization |
| `gamma` | ['scale', 'auto', 0.001, 0.01, 0.1] | RBF kernel width — 'scale' is usually a good default |
| `class_weight` | ['balanced'] | Addresses class imbalance |

**Note:** The feature vectors must be standardized (zero mean, unit variance) before SVM training. Use `sklearn.preprocessing.StandardScaler` fitted on the training set.

### 2.4 Cross-Validation

For classical ML models, use **5-fold stratified cross-validation** on the training set for hyperparameter tuning, then retrain on the full training set with the best hyperparameters and evaluate on the held-out test set.

---

## 3. Model M2/M3/M4 — Compact 2-D CNN (Mel-Spectrogram)

This is the **primary architecture** for the project. The float32 version (M2) establishes the accuracy ceiling; the int8-PTQ version (M3) and int8-QAT version (M4) are the deployment targets.

### 3.1 Architecture

```
Input: Log-mel spectrogram [40 × 92 × 1]
                │
                ▼
        Conv2D(16 filters, 3×3, padding='same')
        BatchNormalization
        ReLU
        MaxPool2D(2×2)
        ─── Output: [20 × 46 × 16] ───
                │
                ▼
        Conv2D(32 filters, 3×3, padding='same')
        BatchNormalization
        ReLU
        MaxPool2D(2×2)
        ─── Output: [10 × 23 × 32] ───
                │
                ▼
        Conv2D(32 filters, 3×3, padding='same')
        BatchNormalization
        ReLU
        GlobalAveragePooling2D
        ─── Output: [32] ───
                │
                ▼
        Dropout(0.3)
        Dense(N_classes)
        Softmax
        ─── Output: [N_classes] ───
```

### 3.2 Layer-by-Layer Analysis

| Layer | Output Shape | Parameters | Notes |
|-------|-------------|-----------|-------|
| Input | (40, 92, 1) | 0 | Log-mel spectrogram |
| Conv2D(16, 3×3, same) | (40, 92, 16) | 16×(3×3×1 + 1) = 160 | First conv captures local time-frequency patterns |
| BatchNorm | (40, 92, 16) | 64 | 4 params per channel (γ, β, moving_mean, moving_var) |
| ReLU | (40, 92, 16) | 0 | |
| MaxPool2D(2×2) | (20, 46, 16) | 0 | Reduces spatial dimensions by 2× |
| Conv2D(32, 3×3, same) | (20, 46, 32) | 32×(3×3×16 + 1) = 4,640 | Learns higher-level spectro-temporal patterns |
| BatchNorm | (20, 46, 32) | 128 | |
| ReLU | (20, 46, 32) | 0 | |
| MaxPool2D(2×2) | (10, 23, 32) | 0 | |
| Conv2D(32, 3×3, same) | (10, 23, 32) | 32×(3×3×32 + 1) = 9,248 | Deepest feature extraction |
| BatchNorm | (10, 23, 32) | 128 | |
| ReLU | (10, 23, 32) | 0 | |
| GlobalAvgPool2D | (32,) | 0 | Collapses spatial dims; more robust than flatten+dense |
| Dropout(0.3) | (32,) | 0 | Regularization |
| Dense(N) | (N,) | 32×N + N = 33N | N = 2, 6, or 9 depending on tier |
| **Total** | — | **~14,400 + 33N** | Tier 2 (N=6): **~14,600 params** |

### 3.3 Model Size Estimates

| Format | Size | Fits in Flash Budget? |
|--------|------|----------------------|
| Float32 (.h5) | ~57 KB | Marginal (at the edge of 150 KB budget with runtime overhead) |
| Int8 PTQ (.tflite) | ~15–18 KB | **Yes, very comfortable** |
| Int8 QAT (.tflite) | ~15–18 KB | **Yes, very comfortable** |

At only ~14,600 parameters, this model is quite small. The int8 version uses ~1 byte per parameter, yielding a model of roughly 15 KB — well within the 150 KB flash budget. This leaves room for a larger model if accuracy is insufficient.

### 3.4 Tensor Arena (SRAM) Estimate

The largest intermediate activation determines the peak SRAM usage (the "tensor arena" size). TFLite Micro allocates a contiguous buffer to hold the current layer's input and output simultaneously:

- After first Conv2D + BN + ReLU: 40 × 92 × 16 = 58,880 values → 58,880 bytes (int8)
- After first MaxPool: 20 × 46 × 16 = 14,720 bytes
- After second Conv2D + BN + ReLU: 20 × 46 × 32 = 29,440 bytes

Peak is at the first Conv2D output: the arena must hold both the input (40×92×1 = 3,680 bytes) and output (58,880 bytes) simultaneously, plus scratch buffers. Estimated tensor arena: **~70–80 KB**.

This is at the upper end of our 50–100 KB SRAM budget. If it doesn't fit:
- Reduce `n_mels` from 40 to 32: first Conv output drops to 32×92×16 = 47,104 bytes
- Reduce first Conv filters from 16 to 8: first Conv output drops to 40×92×8 = 29,440 bytes

### 3.5 Keras Implementation

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_compact_cnn(input_shape=(40, 92, 1), num_classes=6):
    """Build the compact 2-D CNN for mel-spectrogram classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(16, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D((2, 2)),

        # Block 2
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D((2, 2)),

        # Block 3
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),

        # Classifier
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model
```

### 3.6 Variant: Deeper CNN (If Accuracy Is Insufficient)

If the 3-block CNN under-fits (training accuracy plateaus below acceptable levels), add a 4th convolutional block:

```
        Conv2D(64 filters, 3×3, padding='same')
        BatchNormalization
        ReLU
        GlobalAveragePooling2D
```

This adds ~18,500 parameters (total ~33,000), and the int8 model grows to ~35 KB. Still well within the flash budget, but the tensor arena grows. Use this variant only if the 3-block model's accuracy is clearly insufficient.

---

## 4. Model M5 — 1-D CNN (MFCC)

### 4.1 Architecture

```
Input: MFCC frames [92 × 13]
            │
            ▼
    Conv1D(16 filters, kernel_size=5, padding='same')
    BatchNormalization
    ReLU
    MaxPool1D(pool_size=2)
    ─── Output: [46 × 16] ───
            │
            ▼
    Conv1D(32 filters, kernel_size=3, padding='same')
    BatchNormalization
    ReLU
    MaxPool1D(pool_size=2)
    ─── Output: [23 × 32] ───
            │
            ▼
    Conv1D(32 filters, kernel_size=3, padding='same')
    BatchNormalization
    ReLU
    GlobalAveragePooling1D
    ─── Output: [32] ───
            │
            ▼
    Dropout(0.3)
    Dense(N_classes)
    Softmax
    ─── Output: [N_classes] ───
```

### 4.2 Layer-by-Layer Analysis

| Layer | Output Shape | Parameters | Notes |
|-------|-------------|-----------|-------|
| Input | (92, 13) | 0 | MFCC frames (time × coefficients) |
| Conv1D(16, 5, same) | (92, 16) | 16×(5×13 + 1) = 1,056 | Wider kernel (5) for temporal context |
| BatchNorm | (92, 16) | 64 | |
| ReLU + MaxPool1D(2) | (46, 16) | 0 | |
| Conv1D(32, 3, same) | (46, 32) | 32×(3×16 + 1) = 1,568 | |
| BatchNorm | (46, 32) | 128 | |
| ReLU + MaxPool1D(2) | (23, 32) | 0 | |
| Conv1D(32, 3, same) | (23, 32) | 32×(3×32 + 1) = 3,104 | |
| BatchNorm | (23, 32) | 128 | |
| ReLU + GlobalAvgPool1D | (32,) | 0 | |
| Dropout(0.3) + Dense(N) | (N,) | 33N | |
| **Total** | — | **~6,050 + 33N** | Tier 2 (N=6): **~6,250 params** |

### 4.3 Model Size Estimates

| Format | Size |
|--------|------|
| Float32 | ~25 KB |
| Int8 | ~7–9 KB |

This is extremely compact — roughly half the size of the 2-D CNN. If accuracy is competitive, this is the ideal deployment candidate.

### 4.4 Tensor Arena Estimate

- Peak activation: Conv1D output at 92×16 = 1,472 values (int8) — much smaller than the 2-D CNN
- Estimated tensor arena: **~15–25 KB**

This leaves generous SRAM headroom, which could be used for a larger audio buffer, double-buffering, or a sliding window approach.

### 4.5 Trade-offs vs. 2-D CNN

| Aspect | 1-D CNN (MFCC) | 2-D CNN (Mel-Spectrogram) |
|--------|----------------|--------------------------|
| Model size (int8) | ~8 KB | ~16 KB |
| Tensor arena | ~20 KB | ~75 KB |
| Feature richness | Lower (MFCCs are a compressed representation — the DCT discards fine spectral detail) | Higher (mel-spectrogram retains full spectral envelope) |
| On-device feature cost | Higher (requires FFT + mel filterbank + log + DCT) | Lower (requires FFT + mel filterbank + log — no DCT) |
| Expected accuracy | Slightly lower | Slightly higher |

### 4.6 Keras Implementation

```python
def build_1d_cnn(input_shape=(92, 13), num_classes=6):
    """Build the 1-D CNN for MFCC classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv1D(16, 5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool1D(2),

        # Block 2
        layers.Conv1D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool1D(2),

        # Block 3
        layers.Conv1D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling1D(),

        # Classifier
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model
```

---

## 5. Model M6 — Depthwise Separable CNN (DS-CNN)

### 5.1 Rationale

Depthwise separable convolutions factorize a standard convolution into a **depthwise convolution** (applies a single filter per input channel) and a **pointwise convolution** (1×1 convolution to mix channels). This reduces parameters and computation by a factor of roughly `1/k² + 1/C_out` compared to standard convolution (where k is kernel size and C_out is output channels). DS-CNNs are the architecture behind Google's "micro_speech" keyword spotting model and are well-supported by CMSIS-NN.

### 5.2 Architecture

This follows the DS-CNN-S (small) variant from the ARM ML Zoo, adapted for our input dimensions:

```
Input: Log-mel spectrogram [40 × 92 × 1]
                │
                ▼
        Conv2D(64, 4×10, strides=(2, 2), padding='same')
        BatchNormalization
        ReLU
        ─── Output: [20 × 46 × 64] ───
                │
                ▼
        DepthwiseConv2D(3×3, padding='same')
        BatchNormalization
        ReLU
        Conv2D(64, 1×1)       ← pointwise
        BatchNormalization
        ReLU
        ─── Output: [20 × 46 × 64] ───
                │
                ▼
        (Repeat DS block 3 more times, same configuration)
        ─── Output: [20 × 46 × 64] ───
                │
                ▼
        AveragePooling2D(pool_size=(20, 46))    ← global average pool
        Flatten
        ─── Output: [64] ───
                │
                ▼
        Dense(N_classes)
        Softmax
        ─── Output: [N_classes] ───
```

### 5.3 Parameter Count

| Layer | Parameters |
|-------|-----------|
| Initial Conv2D(64, 4×10×1) | 64 × (4×10×1 + 1) = 2,624 |
| BatchNorm | 256 |
| DS Block × 4: DepthwiseConv2D(3×3) | 64 × (3×3 + 1) × 4 = 2,560 |
| DS Block × 4: Conv2D(64, 1×1) (pointwise) | 64 × (64 + 1) × 4 = 16,640 |
| DS Block × 4: BatchNorm × 2 | 256 × 2 × 4 = 2,048 |
| Dense(N) | 64×N + N = 65N |
| **Total** | **~24,100 + 65N** |

For Tier 2 (N=6): **~24,500 parameters**

### 5.4 Model Size Estimates

| Format | Size |
|--------|------|
| Float32 | ~98 KB |
| Int8 | ~26–30 KB |

The int8 model fits comfortably in the flash budget.

### 5.5 Tensor Arena Estimate

The first Conv2D output is the peak: 20 × 46 × 64 = 58,880 bytes (int8). However, the DS blocks maintain this spatial dimension (no pooling until the end), so the arena must hold 58,880 bytes for input + 58,880 bytes for output of each DS block simultaneously. Estimated tensor arena: **~120–140 KB**.

**This exceeds our 80 KB SRAM budget.** Mitigations:

1. **Reduce channels:** Use 32 channels instead of 64 throughout. This reduces the tensor arena by ~4× (since arena scales with channels²) and parameters by ~4×. Estimated arena: ~35–40 KB. Parameters: ~6,500. This makes the DS-CNN comparable in size to the 1-D CNN but with potentially better accuracy.
2. **Add pooling between DS blocks:** Insert MaxPool2D(2×2) after the second DS block to reduce spatial dimensions.
3. **Use Option 1 (reduced channels) as the primary DS-CNN variant.**

### 5.6 Revised DS-CNN Architecture (32 channels, fits in SRAM)

```
Input: Log-mel spectrogram [40 × 92 × 1]
                │
                ▼
        Conv2D(32, 4×10, strides=(2, 2), padding='same')
        BatchNormalization
        ReLU
        ─── Output: [20 × 46 × 32] ───
                │
                ▼
        4 × DS Block:
            DepthwiseConv2D(3×3, padding='same')
            BatchNormalization
            ReLU
            Conv2D(32, 1×1)
            BatchNormalization
            ReLU
        ─── Output: [20 × 46 × 32] ───
                │
                ▼
        GlobalAveragePooling2D
        Dense(N_classes)
        Softmax
```

**Parameters:** ~6,700 (Tier 2). **Int8 size:** ~8–10 KB. **Arena:** ~35–45 KB. **Fits within budget.**

### 5.7 Keras Implementation (Revised)

```python
def build_ds_cnn(input_shape=(40, 92, 1), num_classes=6, channels=32, num_ds_blocks=4):
    """Build a Depthwise Separable CNN for mel-spectrogram classification."""
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(channels, (4, 10), strides=(2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # DS blocks
    for _ in range(num_ds_blocks):
        x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(channels, (1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Classifier
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)
```

---

## 6. Training Configuration (Common to All Neural Networks)

### 6.1 Optimizer

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rate, widely used default for small models |
| **Initial learning rate** | 0.001 | Standard Adam default; will be reduced by scheduler |
| **β₁, β₂** | 0.9, 0.999 | Standard Adam defaults |
| **ε** | 1e-7 | Standard |

### 6.2 Learning Rate Schedule

Use a **ReduceLROnPlateau** callback to reduce the learning rate when validation loss plateaus:

| Parameter | Value |
|-----------|-------|
| `monitor` | `val_loss` |
| `factor` | 0.5 (halve the LR) |
| `patience` | 5 epochs |
| `min_lr` | 1e-6 |

Additionally, consider a **cosine annealing** schedule as an alternative:
- Decays the learning rate from `lr_max` to `lr_min` following a cosine curve over the total number of epochs
- No hyperparameter tuning needed beyond `lr_max`, `lr_min`, and epoch count
- Often produces smoother convergence than step-based schedules

### 6.3 Epochs and Early Stopping

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Max epochs** | 100 | Upper bound; training will likely converge much earlier |
| **Early stopping patience** | 15 epochs | Stop if validation loss hasn't improved for 15 epochs |
| **Early stopping restore** | Best weights | Restore the model weights from the epoch with lowest validation loss |

With only ~970 training samples (before augmentation), neural networks on this task should converge within 30–60 epochs. The 100-epoch ceiling with early stopping prevents both under-training and unnecessary compute.

### 6.4 Batch Size

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch size** | 32 | With ~970 training samples, batch size 32 gives ~30 gradient updates per epoch. This is a good balance: small enough for noisy (regularizing) gradients, large enough for stable training. If augmentation expands the effective dataset 5×, we get ~150 updates/epoch. |

### 6.5 Model Checkpointing

Save the model weights at the epoch with the best validation metric:

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='models/{model_id}_tier{tier}_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max'
)
```

Also save at the end of training (regardless of whether it's the best) for analysis:

```python
# After training:
model.save(f'models/{model_id}_tier{tier}_final.h5')
```

---

## 7. Hyperparameter Search Strategy

### 7.1 Approach: Structured Manual Search

Given the small dataset and limited model complexity, a full automated hyperparameter search (e.g., Bayesian optimization) is overkill. Instead, we use a **structured manual search** approach:

1. **Start with the default configuration** described in this document
2. **Vary one factor at a time** from the defaults, evaluating on the validation set
3. **Record all results** in a structured log for the final report

### 7.2 Hyperparameters to Tune (Neural Networks)

| Hyperparameter | Default | Search Values | Priority |
|---------------|---------|--------------|----------|
| Learning rate | 0.001 | [0.0005, 0.001, 0.002] | High |
| Batch size | 32 | [16, 32, 64] | Medium |
| Dropout rate | 0.3 | [0.2, 0.3, 0.5] | Medium |
| Conv filters (Block 1) | 16 | [8, 16, 24] | High (affects model size and SRAM) |
| Conv filters (Block 2/3) | 32 | [16, 32, 48] | High (affects model size and SRAM) |
| Number of conv blocks | 3 | [2, 3, 4] | Medium |
| Augmentation intensity | Default table | [Reduced, Default, Aggressive] | High |

### 7.3 Hyperparameters to Tune (Classical ML)

See Section 2.3 for the full parameter grids. Use `sklearn.model_selection.GridSearchCV` with 5-fold stratified CV.

### 7.4 Search Budget

To keep the search tractable, limit to a maximum of **~50 training runs total** across all models and tiers. Each training run on this small dataset should take 1–5 minutes on a modern GPU (or 10–30 minutes on CPU), so the total search time is manageable.

---

## 8. Loss Functions and Class Weighting

### 8.1 Loss Function: Categorical Cross-Entropy

Standard categorical (sparse) cross-entropy is used for all multi-class models:

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

For the binary (Tier 1) case, binary cross-entropy could be used, but sparse categorical cross-entropy with 2 classes is functionally equivalent and keeps the code uniform across tiers.

### 8.2 Class Weighting

As discussed in [data_preprocessing_spec.md](data_preprocessing_spec.md) Section 10.4, class weights are computed from the training set and passed to `model.fit()`:

```python
model.fit(
    X_train, y_train,
    class_weight=class_weight_dict,
    ...
)
```

This upweights the loss contribution of minority classes so the model doesn't learn to always predict the majority class.

### 8.3 Alternative: Focal Loss

If class weighting alone doesn't sufficiently address the imbalance (i.e., the model still struggles on minority classes), consider **focal loss**:

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```

Where γ (focusing parameter, typically 2.0) down-weights easy/well-classified examples and focuses learning on hard examples. This is more aggressive than class weighting and can help with severely imbalanced datasets.

---

## 9. Regularization Strategy

With only ~970 training samples, regularization is critical to prevent overfitting. The following techniques are employed:

### 9.1 Dropout

Applied after the GlobalAveragePooling layer and before the final Dense layer. Dropout rate of 0.3 is the default, tunable in the range [0.2, 0.5].

### 9.2 Batch Normalization

Applied after every Conv layer. Batch normalization provides implicit regularization by introducing noise during training (the batch statistics vary from batch to batch). It also stabilizes training and allows higher learning rates.

### 9.3 Global Average Pooling (vs. Flatten)

Global average pooling reduces each feature map to a single value by averaging over all spatial positions. Compared to flattening + dense layers, this:
- Drastically reduces the number of parameters (no dense layer connecting spatial features to the classifier)
- Acts as a structural regularizer by forcing each feature map to represent a single "concept"
- Makes the model more robust to spatial translations in the spectrogram

### 9.4 Data Augmentation

The most important regularizer for small datasets. See [data_preprocessing_spec.md](data_preprocessing_spec.md) Section 8 for the full augmentation pipeline.

### 9.5 Early Stopping

Prevents overfitting by stopping training when the validation loss stops improving. The patience (15 epochs) is set high enough to allow the learning rate scheduler to take effect before stopping.

### 9.6 Weight Decay (L2 Regularization)

If overfitting persists despite the above measures, add L2 regularization to Conv and Dense layers:

```python
layers.Conv2D(16, (3, 3), padding='same',
              kernel_regularizer=keras.regularizers.l2(1e-4))
```

Start with λ = 1e-4. This is a secondary measure — augmentation and dropout should be the primary defenses.

---

## 10. Model Size Estimation and Memory Feasibility

### 10.1 Summary Table

| Model | Params | Float32 Size | Int8 Size | Tensor Arena (int8) | Fits Flash? | Fits SRAM? |
|-------|--------|-------------|-----------|--------------------|-----------|-----------|
| M2/M3/M4 (2-D CNN, Tier 2) | ~14,600 | ~57 KB | ~16 KB | ~70–80 KB | Yes | Tight (may need reduced n_mels) |
| M5 (1-D CNN, Tier 2) | ~6,250 | ~25 KB | ~8 KB | ~20 KB | Yes | Yes (generous headroom) |
| M6 (DS-CNN 32ch, Tier 2) | ~6,700 | ~27 KB | ~9 KB | ~35–45 KB | Yes | Yes |

### 10.2 Deployment Priority

Based on the feasibility analysis:

1. **M5 (1-D CNN)** — Most likely to fit comfortably. Smallest tensor arena. Deploy this first to validate the on-device pipeline.
2. **M6 (DS-CNN, 32 channels)** — Second choice. Slightly larger arena but potentially better accuracy.
3. **M3/M4 (2-D CNN)** — Third choice. The tensor arena is tight. If it fits, this is likely the most accurate option. If it doesn't fit, reduce `n_mels` from 40 to 32.

### 10.3 TFLite Model Size Profiling

After conversion, use `tf.lite.Interpreter` to profile the model:

```python
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get tensor arena size
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")

# Model file size
import os
model_size_kb = os.path.getsize('model.tflite') / 1024
print(f"Model size: {model_size_kb:.1f} KB")
```

For accurate tensor arena size on Cortex-M4, use the TFLite Micro `RecordingMicroAllocator` or compile the Arduino sketch and check the linker's SRAM usage report.

---

## 11. Training Infrastructure

### 11.1 Hardware Requirements

Training these small models does not require significant compute:
- **GPU:** Any modern GPU (e.g., NVIDIA GTX 1060 or better) will train a model in 1–5 minutes
- **CPU-only:** Feasible but slower (10–30 minutes per training run). Acceptable given the model size.
- **Cloud:** Google Colab (free tier) provides sufficient GPU time for all training runs

### 11.2 Software Environment

```
Python >= 3.10
TensorFlow >= 2.15.0 (or 2.16+ for latest TFLite converter features)
tensorflow-model-optimization >= 0.8.0  (for QAT)
scikit-learn >= 1.3.0
librosa >= 0.10.0
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
```

### 11.3 Experiment Tracking

All training runs should be logged with:
- Hyperparameters used
- Training/validation loss and accuracy curves
- Best epoch number
- Test set metrics (computed ONCE at the end, not during tuning)
- Model file sizes (float32 and int8)

Use either:
- **MLflow** (local tracking server) — recommended for structured experiment management
- **TensorBoard** — for real-time training curve visualization
- **CSV log** — minimum viable approach; record all metrics in a structured spreadsheet

### 11.4 Model Artifact Organization

```
models/
├── m1_classical/
│   ├── rf_tier1.joblib
│   ├── rf_tier2.joblib
│   ├── rf_tier3.joblib
│   ├── svm_tier1.joblib
│   ├── svm_tier2.joblib
│   └── svm_tier3.joblib
├── m2_cnn2d_float32/
│   ├── tier1_best.h5
│   ├── tier2_best.h5
│   └── tier3_best.h5
├── m3_cnn2d_int8_ptq/
│   ├── tier1.tflite
│   ├── tier2.tflite
│   └── tier3.tflite
├── m4_cnn2d_int8_qat/
│   ├── tier1.tflite
│   ├── tier2.tflite
│   └── tier3.tflite
├── m5_cnn1d_int8_qat/
│   ├── tier1.tflite
│   ├── tier2.tflite
│   └── tier3.tflite
├── m6_dscnn_int8_qat/
│   ├── tier1.tflite
│   ├── tier2.tflite
│   └── tier3.tflite
└── experiment_log.csv
```
