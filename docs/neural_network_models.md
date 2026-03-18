# Neural Network Models — Model Documentation

**Phase:** 3 (Neural Network Training)
**Date:** March 18, 2026
**Related code:** `src/models.py`, `src/train_nn.py`, `src/evaluate.py`, `notebooks/03_neural_networks.ipynb`

---

## 1. Purpose

Phase 3 trains three neural network architectures across all three taxonomy tiers, producing 9 float32 models. These models serve two purposes:

1. **Accuracy comparison:** Determine whether learned feature representations (via convolutional neural networks) outperform hand-crafted classical features (Phase 2 baselines).
2. **Quantization candidates:** The float32 models are the starting point for Phase 4 (int8 quantization via PTQ and QAT), which produces the models that will be deployed on the Arduino Nano 33 BLE Sense Rev2.

---

## 2. Model Matrix

| Model ID | Architecture | Input | Params (Tier 2) | Float32 Size |
|----------|-------------|-------|-----------------|-------------|
| M2 | Compact 2-D CNN | Log-mel spectrogram (40×92×1) | 14,566 | ~237 KB |
| M5 | 1-D CNN | MFCC frames (92×13) | 6,246 | ~140 KB |
| M6 | DS-CNN (32-channel) | Log-mel spectrogram (40×92×1) | 8,166 | ~246 KB |

Each architecture was trained on all 3 tiers: Tier 1 (2-class), Tier 2 (6-class), Tier 3 (9-class).

---

## 3. Input Features

### 3.1 Log-Mel Spectrogram (M2 and M6 input)

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 kHz |
| Duration | 1.5 s (24,000 samples) |
| n_fft | 512 |
| hop_length | 256 |
| n_mels | 40 |
| fmin | 20 Hz |
| fmax | 8000 Hz |
| center | False |
| Output shape | (40, 92, 1) |
| Normalization | Per-mel-band z-score (train-set statistics) |

### 3.2 MFCCs (M5 input)

| Parameter | Value |
|-----------|-------|
| n_mfcc | 13 coefficients |
| Same FFT/hop/mel settings as above | |
| Output shape | (92, 13) |
| Normalization | Per-coefficient z-score (train-set statistics) |

Features were pre-computed in Phase 1 and stored un-normalized. Z-score normalization was applied at training time using statistics from `data/normalization_stats.npz` (computed from the training set only).

---

## 4. Architecture Details

### 4.1 M2 — Compact 2-D CNN

The primary architecture for the project. A 3-block 2-D CNN on mel-spectrograms, designed for compactness while maintaining sufficient capacity for audio classification.

```
Input: (40, 92, 1)
    │
    Conv2D(16, 3×3, padding=same) → BatchNorm → ReLU → MaxPool2D(2×2)
    Output: (20, 46, 16)
    │
    Conv2D(32, 3×3, padding=same) → BatchNorm → ReLU → MaxPool2D(2×2)
    Output: (10, 23, 32)
    │
    Conv2D(32, 3×3, padding=same) → BatchNorm → ReLU → GlobalAvgPool2D
    Output: (32,)
    │
    Dropout(0.3) → Dense(N, softmax)
    Output: (N,)
```

| Tier | Output classes | Total parameters |
|------|---------------|-----------------|
| 1 | 2 | 14,434 |
| 2 | 6 | 14,566 |
| 3 | 9 | 14,665 |

### 4.2 M5 — 1-D CNN

A lighter alternative operating on MFCC features. Uses 1-D convolutions along the time axis, making it the smallest model and easiest to deploy.

```
Input: (92, 13)
    │
    Conv1D(16, kernel=5, padding=same) → BatchNorm → ReLU → MaxPool1D(2)
    Output: (46, 16)
    │
    Conv1D(32, kernel=3, padding=same) → BatchNorm → ReLU → MaxPool1D(2)
    Output: (23, 32)
    │
    Conv1D(32, kernel=3, padding=same) → BatchNorm → ReLU → GlobalAvgPool1D
    Output: (32,)
    │
    Dropout(0.2) → Dense(N, softmax)
    Output: (N,)
```

| Tier | Output classes | Total parameters |
|------|---------------|-----------------|
| 1 | 2 | 6,114 |
| 2 | 6 | 6,246 |
| 3 | 9 | 6,345 |

Note: Dropout rate was tuned to 0.2 (vs. default 0.3) based on hyperparameter search.

### 4.3 M6 — Depthwise Separable CNN (32-channel)

Follows the DS-CNN-S architecture from the ARM ML Zoo, adapted for our input dimensions. Uses the 32-channel variant (revised from the original 64-channel spec) to fit within the 80 KB SRAM tensor arena budget.

```
Input: (40, 92, 1)
    │
    Conv2D(32, 4×10, stride=(2,2), padding=same) → BatchNorm → ReLU
    Output: (20, 46, 32)
    │
    4 × DS Block:
        DepthwiseConv2D(3×3, padding=same) → BatchNorm → ReLU
        Conv2D(32, 1×1) → BatchNorm → ReLU
    Output: (20, 46, 32)
    │
    GlobalAvgPool2D → Dense(N, softmax)
    Output: (N,)
```

| Tier | Output classes | Total parameters |
|------|---------------|-----------------|
| 1 | 2 | 8,034 |
| 2 | 6 | 8,166 |
| 3 | 9 | 8,265 |

---

## 5. Training Configuration

### 5.1 Common Settings (all architectures)

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6) |
| Max epochs | 100 |
| Early stopping | patience=15, restore_best_weights=True, monitor=val_loss |
| Batch size | 32 |
| Loss function | sparse_categorical_crossentropy |
| Class weighting | sklearn compute_class_weight('balanced') |
| Checkpoint metric | val_accuracy (save best) |
| Random state | 42 |

### 5.2 Per-Architecture Settings (from hyperparameter search)

| Parameter | M2 (2-D CNN) | M5 (1-D CNN) | M6 (DS-CNN) |
|-----------|-------------|-------------|-------------|
| Learning rate | 0.002 | 0.002 | 0.0005 |
| Dropout rate | 0.3 | 0.2 | N/A (no dropout layer) |
| Augmentation | Aggressive SpecAugment | None | Default SpecAugment |

### 5.3 Data Augmentation: SpecAugment

SpecAugment was applied on-the-fly during training (50% probability per sample per epoch) using the pre-computed features. No waveform-level augmentation was used.

| Setting | Mel-spectrogram | MFCC |
|---------|----------------|------|
| Default | F=7 (freq mask), T=10 (time mask) | F=2, T=10 |
| Aggressive | F=10, T=15 | F=3, T=15 |
| Application axis | Frequency: mel bands (40), Time: frames (92) | Frequency: coefficients (13), Time: frames (92) |

### 5.4 Data Splits

| Split | Tier 1 & 2 | Tier 3 |
|-------|-----------|--------|
| Train | 970 samples | 664 samples |
| Val (for early stopping) | 208 samples | 142 samples |
| Test (held out) | 208 samples | 143 samples |

Unlike Phase 2 (classical ML), train and val sets were kept separate — val was used for early stopping, learning rate scheduling, and model checkpointing.

---

## 6. Hyperparameter Search

A structured manual search was conducted on Tier 2 (primary target) for all three architectures. Three sweeps were performed sequentially: learning rate → dropout → augmentation intensity.

### 6.1 M2 (2-D CNN) Search Results

| Sweep | Setting | Best Val Accuracy |
|-------|---------|------------------|
| **LR** | 0.0005 → 0.750, **0.002 → 0.793**, 0.001 → 0.774 | LR=0.002 |
| **Dropout** | 0.2 → 0.779, **0.3 → 0.803**, 0.5 → 0.731 | Dropout=0.3 |
| **Augmentation** | none → 0.813, default → 0.760, **aggressive → 0.827** | Aggressive |

### 6.2 M5 (1-D CNN) Search Results

| Sweep | Setting | Best Val Accuracy |
|-------|---------|------------------|
| **LR** | 0.0005 → 0.712, 0.001 → 0.764, **0.002 → 0.813** | LR=0.002 |
| **Dropout** | **0.2 → 0.865**, 0.3 → 0.817, 0.5 → 0.784 | Dropout=0.2 |
| **Augmentation** | **none → 0.851**, default → 0.793, aggressive → 0.817 | None |

### 6.3 M6 (DS-CNN) Search Results

| Sweep | Setting | Best Val Accuracy |
|-------|---------|------------------|
| **LR** | **0.0005 → 0.736**, 0.001 → 0.231, 0.002 → 0.260 | LR=0.0005 |
| **Augmentation** | none → 0.716, **default → 0.784**, aggressive → 0.188 | Default |

**Key finding:** M6 was highly sensitive to learning rate — LR ≥ 0.001 caused training instability (val accuracy ~0.23 = near random for 6 classes). The DS-CNN's depthwise convolutions have fewer parameters per layer, making them more susceptible to large learning rates on small datasets.

---

## 7. Results

### 7.1 Summary Table

| Model | Tier | Accuracy | F1 Macro | F1 Weighted | Epochs | Time (s) |
|-------|------|----------|----------|-------------|--------|----------|
| M2 (2-D CNN) | 1 | 0.8846 | 0.8679 | 0.8873 | 60 | 23.5 |
| M2 (2-D CNN) | 2 | 0.7837 | 0.6658 | 0.7965 | 70 | 25.5 |
| M2 (2-D CNN) | 3 | 0.7343 | 0.7150 | 0.7443 | 100 | 34.8 |
| M5 (1-D CNN) | 1 | 0.8942 | 0.8748 | 0.8952 | 43 | 13.8 |
| M5 (1-D CNN) | 2 | 0.7692 | 0.6164 | 0.7821 | 99 | 31.5 |
| M5 (1-D CNN) | 3 | 0.6783 | 0.6163 | 0.6836 | 39 | 15.5 |
| M6 (DS-CNN) | 1 | 0.8750 | 0.8580 | 0.8783 | 52 | 38.2 |
| M6 (DS-CNN) | 2 | 0.6538 | 0.5706 | 0.6815 | 93 | 51.5 |
| M6 (DS-CNN) | 3 | 0.6224 | 0.5997 | 0.6264 | 100 | 55.0 |

### 7.2 Comparison with Classical ML Baselines

| Tier | Best Classical ML | Best NN | NN Beats Baseline? |
|------|------------------|---------|--------------------|
| 1 | SVM (F1=0.916) | M5 (F1=0.875) | No (−0.041) |
| 2 | SVM (F1=0.699) | M2 (F1=0.666) | No (−0.033) |
| 3 | SVM (F1=0.711) | M2 (F1=0.715) | Yes (+0.004) |

The neural networks did not exceed classical ML baselines on Tiers 1–2 but marginally exceeded on Tier 3. This is consistent with expectations for a small dataset (970 training samples) where hand-crafted features can capture the most discriminative information without the need for learned representations.

### 7.3 Per-Class Results — Tier 2 (M2 2-D CNN, best NN)

| Class | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Normal Braking | 12 | 0.5882 | 0.8333 | 0.6897 |
| Braking Fault | 11 | 0.5333 | 0.7273 | 0.6154 |
| Normal Idle | 40 | 0.7778 | 0.8750 | 0.8235 |
| Idle Fault | 118 | 0.9596 | 0.8051 | 0.8756 |
| Normal Start-Up | 9 | 0.2941 | 0.5556 | 0.3846 |
| Start-Up Fault | 18 | 0.6667 | 0.5556 | 0.6061 |

**"Normal Start-Up" improved** from 11-22% recall (classical ML) to 55.6% recall (M2 NN). The NN learned to better distinguish this minority class, though at the cost of increased false positives (29.4% precision). The aggressive SpecAugment and class weighting helped surface minority-class patterns.

### 7.4 Per-Class Results — Tier 3 (M2 2-D CNN)

| Class | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Normal Braking | 12 | 0.5556 | 0.8333 | 0.6667 |
| Worn Brakes | 11 | 0.5556 | 0.4545 | 0.5000 |
| Normal Idle | 40 | 0.8684 | 0.8250 | 0.8462 |
| Low Oil | 16 | 0.7500 | 0.5625 | 0.6429 |
| Power Steering Fault | 19 | 0.6364 | 0.7368 | 0.6829 |
| Serpentine Belt Fault | 18 | 0.8421 | 0.8889 | 0.8649 |
| Normal Start-Up | 9 | 0.5000 | 0.5556 | 0.5263 |
| Bad Ignition | 9 | 0.6000 | 0.6667 | 0.6316 |
| Dead Battery | 9 | 0.6250 | 0.5556 | 0.5882 |

---

## 8. Architecture Comparison

### 8.1 M2 vs M5 vs M6

| Aspect | M2 (2-D CNN) | M5 (1-D CNN) | M6 (DS-CNN) |
|--------|-------------|-------------|-------------|
| Best F1 macro (Tier 2) | **0.666** | 0.616 | 0.571 |
| Parameters (Tier 2) | 14,566 | **6,246** | 8,166 |
| Float32 model size | 237 KB | **140 KB** | 246 KB |
| Est. int8 model size | ~16 KB | **~8 KB** | ~9 KB |
| Est. tensor arena | ~70-80 KB | **~15-25 KB** | ~35-45 KB |
| Training stability | Stable | Stable | **Unstable at LR ≥ 0.001** |
| Convergence speed | ~60-70 epochs | ~40-100 epochs | ~50-100 epochs |

### 8.2 Deployment Recommendations

1. **M5 (1-D CNN) — Safest deployment candidate.** Smallest model (~8 KB int8) and tensor arena (~20 KB), leaving generous SRAM headroom. Accuracy is acceptable though not the highest.
2. **M2 (2-D CNN) — Highest accuracy candidate.** Best F1 macro across all tiers. Tensor arena (~75 KB) is tight but within the 80 KB budget. If it fits, it is the best performer.
3. **M6 (DS-CNN) — Not recommended.** Lowest accuracy, training instability, and no clear advantage over M2 or M5. The DS-CNN architecture may be better suited to larger datasets.

---

## 9. Discussion

### 9.1 Why Classical ML Outperforms NNs on Tiers 1–2

With only 970 training samples, the classical ML models have an advantage: their 441-dimensional hand-crafted feature vectors are computed using domain-specific signal processing (MFCCs, spectral features, chroma) with 7 aggregation statistics each. These features are highly informative and well-matched to the task. The NNs must learn equivalent feature extractors from raw spectrograms with a similar number of parameters — a harder task with limited data.

Additionally, the classical ML models used merged train+val sets (1,178 samples) with 5-fold CV, while the NNs used only the 970 training samples with the val set reserved for early stopping.

### 9.2 SpecAugment Effect

The augmentation sweep results were mixed:
- **M2 benefited from aggressive SpecAugment** (F=10, T=15), suggesting the 2-D CNN can leverage spectrogram masking to regularize.
- **M5 performed best without augmentation.** Masking MFCC coefficients (only 13 dimensions) may destroy too much information, even with F=2.
- **M6 was harmed by aggressive augmentation** (val acc dropped to 18.8%), indicating it lacks the capacity to recover from heavy masking.

### 9.3 Limitations and Potential Improvements

- **No waveform-level augmentation** was used. Time-shifting, noise injection, pitch shifting, and speed perturbation (implemented in `src/augmentation.py`) could increase effective dataset diversity and improve NN performance, especially on minority classes.
- **No transfer learning.** Pre-trained audio models (e.g., YAMNet) could provide better feature extractors, though this complicates on-device deployment.
- **Small dataset.** 970 training samples is at the lower bound for CNN-based audio classification. The NNs are data-limited rather than model-limited.

---

## 10. Reproducibility

### 10.1 How to Reproduce

```bash
source .venv/bin/activate
python src/train_nn.py
```

Requires TensorFlow 2.18 with GPU support. Runtime: ~14 minutes on NVIDIA RTX 2000 Ada.

### 10.2 Output Artifacts

| Artifact | Location |
|----------|----------|
| M2 float32 models | `models/m2_cnn2d_float32/tier{1,2,3}_best.h5` |
| M5 float32 models | `models/m5_cnn1d_int8_qat/tier{1,2,3}_best.h5` |
| M6 float32 models | `models/m6_dscnn_int8_qat/tier{1,2,3}_best.h5` |
| Training logs | `models/{model_dir}/tier{1,2,3}_training_log.csv` |
| Per-architecture metrics | `models/{model_dir}/evaluation_results.json` |
| Hyperparameter search | `models/hyperparameter_search.json` |
| Training curves | `results/training_curves_m{2,5,6}_tier{1,2,3}.png` |
| Confusion matrices | `results/cm_m{2,5,6}_tier_{1,2,3}[_norm].png` |
| ROC curves (Tier 1) | `results/roc_m{2,5,6}_tier_1.png` |
| Summary CSV | `results/nn_summary.csv` |
| Full results JSON | `results/nn_evaluation_results.json` |
| Analysis notebook | `notebooks/03_neural_networks.ipynb` |

### 10.3 Software Versions

| Package | Version |
|---------|---------|
| Python | 3.12 |
| TensorFlow | 2.18.0 |
| scikit-learn | ≥ 1.3.0 |
| numpy | ≥ 1.24.0 |
| GPU | NVIDIA RTX 2000 Ada (8 GB, CUDA 12.8) |
