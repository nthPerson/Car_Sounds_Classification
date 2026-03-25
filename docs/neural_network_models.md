# Neural Network Models — Model Documentation

**Phase:** 3 (Neural Network Training)
**Date:** March 22, 2026 (updated with noise-augmented results)
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

Features were pre-computed in Phase 1 and stored un-normalized. Z-score normalization was applied at training time using statistics computed from the training set only (`data/normalization_stats_noise_augmented.npz` for the final noise-augmented training condition).

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
| Learning rate | 0.002 | 0.001 | 0.002 |
| Dropout rate | 0.2 | 0.3 | N/A (no dropout layer) |
| SpecAugment | None | Default | None |

### 5.3 Data Augmentation

Two levels of augmentation are used:

**Level 1 — Pre-computed waveform augmentation (class-balanced):**

Training data was expanded from 970 → 3,317 samples using `src/generate_augmented_data.py`. Five waveform-domain augmentations are applied with stochastic probability per sample:

| Augmentation | Probability | Parameters |
|-------------|------------|-----------|
| Time shift | 50% | ±100 ms (1,600 samples) |
| Noise injection | 50% | SNR 5–25 dB (Gaussian) |
| Pitch shift | 30% | ±2 semitones |
| Speed perturbation | 30% | 0.9–1.1x rate |
| Random gain | 50% | 0.7–1.3x amplitude |

**Class-aware augmentation multipliers** target ~550 samples per Tier 2 class:

| Tier 2 Class | Original | Multiplier | Final |
|-------------|----------|-----------|-------|
| Normal Braking | 54 | 9x | 540 |
| Braking Fault | 53 | 9x | 530 |
| Normal Idle | 185 | 2x | 555 |
| Idle Fault | 552 | 0x (no augmentation) | 552 |
| Normal Start-Up | 43 | 12x | 559 |
| Start-Up Fault | 83 | 6x | 581 |

This reduces the Tier 2 imbalance ratio from 12.8x (552:43) to 1.1x (581:530).

**Level 2 — Real-world noise injection (100% of augmented samples):**

Each augmented sample receives noise from a bank of 360 real-world recordings collected with the Arduino Nano 33 BLE Sense Rev2's PDM microphone in and around a 2008 Volkswagen Jetta. Noise is mixed at a random SNR from {5, 10, 15, 20} dB, with operational-state-aware noise group selection (e.g., braking samples receive road/driving noise, idle samples receive engine/HVAC noise). See `docs/noise_data_collection_and_augmentation.md` for full details.

**Level 3 — On-the-fly SpecAugment (optional per architecture):**

Applied during training with 50% probability per sample per epoch. Only used for M5 (1-D CNN on MFCCs) based on hyperparameter search results; M2 and M6 performed best without SpecAugment when waveform augmentation was already applied.

### 5.4 Data Splits

| Split | Tier 1 & 2 | Tier 3 |
|-------|-----------|--------|
| Train (augmented) | 3,317 samples | ~2,290 samples |
| Val (unaugmented) | 208 samples | 142 samples |
| Test (unaugmented) | 208 samples | 143 samples |

Augmentation is applied only to the training set. Val and test sets remain unaugmented to ensure unbiased evaluation. Train and val are kept separate (val is used for early stopping, LR scheduling, and model checkpointing).

---

## 6. Hyperparameter Search

A structured manual search was conducted on Tier 2 (primary target) for all three architectures. Three sweeps were performed sequentially: learning rate → dropout → augmentation intensity.

### 6.1 M2 (2-D CNN) Search Results

| Sweep | Setting | Best Val Accuracy |
|-------|---------|------------------|
| **LR** | 0.0005 → 0.837, 0.001 → 0.899, **0.002 → 0.904** | LR=0.002 |
| **Dropout** | **0.2 → 0.909**, 0.3 → 0.899, 0.5 → 0.885 | Dropout=0.2 |
| **Augmentation** | **none → 0.904**, default → 0.894, aggressive → 0.904 | None |

### 6.2 M5 (1-D CNN) Search Results

| Sweep | Setting | Best Val Accuracy |
|-------|---------|------------------|
| **LR** | 0.0005 → 0.870, **0.001 → 0.885**, 0.002 → 0.875 | LR=0.001 |
| **Dropout** | 0.2 → 0.846, 0.3 → 0.861, **0.5 → 0.861** | Dropout=0.3 |
| **Augmentation** | none → 0.875, **default → 0.889**, aggressive → 0.856 | Default |

### 6.3 M6 (DS-CNN) Search Results

| Sweep | Setting | Best Val Accuracy |
|-------|---------|------------------|
| **LR** | 0.0005 → 0.904, 0.001 → 0.894, **0.002 → 0.914** | LR=0.002 |
| **Augmentation** | **none → 0.909**, default → 0.899, aggressive → 0.904 | None |

**Key finding compared to pre-augmentation:** M6 (DS-CNN) was previously unstable at LR ≥ 0.001 (val accuracy ~23%), but with the augmented + balanced dataset it trains stably at LR=0.002 (val accuracy 91.4%). The expanded dataset (3,317 vs 970 samples) provides sufficient gradient diversity for the depthwise separable convolutions to converge reliably.

---

## 7. Results

### 7.1 Summary Table (Condition C — Noise-Augmented)

| Model | Tier | Accuracy | F1 Macro | F1 Weighted | Epochs |
|-------|------|----------|----------|-------------|--------|
| M2 (2-D CNN) | 1 | 0.8798 | 0.8607 | 0.8820 | 41 |
| M2 (2-D CNN) | 2 | 0.8606 | 0.7335 | 0.8582 | 57 |
| M2 (2-D CNN) | 3 | 0.7762 | 0.7217 | 0.7740 | 66 |
| M5 (1-D CNN) | 1 | 0.8942 | 0.8736 | 0.8947 | 50 |
| M5 (1-D CNN) | 2 | 0.7933 | 0.5956 | 0.7944 | 65 |
| M5 (1-D CNN) | 3 | 0.6923 | 0.6116 | 0.6909 | 72 |
| M6 (DS-CNN) | 1 | 0.9087 | 0.8903 | 0.9089 | 49 |
| M6 (DS-CNN) | 2 | 0.8413 | 0.7388 | 0.8403 | 25 |
| M6 (DS-CNN) | 3 | 0.8182 | 0.7624 | 0.8147 | 62 |

### 7.2 Comparison with Classical ML Baselines

| Tier | Best Classical ML | Best NN | NN Beats Baseline? |
|------|------------------|---------|--------------------|
| 1 | SVM (F1=0.916) | M6 (F1=0.890) | No (−0.026) |
| 2 | SVM (F1=0.699) | **M6 (F1=0.739)** | **Yes (+0.040)** |
| 3 | SVM (F1=0.711) | **M6 (F1=0.762)** | **Yes (+0.051)** |

With noise-augmented + class-balanced training data, the NN architectures exceed classical ML baselines on Tiers 2 and 3. On Tier 1 (binary), the SVM baseline remains stronger. The M6 DS-CNN continues to be the best architecture on Tiers 2 and 3, with a particularly strong showing on Tier 3 (+0.051 over SVM).

### 7.3 Per-Class Results — Tier 2 (M6 DS-CNN, best NN)

| Class | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Normal Braking | 12 | 0.5882 | 0.8333 | 0.6897 |
| Braking Fault | 11 | 1.0000 | 0.6364 | 0.7778 |
| Normal Idle | 40 | 0.8611 | 0.7750 | 0.8158 |
| Idle Fault | 118 | 0.8880 | 0.9407 | 0.9136 |
| Normal Start-Up | 9 | 0.5000 | 0.5556 | 0.5263 |
| Start-Up Fault | 18 | 0.8462 | 0.6111 | 0.7097 |

**"Normal Start-Up" recall remains improved** over classical baselines (55.6% vs 11-22%) due to class-balanced augmentation (12x expansion). Braking Fault achieves perfect precision (1.0) with the noise-augmented training data.

### 7.4 Per-Class Results — Tier 3 (M6 DS-CNN)

| Class | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Normal Braking | 12 | 0.9231 | 1.0000 | 0.9600 |
| Worn Brakes | 11 | 0.8182 | 0.8182 | 0.8182 |
| Normal Idle | 40 | 0.8974 | 0.8750 | 0.8861 |
| Low Oil | 16 | 0.7222 | 0.8125 | 0.7647 |
| Power Steering Fault | 19 | 0.8889 | 0.8421 | 0.8649 |
| Serpentine Belt Fault | 18 | 0.9474 | 1.0000 | 0.9730 |
| Normal Start-Up | 9 | 0.4286 | 0.3333 | 0.3750 |
| Bad Ignition | 9 | 0.6250 | 0.5556 | 0.5882 |
| Dead Battery | 9 | 0.6000 | 0.6667 | 0.6316 |

Tier 3 performance improved significantly with noise augmentation, with Normal Braking reaching 96.0% F1 and Serpentine Belt Fault reaching 97.3% F1. "Normal Start-Up" remains the weakest class (F1=0.375) due to limited source recordings (43 original samples).

---

## 8. Architecture Comparison

### 8.1 M2 vs M5 vs M6

| Aspect | M2 (2-D CNN) | M5 (1-D CNN) | M6 (DS-CNN) |
|--------|-------------|-------------|-------------|
| Best F1 macro (Tier 2) | 0.734 | 0.596 | **0.739** |
| Parameters (Tier 2) | 14,566 | **6,246** | 8,166 |
| Float32 model size | 237 KB | **140 KB** | 246 KB |
| Est. int8 model size | ~16 KB | **~8 KB** | ~9 KB |
| Est. tensor arena | ~70-80 KB | **~15-25 KB** | ~35-45 KB |
| Training stability (with aug) | Stable | Stable | Stable |
| Convergence speed | ~50-75 epochs | ~55-81 epochs | ~41-65 epochs |

### 8.2 Deployment Recommendations

1. **M6 (DS-CNN) — Best accuracy candidate.** Highest F1 macro on Tiers 2 (0.739) and 3 (0.762). Tensor arena (~35-45 KB) fits comfortably within the 80 KB budget. Int8 model size (~9 KB) is well within flash budget.
2. **M2 (2-D CNN) — Strong alternative.** Competitive accuracy (Tier 2 F1=0.734). Tensor arena (~75 KB) is tight but within budget. More parameters = more room for quantization to degrade accuracy.
3. **M5 (1-D CNN) — Smallest footprint.** Safest deployment if SRAM is constrained (~20 KB arena). Lower accuracy but the most compact model (~8 KB int8).

---

## 9. Augmentation Ablation Study

Four training conditions were compared to measure the impact of augmentation strategy on model performance. All conditions use the same model architectures, hyperparameters, validation set, and test set.

| Condition | Description | Train Size |
|-----------|-------------|-----------|
| A (Original) | No augmentation, class-imbalanced | 970 |
| B (Synthetic) | Waveform augmentation + class balancing | 3,317 |
| C (Noise) | Waveform aug + class balancing + real-world noise injection | 3,317 |
| D (Combined) | B + C augmented copies together (originals + synthetic + noise) | 5,664 |

Note: Condition D combines all of Condition B (970 originals + 2,347 synthetic copies) with the 2,347 noise-augmented copies from Condition C. The originals appear once; the augmented copies appear in both their synthetic and noise-injected forms. This inverts the original class imbalance — Idle Fault (552 samples, no augmentation) becomes the smallest class while augmented classes have ~1,000+ samples. The `compute_class_weight('balanced')` setting in training compensates for this.

### 9.1 F1 Macro Comparison (Primary Metric)

| Model | Tier | Cond. A | Cond. B | Cond. C | Cond. D | Best |
|-------|------|---------|---------|---------|---------|------|
| M2 (2-D CNN) | 1 | 0.868 | **0.898** | 0.861 | **0.898** | B/D |
| M2 (2-D CNN) | 2 | 0.666 | 0.732 | 0.734 | **0.751** | **D** |
| M2 (2-D CNN) | 3 | 0.715 | 0.717 | 0.722 | **0.764** | **D** |
| M5 (1-D CNN) | 1 | **0.875** | 0.864 | 0.874 | 0.858 | A |
| M5 (1-D CNN) | 2 | 0.616 | **0.638** | 0.596 | 0.625 | B |
| M5 (1-D CNN) | 3 | 0.616 | **0.657** | 0.612 | 0.651 | B |
| M6 (DS-CNN) | 1 | 0.858 | 0.900 | 0.890 | **0.909** | **D** |
| M6 (DS-CNN) | 2 | 0.571 | **0.777** | 0.739 | 0.742 | B |
| M6 (DS-CNN) | 3 | 0.600 | 0.725 | **0.762** | 0.686 | C |

### 9.2 Ablation Observations

**A→B (effect of synthetic augmentation + class balancing):**
- Universally beneficial, especially for M6 DS-CNN (+0.206 on Tier 2), confirming that DS-CNN is highly sensitive to dataset size and class balance
- The single most impactful improvement across the entire project

**B→C (isolated effect of real-world noise injection):**
- Mixed results on the clean test set: some model-tier combinations improve slightly (M2 Tier 2/3, M6 Tier 3), others decline slightly (M6 Tier 2, M5 Tier 2/3)
- The deltas are small (±0.01–0.05), consistent with the expectation that noise injection primarily improves robustness to deployment-time noise rather than performance on clean test data

**D (combined — effect of maximum data volume):**
- M2 (2-D CNN) benefits the most from combined data: best scores on Tiers 2 and 3 (0.751 and 0.764), suggesting the 2-D CNN architecture can effectively leverage the additional 5,664 training samples
- M6 (DS-CNN) achieves its best Tier 1 score (0.909) with combined data, but Tier 2 and 3 performance doesn't improve over Condition B — the inverted class imbalance may hurt DS-CNN despite class weighting
- M5 (1-D CNN) does not benefit from combined data — Condition B remains best across all tiers, suggesting MFCCs don't gain from noise-diversity augmentation

**Best condition per architecture (Tier 2, primary target):**
- M2: Condition D (F1=0.751) — benefits from data volume
- M5: Condition B (F1=0.638) — noise augmentation hurts MFCC-based models
- M6: Condition B (F1=0.777) — already saturated at 3,317 samples for this architecture

### 9.3 Selection for Phase 4 (Quantization)

For Phase 4, we should select the best-performing model+condition combination per architecture to quantize and deploy. The top candidates for Tier 2 (primary deployment target):

| Rank | Model + Condition | Tier 2 F1 Macro | Notes |
|------|------------------|----------------|-------|
| 1 | M6 (DS-CNN) + Cond. B | **0.777** | Best overall Tier 2 score |
| 2 | M2 (2-D CNN) + Cond. D | **0.751** | Best M2 score; benefits from combined data |
| 3 | M6 (DS-CNN) + Cond. D | 0.742 | Slight degradation from inverted imbalance |
| 4 | M6 (DS-CNN) + Cond. C | 0.739 | Noise robustness for on-device use |

The final choice depends on the Phase 4 quantization results — the best float32 model may not remain the best after int8 quantization.

---

## 10. Discussion

### 10.1 Impact of Waveform Augmentation and Class Balancing

The pre-computed waveform augmentation with class-aware balancing was the single most impactful improvement in Phase 3. Key effects:

**Dataset size:** Expanding from 970 → 3,317 training samples provided sufficient gradient diversity for all architectures, particularly the DS-CNN which requires more data due to its factorized convolutions.

**Class balancing:** Reducing the Tier 2 imbalance from 12.8x to 1.1x had a transformative effect on minority class performance. "Normal Start-Up" (the hardest class) went from 11% recall (classical RF) to 55.6% recall (M6 DS-CNN).

**Training stability:** The M6 DS-CNN was previously unstable at LR ≥ 0.001 with 970 samples but trains reliably at LR=0.002 with 3,317 balanced samples. This confirms that DS-CNN's sensitivity was data-driven, not architectural.

### 10.2 SpecAugment Interaction with Waveform Augmentation

With waveform augmentation already applied, additional SpecAugment was generally not needed:
- **M2 and M6 performed best without SpecAugment** — waveform augmentation provides sufficient regularization
- **M5 benefited from default SpecAugment** — MFCCs are a more compressed representation, and the additional masking-based regularization helps

This suggests that waveform and spectrogram augmentation are partially redundant for mel-spectrogram models but complementary for MFCC models.

### 10.3 Remaining Limitations

- **"Normal Start-Up" remains the weakest class** (55.6% recall for M6, 44.4% for Tier 3). With only 43 original samples, even 12x augmentation produces limited acoustic diversity. All augmented variants derive from the same 43 source recordings.
- **Small test set.** With 208 test samples (143 for Tier 3), per-class metrics have high variance. Differences of 1-2 samples can swing F1 scores by several percentage points.
- **No transfer learning.** Pre-trained audio models (e.g., YAMNet) could provide better feature extractors, though this complicates on-device deployment.

---

## 11. Reproducibility

### 11.1 How to Reproduce

```bash
source .venv/bin/activate

# Generate noise-augmented data (requires noise bank from notebooks/04 + 05)
# Run notebooks/05_noise_augmentation.ipynb

# Train on noise-augmented data (Condition C, default)
python src/train_nn.py                  # ~6 minutes on GPU (no HP search)

# To train on synthetic-augmented data (Condition B), edit src/train_nn.py:
#   AUGMENTATION_VARIANT = "standard"
# To train on original data (Condition A):
#   AUGMENTATION_VARIANT = "none"
```

Requires TensorFlow 2.18 with GPU support.

### 11.2 Output Artifacts

| Artifact | Location |
|----------|----------|
| Noise-augmented training features | `data/features/{mel_spectrograms,mfcc}/train_noise_augmented.npz` |
| Noise-augmented normalization stats | `data/normalization_stats_noise_augmented.npz` |
| Noise augmentation report | `data/noise_augmentation_report.json` |
| Noise bank | `data/noise_bank/noise_bank.npz` |
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
| Condition B archive | `results/nn_summary_condition_b.csv` |
| Condition C archive | `results/nn_summary_condition_c.csv` |
| Full results JSON | `results/nn_evaluation_results.json` |
| Analysis notebook | `notebooks/03_neural_networks.ipynb` |
| Noise analysis notebook | `notebooks/04_noise_analysis.ipynb` |
| Noise augmentation notebook | `notebooks/05_noise_augmentation.ipynb` |

### 11.3 Software Versions

| Package | Version |
|---------|---------|
| Python | 3.12 |
| TensorFlow | 2.18.0 |
| scikit-learn | ≥ 1.3.0 |
| numpy | ≥ 1.24.0 |
| librosa | ≥ 0.10.0 |
| GPU | NVIDIA RTX 2000 Ada (8 GB, CUDA 12.8) |
