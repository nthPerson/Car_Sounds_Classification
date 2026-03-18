# Classical ML Baselines — Model Documentation

**Phase:** 2 (Baseline Model Training)
**Date:** March 17, 2026
**Related code:** `src/train_classical.py`, `src/evaluate.py`, `notebooks/02_classical_ml.ipynb`

---

## 1. Purpose

Classical ML models serve as a **performance floor** for the project. They establish what accuracy is achievable with hand-crafted feature engineering and traditional algorithms, without learned representations. If a neural network cannot outperform these baselines, the deep learning pipeline needs investigation.

The baselines also provide **interpretability** through feature importance analysis, revealing which acoustic characteristics are most discriminative for each classification tier.

---

## 2. Input Features

Each audio clip is represented as a **441-dimensional hand-crafted feature vector** (see `data_preprocessing_spec.md` §7.3 and `src/features.py`).

### 2.1 Frame-Level Features (63 per frame)

| Feature Group | Count | Description |
|---------------|-------|-------------|
| MFCCs | 13 | Mel-frequency cepstral coefficients (coefficients 0–12) |
| Delta MFCCs | 13 | First-order temporal derivatives of MFCCs |
| Delta-delta MFCCs | 13 | Second-order temporal derivatives of MFCCs |
| Spectral centroid | 1 | Weighted mean of frequencies |
| Spectral bandwidth | 1 | Weighted standard deviation of frequencies |
| Spectral rolloff | 1 | Frequency below which 85% of energy is concentrated |
| Spectral contrast | 7 | Peak-to-valley ratio in 7 frequency sub-bands |
| RMS energy | 1 | Root mean square energy per frame |
| Zero-crossing rate | 1 | Rate of sign changes in the waveform |
| Chroma features | 12 | Energy in each of the 12 pitch classes |

### 2.2 Aggregation Statistics (7 per feature)

Each of the 63 frame-level features is summarized across the time axis using: **mean, standard deviation, minimum, maximum, median, skewness, kurtosis**.

Total: 63 features × 7 statistics = **441 dimensions**.

### 2.3 Feature Extraction Parameters

All features are computed using the same FFT parameters as the mel-spectrogram pipeline:

| Parameter | Value |
|-----------|-------|
| n_fft | 512 |
| hop_length | 256 |
| n_mels | 40 |
| fmin | 20 Hz |
| fmax | 8000 Hz |
| center | False |

---

## 3. Experimental Setup

### 3.1 Data Splits

| Split | Tier 1 & 2 | Tier 3 |
|-------|-----------|--------|
| Train + Val (merged for CV) | 1,178 samples | 806 samples |
| Test (held out) | 208 samples | 143 samples |

Train and validation sets were merged for classical ML training because cross-validation handles internal validation. The validation set was designed for neural network early stopping and is not needed here.

For Tier 3, samples with combined-fault labels (tier3_label = −1) were excluded from both training and test sets, as specified in the project taxonomy.

### 3.2 Evaluation Protocol

- **Hyperparameter tuning:** 5-fold stratified cross-validation on the merged train+val set
- **Scoring metric:** Macro-averaged F1 score (treats all classes equally, robust to class imbalance)
- **Final evaluation:** Single pass on the held-out test set (never used during tuning)
- **Random state:** 42 for all randomized operations (reproducibility)

### 3.3 Class Imbalance Handling

Both algorithms use class weighting to upweight minority classes:
- RF: `class_weight` parameter in the search space
- SVM: `class_weight='balanced'` (inversely proportional to class frequencies)

---

## 4. Model M1a — Random Forest

### 4.1 Algorithm

Scikit-learn `RandomForestClassifier` — an ensemble of decision trees trained on bootstrap samples with random feature subsets. Predictions are made by majority vote across all trees.

### 4.2 Hyperparameter Search

**Method:** `RandomizedSearchCV` (100 random combinations, 5-fold stratified CV, macro F1 scoring)

**Search space:**

| Parameter | Search Values |
|-----------|--------------|
| `n_estimators` | [100, 200, 500] |
| `max_depth` | [None, 10, 20, 30] |
| `min_samples_split` | [2, 5, 10] |
| `min_samples_leaf` | [1, 2, 4] |
| `max_features` | ['sqrt', 'log2', 0.3] |
| `class_weight` | ['balanced', 'balanced_subsample'] |

Full grid size: 3 × 4 × 3 × 3 × 3 × 2 = 648 combinations. RandomizedSearchCV with n_iter=100 was used to make the search tractable while still providing good coverage.

### 4.3 Best Hyperparameters (Selected by CV)

| Parameter | Tier 1 | Tier 2 | Tier 3 |
|-----------|--------|--------|--------|
| `n_estimators` | 100 | 200 | 500 |
| `max_depth` | None | 10 | None |
| `min_samples_split` | 10 | 10 | 2 |
| `min_samples_leaf` | 4 | 4 | 2 |
| `max_features` | 0.3 | sqrt | 0.3 |
| `class_weight` | balanced | balanced_subsample | balanced |

**Notable patterns:** Tier 2 selected a shallower tree (max_depth=10), suggesting it benefits from regularization to avoid fitting the imbalanced class distribution. Tier 3 selected more trees (500) and less regularization (min_samples_split=2, min_samples_leaf=2), consistent with a harder problem that benefits from more model capacity.

---

## 5. Model M1b — Support Vector Machine (SVM)

### 5.1 Algorithm

Scikit-learn `SVC` wrapped in a `Pipeline` with `StandardScaler`. The pipeline ensures features are z-score normalized (zero mean, unit variance) using training-set statistics before being passed to the SVM. The scaler is saved as part of the model artifact.

`probability=True` was enabled (Platt scaling) to obtain predicted probabilities for ROC curves and future comparison analysis.

### 5.2 Hyperparameter Search

**Method:** `GridSearchCV` (exhaustive, 5-fold stratified CV, macro F1 scoring)

**Search space** (two grids to handle kernel-dependent parameters):

| Grid | Parameters |
|------|-----------|
| RBF kernel | C ∈ {0.01, 0.1, 1, 10, 100}, gamma ∈ {scale, auto, 0.001, 0.01, 0.1} |
| Linear kernel | C ∈ {0.01, 0.1, 1, 10, 100} |

`class_weight='balanced'` was fixed (not searched).

Total combinations: 25 (RBF) + 5 (linear) = 30 × 5 folds = 150 fits per tier.

### 5.3 Best Hyperparameters (Selected by CV)

| Parameter | Tier 1 | Tier 2 | Tier 3 |
|-----------|--------|--------|--------|
| `kernel` | rbf | rbf | rbf |
| `C` | 100 | 10 | 10 |
| `gamma` | scale | 0.001 | scale |

**Notable patterns:** RBF kernel was selected for all tiers, indicating the class boundaries are non-linear in the 441-dimensional feature space. Tier 1 selected a larger C (100), consistent with a simpler binary task where less regularization is needed.

---

## 6. Results

### 6.1 Summary Table

| Model | Tier | Accuracy | F1 Macro | F1 Weighted | CV F1 Macro |
|-------|------|----------|----------|-------------|-------------|
| Random Forest | 1 (2-class) | 0.8990 | 0.8724 | 0.8965 | 0.8899 |
| SVM | 1 (2-class) | **0.9327** | **0.9163** | **0.9316** | 0.9244 |
| Random Forest | 2 (6-class) | 0.8413 | 0.6794 | 0.8272 | 0.7246 |
| SVM | 2 (6-class) | **0.8510** | **0.6992** | **0.8422** | 0.7753 |
| Random Forest | 3 (9-class) | **0.7622** | 0.7019 | **0.7522** | 0.7778 |
| SVM | 3 (9-class) | 0.7552 | **0.7106** | 0.7494 | 0.7917 |
| RF (top-50 features) | 2 (6-class) | 0.8125 | 0.6724 | 0.8124 | — |

### 6.2 Per-Class Results — Tier 1 (Binary: Normal vs. Anomaly)

| Class | Support | RF Prec. | RF Rec. | RF F1 | SVM Prec. | SVM Rec. | SVM F1 |
|-------|---------|----------|---------|-------|-----------|----------|--------|
| Normal | 61 | 0.8846 | 0.7541 | 0.8142 | 0.9273 | 0.8361 | 0.8793 |
| Anomaly | 147 | 0.9038 | 0.9592 | 0.9307 | 0.9346 | 0.9728 | 0.9533 |

Both models are stronger at detecting anomalies than confirming normal operation. The Normal class has lower recall — some normal samples are misclassified as anomalous (false alarms). SVM is notably better at recalling Normal samples (83.6% vs 75.4%).

### 6.3 Per-Class Results — Tier 2 (6-class, Primary Target)

| Class | Support | RF Prec. | RF Rec. | RF F1 | SVM Prec. | SVM Rec. | SVM F1 |
|-------|---------|----------|---------|-------|-----------|----------|--------|
| Normal Braking | 12 | 0.6923 | 0.7500 | 0.7200 | 0.6923 | 0.7500 | 0.7200 |
| Braking Fault | 11 | 1.0000 | 0.5455 | 0.7059 | 0.7273 | 0.7273 | 0.7273 |
| Normal Idle | 40 | 0.8049 | 0.8250 | 0.8148 | 0.8293 | 0.8500 | 0.8395 |
| Idle Fault | 118 | 0.8889 | 0.9492 | 0.9180 | 0.9113 | 0.9576 | 0.9339 |
| Normal Start-Up | 9 | 1.0000 | 0.1111 | 0.2000 | 0.5000 | 0.2222 | 0.3077 |
| Start-Up Fault | 18 | 0.6667 | 0.7778 | 0.7179 | 0.7333 | 0.6111 | 0.6667 |

**Key finding:** "Normal Start-Up" is by far the hardest class, with RF achieving only 11.1% recall and SVM 22.2%. With only 9 test samples and 52 training samples, this class is both rare and acoustically difficult to distinguish. The macro F1 is heavily dragged down by this class.

The "Idle Fault" majority class (118/208 test samples) achieves strong results (F1 > 0.91), which inflates overall accuracy to ~85% despite poor minority-class performance.

### 6.4 Per-Class Results — Tier 3 (9-class)

| Class | Support | RF Prec. | RF Rec. | RF F1 | SVM Prec. | SVM Rec. | SVM F1 |
|-------|---------|----------|---------|-------|-----------|----------|--------|
| Normal Braking | 12 | 0.6667 | 0.6667 | 0.6667 | 0.5833 | 0.5833 | 0.5833 |
| Worn Brakes | 11 | 0.8333 | 0.4545 | 0.5882 | 0.8000 | 0.7273 | 0.7619 |
| Normal Idle | 40 | 0.7872 | 0.9250 | 0.8506 | 0.7907 | 0.8500 | 0.8193 |
| Low Oil | 16 | 0.7333 | 0.6875 | 0.7097 | 0.7500 | 0.7500 | 0.7500 |
| Power Steering Fault | 19 | 0.8421 | 0.8421 | 0.8421 | 0.8235 | 0.7368 | 0.7778 |
| Serpentine Belt Fault | 18 | 0.8421 | 0.8889 | 0.8649 | 0.7826 | 1.0000 | 0.8780 |
| Normal Start-Up | 9 | 0.7500 | 0.3333 | 0.4615 | 0.6667 | 0.4444 | 0.5333 |
| Bad Ignition | 9 | 0.5833 | 0.7778 | 0.6667 | 0.7143 | 0.5556 | 0.6250 |
| Dead Battery | 9 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.6667 |

Tier 3 splits the aggregate Tier 2 fault classes into specific conditions. Despite having 9 classes instead of 6, macro F1 is comparable (~0.70–0.71 vs ~0.68–0.70), suggesting that splitting faults into specific conditions creates more acoustically distinguishable categories. "Normal Start-Up" remains the weakest class.

---

## 7. Feature Importance Analysis

### 7.1 Top-20 Most Important Features (Tier 2 RF)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | delta2_mfcc_0_std | 0.0195 |
| 2 | mfcc_0_std | 0.0162 |
| 3 | delta_mfcc_0_min | 0.0158 |
| 4 | delta2_mfcc_0_max | 0.0145 |
| 5 | delta_mfcc_0_std | 0.0137 |
| 6 | mfcc_1_std | 0.0123 |
| 7 | spectral_centroid_mean | 0.0109 |
| 8 | rms_min | 0.0108 |
| 9 | spectral_centroid_median | 0.0107 |
| 10 | delta2_mfcc_0_min | 0.0104 |
| 11 | mfcc_1_min | — |
| 12 | spectral_contrast_2_mean | — |
| 13 | delta2_mfcc_1_std | — |
| 14 | mfcc_4_max | — |
| 15 | spectral_centroid_min | — |
| 16 | delta_mfcc_1_mean | — |
| 17 | mfcc_12_std | — |
| 18 | spectral_centroid_max | — |
| 19 | spectral_rolloff_median | — |
| 20 | zcr_std | — |

### 7.2 Interpretation

**MFCC coefficients dominate.** The top 5 features all involve MFCC coefficient 0 (which represents overall spectral energy) and its temporal derivatives. The standard deviation and extrema (min/max) of these features across time frames are more discriminative than their means, suggesting that **temporal variability** in spectral energy is key to distinguishing car sound conditions.

**Feature groups by importance (aggregated across all 7 stats):**

| Rank | Base Feature | Description |
|------|-------------|-------------|
| 1 | delta2_mfcc_0 | Acceleration of spectral energy over time |
| 2 | mfcc_0 | Overall spectral energy level |
| 3 | delta_mfcc_0 | Rate of change of spectral energy |
| 4 | spectral_centroid | Brightness of the sound |
| 5 | mfcc_1 | Spectral slope |
| 6 | zcr | Noisiness/periodicity of the signal |

This aligns with acoustic intuition: different car conditions (e.g., worn brakes, low oil, dead battery) produce different energy levels, spectral brightness, and noise characteristics.

### 7.3 Reduced Feature Set Experiment

Retraining the Tier 2 RF on only the top-50 features (using the same best hyperparameters):

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| RF (441 features) | 0.8413 | 0.6794 | 0.8272 |
| RF (top-50 features) | 0.8125 | 0.6724 | 0.8124 |
| **Delta** | **−0.029** | **−0.007** | **−0.015** |

The accuracy drop of 2.9 percentage points is modest, and the macro F1 drop is only 0.7 pp. This suggests the top 50 features capture most of the discriminative information, but the remaining 391 features collectively contribute a small additional signal. For deployment purposes (not relevant since classical ML models don't run on the Arduino), the full feature set is preferred.

---

## 8. Discussion

### 8.1 Key Takeaways for the Research Paper

1. **Classical ML achieves reasonable performance on this task.** SVM with RBF kernel on 441 hand-crafted features reaches 93.3% accuracy on binary anomaly detection and 85.1% on the 6-class state-aware taxonomy. These are credible baselines that the neural network models must improve upon.

2. **The accuracy–F1 gap reveals class imbalance effects.** On Tier 2, the 16 percentage point gap between accuracy (85.1%) and macro F1 (69.9%) demonstrates that overall accuracy is a misleading metric for this dataset. The models achieve high accuracy by correctly classifying the dominant "Idle Fault" class while performing poorly on minority classes. **Macro F1 should be the primary metric in all comparisons.**

3. **"Normal Start-Up" is an inherently difficult class** with only 52 training samples (after merging train+val) and acoustic characteristics that overlap significantly with "Start-Up Fault." Both RF and SVM achieve very low recall (11–22%) on this class. Data augmentation in the neural network pipeline may help, but this class will likely remain the weakest across all models.

4. **Tier 3 is not significantly harder than Tier 2** despite having 50% more classes (9 vs 6). Splitting aggregate fault classes into specific conditions (e.g., "Idle Fault" → "Low Oil" / "Power Steering Fault" / "Serpentine Belt Fault") appears to create more acoustically distinct categories, partially offsetting the increased class count.

5. **SVM with RBF kernel consistently outperforms Random Forest on Tiers 1–2**, with larger margins on easier tasks. RF is slightly better on the harder Tier 3 task, possibly because the ensemble nature of RF provides more robustness with smaller per-class sample sizes.

6. **Feature importance is concentrated in MFCC-based features**, particularly the zeroth coefficient and its temporal derivatives. This validates the use of mel-spectrogram and MFCC representations as inputs to the neural networks. Spectral centroid and zero-crossing rate also contribute, consistent with the acoustic nature of the diagnostic task.

### 8.2 Limitations

- **No augmentation:** Classical ML models were trained on the raw feature vectors without data augmentation. The neural network models will use both waveform and spectrogram augmentation, which may partly explain any performance gap.
- **Feature engineering ceiling:** The 441-dimensional feature vector, while comprehensive, represents a fixed set of acoustic descriptors. Neural networks can learn task-specific features that may capture patterns not present in these hand-crafted features.
- **Small test set:** With only 208 test samples (143 for Tier 3), per-class metrics for minority classes have high variance. Differences of a few samples can swing F1 scores significantly.

### 8.3 Baseline Targets for Neural Networks

For the neural networks (Phase 3) to be considered successful, they should at minimum match these classical ML results. Target improvements:

| Tier | Classical ML Best (F1 Macro) | Target for NN |
|------|------------------------------|---------------|
| 1 | 0.9163 (SVM) | ≥ 0.92 |
| 2 | 0.6992 (SVM) | ≥ 0.75 |
| 3 | 0.7106 (SVM) | ≥ 0.75 |

The Tier 2 target is aggressive but achievable if augmentation and learned features can improve minority-class recall.

---

## 9. Reproducibility

### 9.1 How to Reproduce

```bash
source .venv/bin/activate
python src/train_classical.py
```

Runtime: ~13 minutes on a modern CPU.

### 9.2 Output Artifacts

| Artifact | Location |
|----------|----------|
| Trained models | `models/m1_classical/{rf,svm}_tier{1,2,3}.joblib` |
| Reduced-feature RF | `models/m1_classical/rf_tier2_top50.joblib` |
| Top-50 feature indices | `models/m1_classical/top50_feature_indices.npy` |
| Search results (all CV scores) | `models/m1_classical/search_results.json` |
| Evaluation metrics | `models/m1_classical/evaluation_results.json` |
| Confusion matrices | `results/cm_{rf,svm}_tier_{1,2,3}[_norm].png` |
| Feature importance chart | `results/feature_importance_tier2.png` |
| ROC curves | `results/roc_{rf,svm}_tier_1.png` |
| Summary CSV | `results/classical_ml_summary.csv` |
| Analysis notebook | `notebooks/02_classical_ml.ipynb` |

### 9.3 Software Versions

| Package | Version |
|---------|---------|
| Python | 3.12 |
| scikit-learn | ≥ 1.3.0 |
| librosa | ≥ 0.10.0 |
| joblib | ≥ 1.3.0 |
| numpy | ≥ 1.24.0 |
