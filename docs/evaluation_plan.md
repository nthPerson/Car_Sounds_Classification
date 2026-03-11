# Evaluation Plan

**Parent document:** [project_overview.md](project_overview.md)
**Depends on:** [model_architecture_spec.md](model_architecture_spec.md), [quantization_deployment_spec.md](quantization_deployment_spec.md)
**Date:** March 10, 2026

---

## Table of Contents

1. [Evaluation Philosophy](#1-evaluation-philosophy)
2. [Evaluation Dimensions](#2-evaluation-dimensions)
3. [Dimension 1 — Classification Performance (PC Evaluation)](#3-dimension-1--classification-performance-pc-evaluation)
4. [Dimension 2 — Quantization Impact Analysis](#4-dimension-2--quantization-impact-analysis)
5. [Dimension 3 — On-Device Performance Benchmarking](#5-dimension-3--on-device-performance-benchmarking)
6. [Dimension 4 — End-to-End System Evaluation](#6-dimension-4--end-to-end-system-evaluation)
7. [Statistical Significance and Confidence](#7-statistical-significance-and-confidence)
8. [Reporting Templates](#8-reporting-templates)
9. [Evaluation Execution Checklist](#9-evaluation-execution-checklist)

---

## 1. Evaluation Philosophy

This project has a dual evaluation objective:

1. **Standard ML evaluation:** How well do the models classify automotive audio conditions? This is the classic machine learning question — accuracy, precision, recall, F1-score, confusion matrices.

2. **Quantization and deployment evaluation:** What is the cost of deploying on a constrained microcontroller? This answers: How much accuracy is lost from quantization? How fast is inference? Does it fit in memory? Is it usable in real time?

Both dimensions are equally important. A model that achieves 95% accuracy on PC but can't run on the Arduino is useless for this project. Conversely, a model that fits on the Arduino but only achieves 50% accuracy isn't practically useful.

### 1.1 Evaluation Principles

- **All models are evaluated on the same held-out test set.** The test set is never used for training, hyperparameter tuning, or model selection. It is used ONCE at the end for final reporting.
- **Validation set is used for all intermediate decisions** — hyperparameter tuning, early stopping, model selection, threshold tuning.
- **Metrics are reported per-tier.** Each taxonomy tier (Tier 1: 2-class, Tier 2: 6-class, Tier 3: 9-class) is evaluated independently.
- **Quantized models are evaluated on PC first** (using the TFLite interpreter), then on-device. The PC TFLite evaluation is bit-exact with on-device inference, so any accuracy differences between PC-TFLite and on-device indicate a feature extraction parity issue (training-serving skew), not a model issue.

---

## 2. Evaluation Dimensions

| Dimension | What It Measures | Where Measured | When |
|-----------|-----------------|----------------|------|
| **D1: Classification Performance** | Accuracy, precision, recall, F1, confusion matrices | PC (Keras for float32, TFLite interpreter for int8) | After training each model |
| **D2: Quantization Impact** | Accuracy degradation, per-class changes, confidence calibration | PC (compare float32 vs int8 predictions) | After quantization |
| **D3: On-Device Performance** | Inference latency, flash usage, SRAM usage, power | Arduino Nano 33 BLE Sense Rev2 | After deployment |
| **D4: End-to-End System** | Real-world inference accuracy with live audio playback | Arduino + speaker setup | Final validation |

---

## 3. Dimension 1 — Classification Performance (PC Evaluation)

### 3.1 Primary Metrics

#### 3.1.1 Overall Accuracy

```
Accuracy = (Number of correct predictions) / (Total predictions)
```

Simple and intuitive, but can be misleading with imbalanced classes. A model that always predicts "Idle Fault" (the majority class in Tier 2) would achieve ~57% accuracy without learning anything useful.

#### 3.1.2 Macro-Averaged F1-Score (Primary Metric for Model Selection)

```
F1_macro = (1/N) × Σ F1_i    for i = 1..N classes
```

Where F1 for each class is the harmonic mean of precision and recall:

```
F1_i = 2 × (Precision_i × Recall_i) / (Precision_i + Recall_i)
```

**Macro-averaged F1** treats all classes equally regardless of their size, making it the best single metric for imbalanced datasets. A model that ignores minority classes will have low macro-F1 even if its overall accuracy is high.

**This is the primary metric for model comparison and selection.**

#### 3.1.3 Weighted-Averaged F1-Score

```
F1_weighted = Σ (n_i / N_total) × F1_i
```

Weights each class's F1 by its support (number of test samples). More representative of real-world performance if the test distribution matches the deployment distribution.

Report both macro and weighted F1 in all results tables.

#### 3.1.4 Per-Class Precision and Recall

For each class i:
```
Precision_i = TP_i / (TP_i + FP_i)
Recall_i    = TP_i / (TP_i + FN_i)
```

These reveal whether the model has specific weaknesses:
- Low precision for class X → the model frequently predicts X when it shouldn't (false alarms)
- Low recall for class X → the model misses actual X samples (missed detections)

For a diagnostic tool, **recall is more important than precision** — it's better to flag a potential fault (false alarm) than to miss a real one (missed detection). The implications of missing a real fault (e.g., worn brakes, low oil) are safety-critical.

### 3.2 Confusion Matrix

Compute and visualize the N×N confusion matrix for each model and tier. The confusion matrix reveals:
- Which classes are most often confused with each other
- Whether errors are concentrated between acoustically similar conditions
- Whether the operational state boundary (braking/idle/startup) is respected

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title(f'Confusion Matrix — {model_name} (Tier {tier})')
plt.tight_layout()
```

#### 3.2.1 Normalized Confusion Matrix

Also compute the row-normalized confusion matrix (each row sums to 1.0), which shows the classification rate per true class:

```python
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
```

This is especially useful for imbalanced classes where raw counts can be misleading.

### 3.3 Classification Report

Use scikit-learn's `classification_report` for a comprehensive per-class summary:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
```

Output format:
```
                    precision    recall  f1-score   support

   Normal Braking     0.9167    0.9167    0.9167        12
    Braking Fault     0.8462    0.9167    0.8800        12
      Normal Idle     0.9500    0.9500    0.9500        40
       Idle Fault     0.9831    0.8814    0.9293       118
  Normal Start-Up     0.8000    0.8000    0.8000        10
   Start-Up Fault     0.8333    0.8333    0.8333        18

         accuracy                         0.9038       210
        macro avg     0.8882    0.8830    0.8849       210
     weighted avg     0.9451    0.9038    0.9224       210
```

### 3.4 ROC and AUC (Tier 1 — Binary Only)

For the binary anomaly detection tier, compute the Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC):

```python
from sklearn.metrics import roc_curve, auc

# Get probability scores (not hard predictions)
y_prob = model.predict(X_test)[:, 1]  # Probability of "Anomaly" class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
```

AUC is a threshold-independent metric that measures the model's ability to rank anomalous samples higher than normal samples. AUC > 0.9 indicates excellent separation; AUC > 0.95 is very strong.

### 3.5 Confidence Calibration

Examine whether the model's predicted probabilities (softmax outputs) are well-calibrated — i.e., when the model says "90% confidence," is it correct 90% of the time?

**Reliability diagram:** Plot predicted confidence (x-axis) vs. actual accuracy (y-axis) using binned calibration:

```python
from sklearn.calibration import calibration_curve

# For each class, compute calibration curve
prob_true, prob_pred = calibration_curve(y_test == class_idx, y_prob[:, class_idx], n_bins=10)
```

**Expected Calibration Error (ECE):** A single number summarizing calibration quality:

```
ECE = Σ (|B_m| / N) × |accuracy(B_m) - confidence(B_m)|
```

Where B_m are bins of predicted confidence. Lower ECE = better calibration. ECE < 0.05 is generally good.

Calibration is especially important for this application because the output confidence should be trustworthy — a "95% confident it's a fault" prediction should be taken more seriously than a "55% confident" prediction.

---

## 4. Dimension 2 — Quantization Impact Analysis

This is the central academic contribution of the project — a systematic comparison of how quantization affects model performance.

### 4.1 Paired Comparison: Float32 vs. Int8

For each neural network architecture and tier, compare the float32 baseline against its quantized variants:

| Comparison | What It Shows |
|-----------|--------------|
| **M2 (float32) → M3 (int8 PTQ)** | Impact of post-training quantization on accuracy |
| **M2 (float32) → M4 (int8 QAT)** | Impact of quantization-aware training on accuracy |
| **M3 (int8 PTQ) vs. M4 (int8 QAT)** | Whether QAT recovers accuracy lost by PTQ |

### 4.2 Metrics for Quantization Impact

#### 4.2.1 Absolute Accuracy Drop

```
ΔAccuracy = Accuracy_float32 - Accuracy_int8
```

Report for overall accuracy AND per-class accuracy. Some classes may be more sensitive to quantization than others.

#### 4.2.2 F1-Score Drop (Macro)

```
ΔF1_macro = F1_macro_float32 - F1_macro_int8
```

More reliable than accuracy drop for imbalanced datasets.

#### 4.2.3 Prediction Agreement Rate

The percentage of test samples where the float32 and int8 models produce the SAME prediction:

```python
agreement = np.mean(y_pred_float32 == y_pred_int8)
```

A high agreement rate (>95%) indicates that quantization rarely changes the model's decision, even if the confidence scores shift.

#### 4.2.4 Confidence Distribution Shift

Compare the distribution of predicted confidence (max softmax probability) between float32 and int8 models:

```python
# Float32 confidences
conf_float32 = np.max(model_float32.predict(X_test), axis=1)

# Int8 confidences (dequantized from TFLite interpreter)
conf_int8 = np.max(int8_probabilities, axis=1)

# Plot histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(conf_float32, bins=50, alpha=0.7, label='Float32')
ax1.hist(conf_int8, bins=50, alpha=0.7, label='Int8')
ax1.legend()
ax1.set_title('Confidence Distributions')
```

Quantization often reduces the "sharpness" of softmax outputs (confidences become less extreme), which can affect threshold-based decisions.

#### 4.2.5 Per-Class Quantization Sensitivity

Compute the accuracy drop per class:

```
ΔAccuracy_class_i = Recall_float32_class_i - Recall_int8_class_i
```

Classes with very few training samples or with subtle acoustic differences may be more sensitive to quantization. Report which classes are most affected and hypothesize why.

#### 4.2.6 Model Size Comparison

| Model | Float32 Size | Int8 PTQ Size | Int8 QAT Size | Compression Ratio |
|-------|-------------|--------------|--------------|-------------------|
| M2/M3/M4 (2-D CNN) | — KB | — KB | — KB | —× |
| M5 (1-D CNN) | — KB | — KB | — KB | —× |
| M6 (DS-CNN) | — KB | — KB | — KB | —× |

### 4.3 Quantization Error Analysis

For a deeper analysis, examine the quantization error at intermediate layers:

1. Run the float32 model and int8 model on the same input
2. Extract intermediate layer outputs (activations) from both
3. Compute the element-wise difference (quantization noise)
4. Report the Signal-to-Quantization-Noise Ratio (SQNR) per layer:

```
SQNR_layer = 10 × log10(Var(activation_float32) / Var(activation_float32 - dequantized_int8))
```

Higher SQNR = less quantization distortion. Layers with low SQNR (< 20 dB) are the primary sources of accuracy degradation and may benefit from mixed-precision quantization (keeping those layers in higher precision).

**Note:** This analysis is academically interesting but may be difficult to implement due to TFLite interpreter limitations in extracting intermediate activations. Consider it a stretch goal.

---

## 5. Dimension 3 — On-Device Performance Benchmarking

### 5.1 Inference Latency

Measure the time taken for a single inference call on the Arduino:

```c
unsigned long t_start = micros();
interpreter->Invoke();
unsigned long t_end = micros();
float inference_ms = (t_end - t_start) / 1000.0;
```

**Measurement protocol:**
- Run 100 consecutive inferences (on the same input, or on different pre-loaded inputs)
- Report: mean, standard deviation, minimum, maximum latency
- Compare latency across models (M3 vs. M4 vs. M5 vs. M6) — all int8, so differences reflect architectural complexity

**Expected range:** 50–300 ms per inference, depending on model size.

### 5.2 Feature Extraction Latency

Separately measure the time for mel-spectrogram computation:

```c
unsigned long t_start = micros();
compute_mel_spectrogram(audio_buffer, AUDIO_SAMPLES, mel_spectrogram, N_MELS, N_FRAMES);
unsigned long t_end = micros();
float feature_ms = (t_end - t_start) / 1000.0;
```

**Expected range:** 30–100 ms for 92 frames of FFT + mel filterbank.

### 5.3 Total Cycle Time

The complete cycle from audio capture start to classification result:

```
Total = Audio Capture (1500 ms) + Feature Extraction (~50 ms) + Inference (~150 ms) + Output (~1 ms)
     ≈ 1700 ms per classification cycle
```

### 5.4 Flash and SRAM Usage

Read from the Arduino IDE compile output:

```
Sketch uses XXXXX bytes (XX%) of program storage space. Maximum is 1,048,576 bytes.
Global variables use XXXXX bytes (XX%) of dynamic memory. Maximum is 262,144 bytes.
```

**Report for each deployed model:**

| Model | Flash Used (KB) | Flash % | SRAM Used (KB) | SRAM % |
|-------|----------------|---------|----------------|--------|
| M3 (2-D CNN, int8 PTQ, Tier 2) | — | —% | — | —% |
| M4 (2-D CNN, int8 QAT, Tier 2) | — | —% | — | —% |
| M5 (1-D CNN, int8 QAT, Tier 2) | — | —% | — | —% |
| M6 (DS-CNN, int8 QAT, Tier 2) | — | —% | — | —% |

### 5.5 Tensor Arena Size Profiling

Determine the minimum tensor arena size that allows `AllocateTensors()` to succeed:

```c
// Binary search for minimum arena size
int arena_min = 1024;
int arena_max = 131072;  // 128 KB

while (arena_max - arena_min > 256) {
    int mid = (arena_min + arena_max) / 2;
    uint8_t* test_arena = (uint8_t*)malloc(mid);
    if (test_arena) {
        tflite::MicroInterpreter test_interp(model, resolver, test_arena, mid);
        if (test_interp.AllocateTensors() == kTfLiteOk) {
            arena_max = mid;
        } else {
            arena_min = mid;
        }
        free(test_arena);
    }
}
Serial.print("Minimum tensor arena size: ");
Serial.print(arena_max);
Serial.println(" bytes");
```

This gives the exact minimum SRAM requirement for the model, which is more precise than the estimation in [model_architecture_spec.md](model_architecture_spec.md).

### 5.6 Power Consumption (Optional/Stretch)

If a current measurement setup is available (e.g., a Nordic Power Profiler Kit, or a multimeter in series with the power supply):

- **Idle power:** Power consumption when waiting for audio capture
- **Active power (inference):** Power consumption during feature extraction + inference
- **Average power per cycle:** Weighted average over a full capture-classify cycle

**Estimated range:** ~5–15 mA at 3.3V for the nRF52840 during active processing = ~16–50 mW.

This is a stretch measurement — it requires hardware instrumentation beyond the Arduino itself.

---

## 6. Dimension 4 — End-to-End System Evaluation

### 6.1 Audio Playback Test

The most realistic evaluation available without an actual vehicle: play back dataset audio through a speaker and let the Arduino's on-board microphone capture and classify it.

**Setup:**
1. Place the Arduino Nano 33 BLE Sense Rev2 at a fixed distance (e.g., 10 cm) from a speaker
2. Play test set audio files through the speaker at a consistent volume
3. Record the Arduino's predictions via Serial output
4. Compare against known ground truth labels

**This test captures:**
- Microphone frequency response effects
- Speaker frequency response effects
- Ambient noise in the test environment
- Any training-serving skew in feature extraction

### 6.2 Playback Test Protocol

1. Set up a quiet test environment (minimize ambient noise)
2. Calibrate speaker volume to a consistent level (use a phone decibel meter app, target ~60–70 dB at the mic)
3. For each test audio file:
   a. Trigger the Arduino to start listening (e.g., via a Serial command or a button press)
   b. Play the audio file through the speaker
   c. Record the Arduino's prediction and confidence
   d. Wait 2 seconds between files
4. Compute the same metrics as D1 (accuracy, F1, confusion matrix)
5. Compare **PC test accuracy** vs. **playback test accuracy** — the difference reveals the domain gap introduced by the speaker-microphone chain and ambient noise

### 6.3 Expected Accuracy Degradation

Playback test accuracy is expected to be **5–20% lower** than PC test accuracy due to:
- Speaker frequency response (may attenuate or amplify certain frequencies)
- Microphone frequency response (the MP34DT06JTR has its own response curve)
- Room acoustics (reverb, reflections)
- Background noise (HVAC, computer fans, etc.)

If the degradation exceeds 20%, the training pipeline needs more aggressive noise augmentation or the playback setup needs improvement.

### 6.4 Robustness Tests (Optional)

If time permits, test the system under degraded conditions:

| Test | Setup | Purpose |
|------|-------|---------|
| **Noise robustness** | Play audio mixed with recorded ambient noise (road noise, wind) at various SNRs | Simulates real-world noise conditions |
| **Distance robustness** | Vary speaker-to-mic distance (5 cm, 10 cm, 20 cm, 50 cm) | Tests robustness to mic placement variability |
| **Volume robustness** | Vary speaker volume (quiet, normal, loud) | Tests robustness to amplitude variation |
| **Multi-fault response** | Play combined-fault audio files (excluded from Tier 3 training) | Tests whether the model activates appropriate fault classes for multi-fault audio |

---

## 7. Statistical Significance and Confidence

### 7.1 Confidence Intervals

With small test sets (~200 samples), point estimates of accuracy can be unreliable. Report **95% confidence intervals** for all metrics using the Wilson score interval for proportions:

```python
from statsmodels.stats.proportion import proportion_confint

correct = np.sum(y_pred == y_test)
total = len(y_test)
ci_low, ci_high = proportion_confint(correct, total, alpha=0.05, method='wilson')
print(f"Accuracy: {correct/total:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])")
```

For a test set of ~200 samples with 85% accuracy, the 95% CI is approximately [79.7%, 89.3%] — a ±5% margin. This is important context for interpreting small accuracy differences between models.

### 7.2 McNemar's Test for Model Comparison

When comparing two models on the same test set, use **McNemar's test** to determine whether the difference is statistically significant. McNemar's test uses the 2×2 contingency table of correct/incorrect predictions:

|  | Model B Correct | Model B Wrong |
|--|----------------|--------------|
| **Model A Correct** | n₁₁ | n₁₀ |
| **Model A Wrong** | n₀₁ | n₀₀ |

The test statistic is:
```
χ² = (|n₁₀ - n₀₁| - 1)² / (n₁₀ + n₀₁)
```

If p < 0.05, the models perform significantly differently. This is especially important when comparing float32 vs. int8 models — a 1% accuracy drop on 200 test samples may not be statistically significant.

### 7.3 Cross-Validation for Classical ML

For M1 (Random Forest, SVM), report the cross-validation accuracy (5-fold) in addition to the test set accuracy. The CV score provides a more robust estimate because it uses all training data:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
print(f"CV F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 8. Reporting Templates

### 8.1 Master Comparison Table (Per Tier)

This is the central result table. One table per tier.

**Tier 2 (6-class) — Example Template:**

| Metric | M1 (RF) | M1 (SVM) | M2 (CNN, F32) | M3 (CNN, PTQ) | M4 (CNN, QAT) | M5 (1D-CNN, QAT) | M6 (DS-CNN, QAT) |
|--------|---------|----------|---------------|--------------|--------------|------------------|------------------|
| **PC Evaluation** | | | | | | | |
| Accuracy | — | — | — | — | — | — | — |
| F1 (macro) | — | — | — | — | — | — | — |
| F1 (weighted) | — | — | — | — | — | — | — |
| **Quantization Metrics** | | | | | | | |
| ΔAccuracy (vs F32) | N/A | N/A | baseline | — | — | — | — |
| ΔF1 (vs F32) | N/A | N/A | baseline | — | — | — | — |
| Prediction agreement | N/A | N/A | baseline | —% | —% | —% | —% |
| **Size & Speed** | | | | | | | |
| Model size (KB) | N/A | N/A | — | — | — | — | — |
| Inference latency (ms) | N/A | N/A | N/A | — | — | — | — |
| Flash usage (KB) | N/A | N/A | N/A | — | — | — | — |
| SRAM usage (KB) | N/A | N/A | N/A | — | — | — | — |
| Tensor arena (KB) | N/A | N/A | N/A | — | — | — | — |
| **Playback Test** | | | | | | | |
| Accuracy (live) | N/A | N/A | N/A | — | — | — | — |
| F1 (live, macro) | N/A | N/A | N/A | — | — | — | — |

### 8.2 Per-Class Performance Table (Per Model)

| Class | Support | Precision | Recall | F1 | F32 Recall | ΔRecall (quant) |
|-------|---------|-----------|--------|-----|-----------|----------------|
| Normal Braking | 12 | — | — | — | — | — |
| Braking Fault | 12 | — | — | — | — | — |
| Normal Idle | 40 | — | — | — | — | — |
| Idle Fault | 118 | — | — | — | — | — |
| Normal Start-Up | 10 | — | — | — | — | — |
| Start-Up Fault | 18 | — | — | — | — | — |

### 8.3 Quantization Impact Summary Table

| Architecture | Tier | Accuracy (F32) | Accuracy (PTQ) | Accuracy (QAT) | ΔPTQ | ΔQAT | QAT Recovery |
|-------------|------|---------------|----------------|----------------|------|------|-------------|
| 2-D CNN | 1 | — | — | — | — | — | — |
| 2-D CNN | 2 | — | — | — | — | — | — |
| 2-D CNN | 3 | — | — | — | — | — | — |
| 1-D CNN | 1 | — | — | — | — | — | — |
| 1-D CNN | 2 | — | — | — | — | — | — |
| 1-D CNN | 3 | — | — | — | — | — | — |
| DS-CNN | 1 | — | — | — | — | — | — |
| DS-CNN | 2 | — | — | — | — | — | — |
| DS-CNN | 3 | — | — | — | — | — | — |

Where:
- **ΔPTQ** = Accuracy_F32 − Accuracy_PTQ
- **ΔQAT** = Accuracy_F32 − Accuracy_QAT
- **QAT Recovery** = ΔPTQ − ΔQAT (how much accuracy QAT recovers vs. PTQ)

### 8.4 On-Device Benchmarking Table

| Model | Tier | Model Size (KB) | Tensor Arena (KB) | Total Flash (KB) | Total SRAM (KB) | Inference (ms) | Feature Ext. (ms) | Total Cycle (ms) |
|-------|------|----------------|--------------------|------------------|-----------------|----------------|-------------------|-----------------|
| M3 | 2 | — | — | — | — | — | — | — |
| M4 | 2 | — | — | — | — | — | — | — |
| M5 | 2 | — | — | — | — | — | — | — |
| M6 | 2 | — | — | — | — | — | — | — |

### 8.5 Visualizations to Produce

| Visualization | Type | Purpose |
|--------------|------|---------|
| Confusion matrices | Heatmap | Per model, per tier — shows error patterns |
| Normalized confusion matrices | Heatmap | Per-class accuracy rates, comparable across classes |
| Training curves | Line plot | Loss + accuracy vs. epoch (train and val) — shows convergence and overfitting |
| Model size vs. accuracy scatter | Scatter plot | Trade-off visualization across all models |
| Latency vs. accuracy scatter | Scatter plot | Trade-off visualization for deployed models |
| Quantization accuracy drop bar chart | Bar chart | ΔPTQ and ΔQAT for each architecture + tier combination |
| Per-class quantization sensitivity | Grouped bar chart | Which classes lose the most accuracy from quantization |
| Confidence distributions | Histogram overlay | Float32 vs. int8 confidence distributions |
| ROC curve (Tier 1) | Line plot | Binary classification performance |
| Feature importance (M1) | Horizontal bar chart | Which audio features are most discriminative |

---

## 9. Evaluation Execution Checklist

Use this checklist to ensure all evaluations are completed systematically:

### Phase 1: PC Evaluation (After Training)
- [ ] Evaluate all M1 variants (RF, SVM) on test set — all tiers
- [ ] Evaluate M2 (float32 CNN) on test set — all tiers
- [ ] Record training curves (loss, accuracy vs. epoch) for all neural networks
- [ ] Compute per-class classification reports for all models

### Phase 2: Quantization Evaluation (After Quantization)
- [ ] Evaluate M3 (int8 PTQ) on test set using TFLite interpreter — all tiers
- [ ] Evaluate M4 (int8 QAT) on test set using TFLite interpreter — all tiers
- [ ] Evaluate M5 (1-D CNN, int8 QAT) on test set — all tiers
- [ ] Evaluate M6 (DS-CNN, int8 QAT) on test set — all tiers
- [ ] Compute all quantization impact metrics (ΔAccuracy, ΔF1, agreement rate, confidence shift)
- [ ] Generate all confusion matrices (float32 and int8 side-by-side)
- [ ] Record model file sizes for all variants

### Phase 3: On-Device Evaluation (After Deployment)
- [ ] Flash each deployable model to the Arduino
- [ ] Measure inference latency (100 repetitions, mean ± std)
- [ ] Measure feature extraction latency
- [ ] Record flash and SRAM usage from Arduino IDE
- [ ] Determine minimum tensor arena size
- [ ] Run playback test (at least test set audio files)
- [ ] Compute playback test accuracy and compare to PC accuracy

### Phase 4: Reporting
- [ ] Fill in all reporting templates
- [ ] Generate all visualizations
- [ ] Compute confidence intervals for key accuracy numbers
- [ ] Run McNemar's test for key model comparisons (float32 vs. int8)
- [ ] Write analysis and conclusions
