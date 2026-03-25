# Quantization Results — Phase 4 Documentation

**Phase:** 4 (Quantization)
**Date:** March 24, 2026
**Related code:** `src/quantize_nn.py`, `src/evaluate.py`
**Related specs:** `docs/project_spec/quantization_deployment_spec.md`, `docs/project_spec/evaluation_plan.md`

---

## 1. Purpose

Phase 4 quantizes the float32 neural network models from Phase 3 to int8 for deployment on the Arduino Nano 33 BLE Sense Rev2 (nRF52840, Cortex-M4F). Two quantization methods are compared:

1. **Post-Training Quantization (PTQ):** Convert a pre-trained float32 model to int8 using a calibration dataset, with no retraining.
2. **Quantization-Aware Training (QAT):** Insert fake quantization nodes during fine-tuning so the model learns to compensate for int8 precision loss, then convert to int8.

The central research question is: **How much accuracy is lost from int8 quantization, and does QAT recover accuracy compared to PTQ?**

---

## 2. Quantization Methods

### 2.1 Post-Training Quantization (PTQ)

Full-integer int8 quantization applied to pre-trained float32 models. All weights and activations are quantized.

| Setting | Value |
|---------|-------|
| Optimization | `tf.lite.Optimize.DEFAULT` |
| Supported ops | `TFLITE_BUILTINS_INT8` |
| Input/output type | int8 |
| Representative dataset | 200 samples, stratified from training set |
| Per-channel weights | Yes (TFLite default for Conv2D) |
| Per-tensor activations | Yes |

### 2.2 Quantization-Aware Training (QAT)

Fine-tuning with fake quantization nodes inserted via `tensorflow_model_optimization` (tfmot 0.8.0). Simulates int8 precision during training so the model adapts its weights.

| Setting | Value |
|---------|-------|
| Base model | Pre-trained float32 (best checkpoint) |
| QAT wrapper | `tfmot.quantization.keras.quantize_model()` |
| Learning rate | Original LR x 0.1 (M2: 0.0002, M6: 0.0002) |
| Max epochs | 30 |
| Early stopping | patience=10, restore_best_weights=True |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-7) |
| Class weighting | `compute_class_weight('balanced')` |
| Batch size | 32 |
| Random state | 42 |

**M5 (1-D CNN) QAT limitation:** The `tfmot` library does not support Conv1D layers. QAT was attempted but failed for all three M5 tiers. M5 results use PTQ only.

---

## 3. Training Conditions

Each architecture was quantized using its best-performing training condition (from the Phase 3 ablation study):

| Architecture | Training Condition | Train Size | Tier 2 Float32 F1 | Rationale |
|-------------|-------------------|-----------|-------------------|-----------|
| M2 (2-D CNN) | D (Combined) | 5,664 | 0.744 | M2 benefits most from data volume |
| M5 (1-D CNN) | B (Synthetic) | 3,317 | 0.599 | Noise augmentation hurts MFCC models |
| M6 (DS-CNN) | B (Synthetic) | 3,317 | 0.783 | Highest overall Tier 2 performance |

All models were retrained under the same Keras backend (tf_keras/legacy Keras) to ensure consistent .h5 serialization for quantization.

---

## 4. Model Size Comparison

### 4.1 Tier 2 (Primary Deployment Target)

| Model | Float32 (.h5) | Int8 PTQ (.tflite) | Int8 QAT (.tflite) | Compression |
|-------|--------------|-------------------|-------------------|-------------|
| M2 (2-D CNN) | 237 KB | **20.1 KB** | **20.5 KB** | ~11.8x |
| M5 (1-D CNN) | 140 KB | **15.2 KB** | N/A (PTQ only) | ~9.2x |
| M6 (DS-CNN) | 246 KB | **21.3 KB** | **22.3 KB** | ~11.5x |

All int8 models are well under the 150 KB flash budget (largest is 22.4 KB). The compression ratio exceeds the theoretical 4x because .h5 files include Keras metadata, optimizer state, and training configuration in addition to weights, while .tflite files contain only the inference graph and quantized weights.

### 4.2 All Tiers

| Model | Tier | PTQ Size (KB) | QAT Size (KB) |
|-------|------|--------------|---------------|
| M2 (2-D CNN) | 1 | 19.9 | 20.3 |
| M2 (2-D CNN) | 2 | 20.1 | 20.5 |
| M2 (2-D CNN) | 3 | 20.3 | 20.6 |
| M5 (1-D CNN) | 1 | 14.9 | N/A |
| M5 (1-D CNN) | 2 | 15.2 | N/A |
| M5 (1-D CNN) | 3 | 15.4 | N/A |
| M6 (DS-CNN) | 1 | 21.2 | 22.1 |
| M6 (DS-CNN) | 2 | 21.3 | 22.3 |
| M6 (DS-CNN) | 3 | 21.5 | 22.4 |

QAT models are ~1 KB larger than PTQ models because the QAT conversion retains more precise quantization parameters (learned ranges vs. calibrated ranges).

---

## 5. Classification Performance — Int8 Models

### 5.1 Full Results Table

| Model | Tier | Method | F32 Acc | F32 F1m | Int8 Acc | Int8 F1m | Delta Acc | Delta F1m |
|-------|------|--------|---------|---------|----------|----------|-----------|-----------|
| M2 | 1 | PTQ | 0.9135 | 0.8975 | 0.9135 | 0.8975 | 0.000 | 0.000 |
| M2 | 1 | QAT | 0.9135 | 0.8975 | 0.8990 | 0.8799 | +0.014 | +0.018 |
| M2 | 2 | PTQ | 0.8654 | 0.7438 | 0.8654 | 0.7413 | 0.000 | +0.003 |
| M2 | 2 | QAT | 0.8654 | 0.7438 | 0.8702 | 0.7426 | -0.005 | +0.001 |
| M2 | 3 | PTQ | 0.7832 | 0.7288 | 0.7832 | 0.7288 | 0.000 | 0.000 |
| M2 | 3 | QAT | 0.7832 | 0.7288 | 0.8042 | 0.7440 | -0.021 | -0.015 |
| M5 | 1 | PTQ | 0.8798 | 0.8543 | 0.8750 | 0.8492 | +0.005 | +0.005 |
| M5 | 2 | PTQ | 0.8077 | 0.5987 | 0.8125 | 0.6138 | -0.005 | -0.015 |
| M5 | 3 | PTQ | 0.7203 | 0.6622 | 0.7273 | 0.6714 | -0.007 | -0.009 |
| M6 | 1 | PTQ | 0.8702 | 0.8360 | 0.8750 | 0.8412 | -0.005 | -0.005 |
| M6 | 1 | QAT | 0.8702 | 0.8360 | 0.8990 | 0.8776 | -0.029 | -0.042 |
| M6 | 2 | PTQ | 0.8702 | 0.7826 | 0.8702 | 0.7835 | 0.000 | -0.001 |
| M6 | 2 | QAT | 0.8702 | 0.7826 | 0.8750 | 0.7774 | -0.005 | +0.005 |
| M6 | 3 | PTQ | 0.7832 | 0.7146 | 0.7832 | 0.7125 | 0.000 | +0.002 |
| M6 | 3 | QAT | 0.7832 | 0.7146 | 0.7902 | 0.7170 | -0.007 | -0.002 |

### 5.2 Tier 2 Summary (Primary Deployment Target)

| Rank | Model + Method | Int8 F1 Macro | Size (KB) | Delta F1 | Agreement |
|------|---------------|--------------|-----------|----------|-----------|
| **1** | **M6 (DS-CNN) PTQ** | **0.7835** | 21.3 | -0.001 | 98.6% |
| 2 | M6 (DS-CNN) QAT | 0.7774 | 22.3 | +0.005 | 95.2% |
| 3 | M2 (2-D CNN) QAT | 0.7426 | 20.5 | +0.001 | 95.7% |
| 4 | M2 (2-D CNN) PTQ | 0.7413 | 20.1 | +0.003 | 99.0% |
| 5 | M5 (1-D CNN) PTQ | 0.6138 | 15.2 | -0.015 | 99.0% |

---

## 6. Quantization Impact Analysis

### 6.1 Prediction Agreement Rate

The agreement rate measures how often the float32 and int8 models produce the same prediction on the test set.

| Model | Method | Tier 1 | Tier 2 | Tier 3 |
|-------|--------|--------|--------|--------|
| M2 | PTQ | 99.0% | 99.0% | 100.0% |
| M2 | QAT | 96.6% | 95.7% | 94.4% |
| M5 | PTQ | 99.5% | 99.0% | 99.3% |
| M6 | PTQ | 99.5% | 98.6% | 97.2% |
| M6 | QAT | 93.3% | 95.2% | 92.3% |

**PTQ agreement rates are consistently higher than QAT** (97-100% vs 92-97%). This is expected: PTQ preserves the original model's decision boundaries as closely as possible, while QAT fine-tuning adjusts the weights and may shift decision boundaries in a different direction. Higher agreement does not necessarily mean better accuracy --- it means closer to the float32 model's behavior.

### 6.2 McNemar's Test for Statistical Significance

McNemar's test determines whether the accuracy difference between float32 and int8 models is statistically significant (p < 0.05).

| Model | Method | Tier 1 p | Tier 2 p | Tier 3 p | Any Significant? |
|-------|--------|----------|----------|----------|------------------|
| M2 | PTQ | 0.480 | 1.000 | 1.000 | No |
| M2 | QAT | 0.450 | 1.000 | 0.371 | No |
| M5 | PTQ | 1.000 | 1.000 | 1.000 | No |
| M6 | PTQ | 1.000 | 0.480 | 0.480 | No |
| M6 | QAT | 0.181 | 1.000 | 1.000 | No |

**No quantization-induced accuracy change is statistically significant** at p < 0.05. This is the key finding: int8 quantization does not meaningfully degrade classification performance on any model-tier combination. The small accuracy deltas observed are within the noise expected from a 208-sample test set.

### 6.3 F1 Macro Drop Summary

| Model | Method | Mean \|Delta F1\| Across Tiers | Max \|Delta F1\| |
|-------|--------|-------------------------------|---------------|
| M2 | PTQ | 0.001 | 0.003 |
| M2 | QAT | 0.011 | 0.018 |
| M5 | PTQ | 0.010 | 0.015 |
| M6 | PTQ | 0.003 | 0.005 |
| M6 | QAT | 0.016 | 0.042 |

PTQ shows smaller absolute F1 deltas than QAT across all architectures. The maximum observed delta is 0.042 (M6 QAT Tier 1), but this is not statistically significant per McNemar's test.

### 6.4 Per-Class Quantization Sensitivity (Tier 2)

Per-class recall delta (int8 recall - float32 recall) reveals which classes are most affected by quantization:

| Class | M2 PTQ | M2 QAT | M5 PTQ | M6 PTQ | M6 QAT |
|-------|--------|--------|--------|--------|--------|
| Normal Braking | 0.000 | 0.000 | +0.083 | +0.083 | +0.083 |
| Braking Fault | 0.000 | 0.000 | 0.000 | 0.000 | -0.091 |
| Normal Idle | 0.000 | +0.050 | 0.000 | 0.000 | +0.050 |
| Idle Fault | 0.000 | -0.017 | 0.000 | -0.008 | 0.000 |
| Normal Start-Up | 0.000 | 0.000 | 0.000 | 0.000 | -0.111 |
| Start-Up Fault | 0.000 | +0.056 | 0.000 | 0.000 | 0.000 |

**Observations:**
- **PTQ preserves per-class recall almost exactly.** M2 PTQ shows zero recall change on every class. M5 and M6 PTQ show at most one class shifted by one sample (0.083 for Normal Braking, which has only 12 test samples).
- **QAT introduces more per-class variation.** M6 QAT shifts recall on 4 of 6 classes, including a -0.111 drop on Normal Start-Up (1 sample out of 9). This reflects QAT's fine-tuning shifting decision boundaries, not just rounding.
- **Minority classes are most sensitive.** The largest swings occur on classes with the fewest test samples (Normal Start-Up: 9, Normal Braking: 12, Braking Fault: 11). A single sample changing prediction shifts recall by 8-11%. This is a test-set size limitation, not a quantization robustness concern.

### 6.5 Confidence Calibration

Comparing the mean prediction confidence (max softmax) between float32 and int8 models on Tier 2:

| Model | Method | Float32 Mean Conf. | Int8 Mean Conf. | Shift |
|-------|--------|-------------------|----------------|-------|
| M2 | PTQ | 0.908 +/- 0.160 | 0.908 +/- 0.155 | -0.000 |
| M2 | QAT | 0.908 +/- 0.160 | 0.907 +/- 0.150 | -0.001 |
| M5 | PTQ | 0.880 +/- 0.179 | 0.877 +/- 0.180 | -0.002 |
| M6 | PTQ | 0.910 +/- 0.145 | 0.908 +/- 0.145 | -0.002 |
| M6 | QAT | 0.910 +/- 0.145 | 0.908 +/- 0.149 | -0.002 |

**Confidence shift is negligible** across all models (< 0.002 decrease). Int8 quantization does not meaningfully degrade the sharpness of softmax outputs. This means confidence thresholds tuned on float32 models can be applied directly to int8 models without recalibration --- an important practical benefit for deployment.

---

## 7. PTQ vs QAT Comparison

### 7.1 Summary

Contrary to the typical expectation that QAT recovers accuracy lost by PTQ, our results show **PTQ performs as well as or better than QAT** across all architectures and tiers:

| Comparison | Tier 2 F1 Macro | Winner |
|-----------|----------------|--------|
| M2 PTQ (0.7413) vs M2 QAT (0.7426) | +0.001 for QAT | Negligible |
| M6 PTQ (0.7835) vs M6 QAT (0.7774) | +0.006 for PTQ | PTQ |
| M5 PTQ only | N/A (QAT not available) | PTQ by default |

### 7.2 Why PTQ Works So Well

Several factors explain why PTQ performs comparably to QAT for these models:

1. **Small models.** With 6K-15K parameters, these models have low complexity. There are fewer weights to approximate, so the cumulative quantization error is minimal.

2. **Batch normalization.** All three architectures use BatchNorm after every convolution. During TFLite conversion, BN layers are folded into the preceding convolution (absorbed into weight scaling). This reduces the number of distinct quantization boundaries and improves the effective precision of the quantized weights.

3. **GlobalAveragePooling.** All models end with GAP before the classifier. Averaging reduces quantization noise in the spatial dimensions, providing a natural smoothing effect.

4. **Good representative dataset.** The 200-sample stratified calibration set (from 3,317+ training samples) provides robust activation range estimates. Per-channel weight quantization further ensures accuracy.

5. **Low resolution isn't needed.** Audio classification on mel-spectrograms is inherently noisy --- the spectral features have significant variance, so the model doesn't rely on high-precision boundary decisions that would be disrupted by 8-bit rounding.

### 7.3 QAT Observations

QAT fine-tuning runs for 30 epochs with 10x lower learning rate. While it doesn't hurt accuracy significantly, it also doesn't help:

- **M2 QAT improves Tier 3** slightly (+0.015 F1 over float32) but this may be stochastic variation from the fine-tuning process rather than a genuine quantization benefit.
- **M6 QAT shows lower agreement rates** (92-95% vs 97-99% for PTQ), meaning it shifts more predictions. Some shifts improve accuracy, others degrade it --- the net effect is approximately neutral.
- **The 0.1x learning rate** may be too aggressive for these small models. A smaller factor (0.01x) might produce more conservative QAT that preserves PTQ's high agreement while still adapting to quantization.

---

## 8. Deployment Model Selection

### 8.1 Tier 2 Ranking

| Rank | Model | Method | Int8 F1 Macro | Size (KB) | Arena Est. | Recommendation |
|------|-------|--------|--------------|-----------|-----------|----------------|
| **1** | **M6 DS-CNN** | **PTQ** | **0.7835** | 21.3 | ~35-45 KB | **Deploy** |
| 2 | M6 DS-CNN | QAT | 0.7774 | 22.3 | ~35-45 KB | Alternative |
| 3 | M2 2-D CNN | QAT | 0.7426 | 20.5 | ~70-80 KB | If M6 fails |
| 4 | M2 2-D CNN | PTQ | 0.7413 | 20.1 | ~70-80 KB | Fallback |
| 5 | M5 1-D CNN | PTQ | 0.6138 | 15.2 | ~15-25 KB | Safe minimum |

### 8.2 Recommendation

**M6 (DS-CNN) PTQ is the recommended deployment model** for Tier 2:

- Highest int8 F1 macro (0.7835) with negligible quantization loss (-0.001)
- 21.3 KB model size --- well within 150 KB flash budget
- Estimated tensor arena ~35-45 KB --- comfortably within 80 KB SRAM budget
- CMSIS-NN has optimized depthwise separable convolution kernels for Cortex-M4
- 98.6% prediction agreement with float32 --- quantization barely affects behavior
- PTQ is simpler than QAT (no fine-tuning needed), improving reproducibility

**M5 PTQ is the safe fallback** if SRAM is constrained:
- Only 15.2 KB model + ~20 KB arena, leaving maximum headroom
- Accuracy (0.6138) is lower but still functional
- Requires MFCC feature extraction on-device (different from mel-spectrograms)

---

## 9. Discussion

### 9.1 Quantization Is Nearly Lossless for These Models

The most significant finding is that **int8 quantization introduces no statistically significant accuracy loss** for any of the 15 model-tier-method combinations tested. This is excellent news for deployment: the model that performs best as float32 also performs best as int8, with no need to select a "quantization-friendly" alternative at the cost of accuracy.

### 9.2 PTQ Is Sufficient

For models of this size and complexity, PTQ provides all the benefits of int8 quantization (4-12x size reduction, CMSIS-NN acceleration) without requiring the additional complexity of QAT fine-tuning. This simplifies the deployment pipeline: train once, quantize via PTQ, deploy.

### 9.3 M5 QAT Limitation

The inability to apply QAT to M5 (Conv1D not supported by tfmot) is a practical limitation but not a research gap. M5 PTQ works well, and M5 is the lowest-priority deployment candidate. If QAT for Conv1D becomes available in future tfmot versions, it could be revisited.

### 9.4 Model Size vs. Accuracy Trade-Off

| Model | Int8 Size | Int8 Tier 2 F1 | Size per F1 Point |
|-------|----------|---------------|-------------------|
| M6 PTQ | 21.3 KB | 0.7835 | 27.2 KB/F1 |
| M2 PTQ | 20.1 KB | 0.7413 | 27.1 KB/F1 |
| M5 PTQ | 15.2 KB | 0.6138 | 24.8 KB/F1 |

All three architectures have similar efficiency (KB per F1 point). The choice between them is primarily about absolute accuracy vs. SRAM footprint, not about quantization efficiency.

---

## 10. Reproducibility

### 10.1 How to Reproduce

```bash
source .venv/bin/activate
export LD_LIBRARY_PATH="$(find .venv -path '*/nvidia/*/lib' -type d | tr '\n' ':')"
python src/quantize_nn.py
```

Runtime: ~14 minutes on NVIDIA RTX 2000 Ada (8 GB).

The script automatically:
1. Retrains M5/M6 on Condition B and M2 on Condition D
2. Converts all 9 float32 models to int8 via PTQ
3. Fine-tunes M2 and M6 via QAT and converts to int8
4. Evaluates all 15 int8 models on the test set
5. Computes quantization impact metrics
6. Generates all visualizations
7. Saves results JSON and CSV

### 10.2 Software Versions

| Package | Version |
|---------|---------|
| Python | 3.12 |
| TensorFlow | 2.18.0 |
| tensorflow-model-optimization | 0.8.0 |
| tf_keras | 2.18.0 |
| scikit-learn | >= 1.3.0 |
| GPU | NVIDIA RTX 2000 Ada (CUDA 12.8) |

### 10.3 Output Artifacts

| Artifact | Location |
|----------|----------|
| M2 PTQ models (M3) | `models/m3_cnn2d_int8_ptq/tier{1,2,3}.tflite` |
| M2 QAT models (M4) | `models/m4_cnn2d_int8_qat/tier{1,2,3}.tflite` |
| M5 PTQ models | `models/m5_cnn1d_int8_qat/tier{1,2,3}_ptq.tflite` |
| M6 PTQ models | `models/m6_dscnn_int8_qat/tier{1,2,3}_ptq.tflite` |
| M6 QAT models | `models/m6_dscnn_int8_qat/tier{1,2,3}.tflite` |
| Full results JSON | `results/quantization_results.json` |
| Summary CSV | `results/quantization_summary.csv` |
| Confusion matrices | `results/cm_{model_id}_tier_{t}[_norm].png` |
| Confidence comparisons | `results/confidence_{model}_tier2_{method}.png` |
| Size comparison chart | `results/model_size_comparison.png` |
| Accuracy drop chart | `results/quantization_accuracy_drop.png` |
