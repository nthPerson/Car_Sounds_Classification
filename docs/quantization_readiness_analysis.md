# Quantization Readiness Analysis — Model Ranking for Phase 4

**Date:** March 24, 2026
**Purpose:** Analyze and rank all trained float32 neural network models by their suitability for int8 quantization and deployment on the Arduino Nano 33 BLE Sense Rev2.
**Related specs:** `docs/project_spec/quantization_deployment_spec.md`, `docs/project_spec/model_architecture_spec.md`, `docs/project_spec/evaluation_plan.md`

---

## 1. Overview

Phase 3 produced 9 float32 models (3 architectures x 3 tiers) across 4 training conditions (A, B, C, D), yielding 36 model-condition combinations. Phase 4 will quantize these to int8 using both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT), then evaluate the accuracy-size-latency trade-offs.

This document identifies which model-condition combinations are the strongest candidates for quantization and ranks them by deployment potential.

---

## 2. Hardware Constraints (Absolute Limits)

All quantized models must fit within these non-negotiable hardware constraints of the Arduino Nano 33 BLE Sense Rev2 (nRF52840):

| Resource | Total Available | Reserved for System | Available for Model |
|----------|----------------|--------------------|--------------------|
| **Flash** | 1,024 KB | ~100-150 KB (TFLite Micro runtime, PDM, feature extraction) | **~150 KB model budget** |
| **SRAM** | 256 KB | ~80 KB (audio buffer, feature arrays, stack, PDM DMA) | **~80 KB tensor arena budget** |

Int8 quantization provides ~4x compression over float32, and int8 activations reduce tensor arena SRAM by ~4x compared to float32 inference.

---

## 3. Float32 Performance Summary (All Conditions)

### 3.1 F1 Macro by Model, Tier, and Condition

| Model | Tier | Cond. A (Original) | Cond. B (Synthetic) | Cond. C (Noise) | Cond. D (Combined) | Best |
|-------|------|-------------------|--------------------|--------------------|-------------------|------|
| M2 (2-D CNN) | 1 | 0.868 | **0.898** | 0.861 | **0.898** | B/D |
| M2 (2-D CNN) | 2 | 0.666 | 0.732 | 0.734 | **0.751** | **D** |
| M2 (2-D CNN) | 3 | 0.715 | 0.717 | 0.722 | **0.764** | **D** |
| M5 (1-D CNN) | 1 | **0.875** | 0.864 | 0.874 | 0.858 | A |
| M5 (1-D CNN) | 2 | 0.616 | **0.638** | 0.596 | 0.625 | B |
| M5 (1-D CNN) | 3 | 0.616 | **0.657** | 0.612 | 0.651 | B |
| M6 (DS-CNN) | 1 | 0.858 | 0.900 | 0.890 | **0.909** | **D** |
| M6 (DS-CNN) | 2 | 0.571 | **0.777** | 0.739 | 0.742 | **B** |
| M6 (DS-CNN) | 3 | 0.600 | 0.725 | **0.762** | 0.686 | **C** |

### 3.2 Best Condition per Architecture (Tier 2 — Primary Deployment Target)

| Architecture | Best Condition | Tier 2 F1 Macro | Runner-Up | Tier 2 F1 Macro |
|-------------|---------------|-----------------|-----------|-----------------|
| **M6 (DS-CNN)** | B (Synthetic) | **0.777** | D (Combined) | 0.742 |
| **M2 (2-D CNN)** | D (Combined) | **0.751** | C (Noise) | 0.734 |
| **M5 (1-D CNN)** | B (Synthetic) | **0.638** | D (Combined) | 0.625 |

---

## 4. Architecture-Level Quantization Analysis

### 4.1 M2 — Compact 2-D CNN (Mel-Spectrogram)

**Architecture:** 3-block Conv2D with BatchNorm, GlobalAvgPool, Dropout, Dense.

| Property | Value | Assessment |
|----------|-------|------------|
| Parameters (Tier 2) | 14,566 | Moderate |
| Float32 model size | 237 KB (.h5) / ~57 KB (weights only) | Well above 150 KB flash limit in float32 |
| **Estimated int8 size** | **~15 KB** | Excellent — 10x under budget |
| Est. int8 tensor arena | ~70-80 KB | Tight — at the edge of 80 KB budget |
| Best Tier 2 F1 | 0.751 (Cond. D) | Strong |

**Quantization-friendly properties:**
- BatchNormalization layers fold into preceding Conv2D during quantization (no additional ops)
- All operations are standard TFLite int8-supported ops (Conv2D, MaxPool2D, GlobalAvgPool2D, Dense)
- GlobalAveragePooling2D is well-suited to quantization (averaging reduces quantization noise)
- Dropout is removed at inference time (no quantization interaction)
- Moderate parameter count means fewer weights to approximate in int8

**Quantization risk factors:**
- Tensor arena is at the upper end of the 80 KB budget. The first Conv2D output (40x92x16 = 58,880 bytes in int8) plus the input tensor (40x92x1 = 3,680 bytes) = ~62 KB peak. With scratch buffers, this could approach or exceed 80 KB.
- Three MaxPool2D layers with stride 2 progressively reduce spatial dimensions, which means the early layers dominate SRAM usage. If the arena is too tight, reducing `n_mels` from 40 to 32 or the first-layer filters from 16 to 8 would be necessary modifications.
- As the largest model by parameter count, M2 has the most weights to quantize — more opportunity for cumulative rounding error across layers.

**PTQ vs QAT expectation:** The spec predicts 0.5-2% accuracy drop for PTQ and 0-1% for QAT. With 14.5K parameters and standard ops, M2 should quantize cleanly. QAT fine-tuning may be especially valuable here because the model's higher capacity means the straight-through estimator has more room to adjust.

### 4.2 M5 — 1-D CNN (MFCCs)

**Architecture:** 3-block Conv1D with BatchNorm, GlobalAvgPool1D, Dropout, Dense.

| Property | Value | Assessment |
|----------|-------|------------|
| Parameters (Tier 2) | 6,246 | Smallest |
| Float32 model size | 140 KB (.h5) / ~24 KB (weights only) | Under budget even in float32 |
| **Estimated int8 size** | **~6 KB** | Excellent — 25x under budget |
| Est. int8 tensor arena | ~15-25 KB | Excellent — well within budget |
| Best Tier 2 F1 | 0.638 (Cond. B) | Weakest |

**Quantization-friendly properties:**
- Smallest model — fewest parameters = least cumulative quantization error
- 1-D convolutions on MFCC input (92x13) have much smaller activations than 2-D convolutions on mel-spectrograms (40x92)
- Extremely small tensor arena requirement guarantees it will fit comfortably on-device
- Conv1D operations are well-supported by CMSIS-NN int8 kernels on Cortex-M4
- The MFCC input is inherently lower-dimensional (13 coefficients vs 40 mel bands), reducing the number of intermediate values that must be quantized

**Quantization risk factors:**
- Lower float32 accuracy (0.638) means there is less "accuracy margin" — even a small quantization-induced drop could push performance below acceptable thresholds
- Smaller models can paradoxically be more sensitive to quantization because each weight carries more information (less redundancy to absorb rounding errors)
- SpecAugment (used during M5 training) creates a regularization effect that may not transfer cleanly to QAT fine-tuning

**PTQ vs QAT expectation:** M5's small size and simple ops should make PTQ nearly lossless. QAT may not provide significant improvement over PTQ since there are fewer weights to adjust. However, the MFCC-based input pipeline requires different on-device feature extraction than mel-spectrograms, which adds deployment complexity.

### 4.3 M6 — Depthwise Separable CNN (DS-CNN)

**Architecture:** Initial Conv2D (4x10 stride-2) followed by 4 depthwise separable blocks (DepthwiseConv2D + pointwise Conv2D), GlobalAvgPool2D, Dense.

| Property | Value | Assessment |
|----------|-------|------------|
| Parameters (Tier 2) | 8,166 | Moderate-small |
| Float32 model size | 246 KB (.h5) / ~32 KB (weights only) | Over budget in float32, fine in int8 |
| **Estimated int8 size** | **~8 KB** | Excellent — 19x under budget |
| Est. int8 tensor arena | ~35-45 KB | Good — comfortable within budget |
| Best Tier 2 F1 | 0.777 (Cond. B) | **Highest** |

**Quantization-friendly properties:**
- DS-CNN architecture was specifically designed for efficient int8 deployment on microcontrollers (ARM ML Zoo heritage)
- Depthwise separable convolutions factorize standard convolutions into depthwise + pointwise operations, each of which quantizes independently. This means per-channel quantization can be applied at a finer granularity (each depthwise channel has its own scale/zero_point)
- The 32-channel variant was specifically selected for this project to fit within the SRAM budget
- CMSIS-NN has highly optimized kernels for depthwise separable convolutions on Cortex-M4 — this architecture will get the best hardware acceleration
- The initial large-kernel Conv2D (4x10, stride 2) immediately halves spatial dimensions, keeping all subsequent activations smaller
- GlobalAveragePooling2D at the end compresses well under quantization
- No dropout layer — one fewer source of training/inference behavioral mismatch

**Quantization risk factors:**
- Depthwise convolutions use fewer parameters per operation than standard convolutions. Each 3x3 depthwise filter has only 9 weights per channel, quantized to 9 int8 values. With only 256 possible int8 levels, the effective precision is limited — a concern when the filter must represent subtle spectral patterns
- The 4 DS blocks share the same 32-channel width throughout. If quantization noise accumulates through the depth of the network (4 sequential DS blocks), later blocks may see degraded feature quality. The straight-through architecture (no residual connections) means there is no "clean" signal path to compensate
- The initial 4x10 kernel is unusual and may have a wider weight distribution than typical 3x3 kernels. If the distribution has outliers, per-tensor quantization of that layer could waste dynamic range. Per-channel quantization mitigates this but should be verified.
- The highest float32 accuracy (0.777 on Tier 2) gives the most margin to absorb quantization loss

**PTQ vs QAT expectation:** DS-CNN architectures are known to benefit meaningfully from QAT over PTQ. The depthwise convolution layers are the most quantization-sensitive components, and QAT allows the model to adjust these specific weights to compensate for int8 rounding. We should expect PTQ to lose 1-3% and QAT to recover most of that gap.

---

## 5. Quantization Scope: What to Quantize

### 5.1 Required by the Spec (Development Roadmap Tasks 4.1-4.11)

The development roadmap and quantization spec mandate:

1. **PTQ for all architectures** (M2→M3, M5 PTQ, M6 PTQ) — all 3 tiers
2. **QAT for all architectures** (M2→M4, M5 QAT, M6 QAT) — all 3 tiers
3. **TFLite evaluation** of all quantized models
4. **Quantization impact analysis** (accuracy drop, F1 drop, agreement rate, confidence shift)
5. **Model size recording** for all variants
6. **Deployment model selection** with justification

This means **18 quantized models** minimum (3 architectures x 3 tiers x 2 methods).

### 5.2 Training Condition Selection

Since we have 4 training conditions and quantizing all 36 combinations x 2 methods = 72 models would be excessive, we should select the best condition per architecture:

| Architecture | Condition for Quantization | Rationale |
|-------------|---------------------------|-----------|
| **M2 (2-D CNN)** | **Condition D** (Combined, 5,664 samples) | Best Tier 2 F1 (0.751) and Tier 3 F1 (0.764). Benefits most from data volume. |
| **M5 (1-D CNN)** | **Condition B** (Synthetic, 3,317 samples) | Best Tier 2 F1 (0.638) and Tier 3 F1 (0.657). Noise augmentation hurts MFCC-based models. |
| **M6 (DS-CNN)** | **Condition B** (Synthetic, 3,317 samples) | Best Tier 2 F1 (0.777) overall. Class balance was the key improvement for DS-CNN. |

**Special case — M6 + Condition C for noise robustness:** M6 Condition C achieves the best Tier 3 F1 (0.762) and may offer better noise robustness for on-device deployment. If time permits, quantizing M6 Condition C as a secondary candidate for the playback test comparison would be informative.

This reduces the scope to **18 primary quantized models** (3 architectures x 3 tiers x PTQ + QAT) plus optionally 6 secondary models (M6 Cond. C x 3 tiers x PTQ + QAT).

---

## 6. Model Rankings

### 6.1 Overall Ranking for Tier 2 Deployment (Primary Target)

Combining float32 accuracy, quantization friendliness, and deployment feasibility:

| Rank | Model + Condition | Tier 2 F1 | Est. Int8 Size | Est. Arena | Quant. Risk | Deployment Fit |
|------|------------------|-----------|---------------|-----------|-------------|---------------|
| **1** | **M6 (DS-CNN) + Cond. B** | **0.777** | ~8 KB | ~35-45 KB | Low-Medium | Excellent |
| **2** | **M2 (2-D CNN) + Cond. D** | **0.751** | ~15 KB | ~70-80 KB | Low | Marginal (arena) |
| **3** | M6 (DS-CNN) + Cond. D | 0.742 | ~8 KB | ~35-45 KB | Low-Medium | Excellent |
| **4** | M6 (DS-CNN) + Cond. C | 0.739 | ~8 KB | ~35-45 KB | Low-Medium | Excellent |
| **5** | M2 (2-D CNN) + Cond. C | 0.734 | ~15 KB | ~70-80 KB | Low | Marginal (arena) |
| **6** | M5 (1-D CNN) + Cond. B | 0.638 | ~6 KB | ~15-25 KB | Low | Excellent |

### 6.2 Ranking Rationale

**Rank 1 — M6 + Condition B** is the clear front-runner:
- Highest Tier 2 F1 macro (0.777), 2.6 percentage points above the next architecture
- DS-CNN architecture was designed for quantized microcontroller deployment
- Small model size (~8 KB) and comfortable tensor arena (~35-45 KB) leave significant headroom
- CMSIS-NN has optimized depthwise separable convolution kernels for Cortex-M4
- Condition B (synthetic augmentation with class balancing) produced the most stable and highest performance for this architecture

**Rank 2 — M2 + Condition D** is the accuracy alternative:
- Second-highest Tier 2 F1 (0.751) with the strongest Tier 3 performance (0.764)
- Standard 2-D CNN operations quantize predictably and cleanly
- The concern is the tensor arena: ~70-80 KB estimated for int8 is at the edge of the 80 KB budget. This will be validated during Phase 4 — if the arena exceeds 80 KB, M2 may need architectural modifications (reduce first-layer filters from 16 to 8 or reduce n_mels from 40 to 32) which could degrade accuracy
- The int8 model itself (~15 KB) fits easily in flash

**Rank 6 — M5 + Condition B** is the safe fallback:
- Lowest accuracy but by far the smallest footprint
- Guaranteed to fit on-device with room to spare (~6 KB model, ~20 KB arena)
- If both M6 and M2 fail to deploy (unlikely), M5 is the insurance policy
- The MFCC input requires different on-device feature extraction code than mel-spectrograms, adding implementation effort. However, MFCCs are computationally cheaper to extract (fewer mel bins, no 2-D structure)

### 6.3 Cross-Tier Performance Context

While Tier 2 is the primary target, Tier 1 and Tier 3 performance matters for the research paper's completeness:

**Tier 1 (binary — normal vs. anomaly):**
- M6 Cond. D leads at 0.909 F1 macro
- All architectures perform well (0.858-0.909) — binary classification is straightforward
- Quantization impact on Tier 1 is expected to be minimal

**Tier 3 (9-class — individual fault identification):**
- M2 Cond. D leads at 0.764 F1 macro
- M6 Cond. C is close at 0.762 — noise augmentation particularly helps the fine-grained task
- Tier 3 is the most sensitive to quantization because it requires the model to distinguish between acoustically similar fault conditions (e.g., Low Oil vs. Power Steering Fault). Small precision losses from quantization could blur these distinctions.

---

## 7. Expected Quantization Outcomes

### 7.1 Model Size Predictions

| Model | Float32 (weights) | Int8 (predicted) | Compression | Flash Budget (150 KB) |
|-------|-------------------|-----------------|-------------|----------------------|
| M2 (2-D CNN, Tier 2) | ~57 KB | ~15-18 KB | ~3.5x | 10-12% used |
| M5 (1-D CNN, Tier 2) | ~24 KB | ~6-8 KB | ~3.5x | 4-5% used |
| M6 (DS-CNN, Tier 2) | ~32 KB | ~8-10 KB | ~3.5x | 5-7% used |

Note: The compression ratio is ~3.5x rather than 4x because TFLite models include metadata, quantization parameters (scale/zero_point per channel), and op graph structure that does not compress.

### 7.2 Accuracy Drop Predictions

Based on the architecture analysis and literature on similar models:

| Model | Float32 F1 | PTQ F1 (predicted) | QAT F1 (predicted) | PTQ Drop | QAT Drop |
|-------|-----------|--------------------|--------------------|----------|----------|
| M2 (Tier 2, Cond. D) | 0.751 | ~0.73-0.74 | ~0.74-0.75 | 1-2% | 0-1% |
| M5 (Tier 2, Cond. B) | 0.638 | ~0.62-0.63 | ~0.63-0.64 | 0.5-1.5% | 0-0.5% |
| M6 (Tier 2, Cond. B) | 0.777 | ~0.75-0.77 | ~0.76-0.78 | 0.5-2.5% | 0-1.5% |

These are estimates only. M6's depthwise convolutions add some uncertainty to the PTQ prediction.

### 7.3 Inference Latency Predictions

On the Cortex-M4F at 64 MHz with CMSIS-NN int8 kernels:

| Model | Estimated Inference Latency | Notes |
|-------|-----------------------------|-------|
| M2 (2-D CNN) | ~150-250 ms | Standard Conv2D operations, moderate activation size |
| M5 (1-D CNN) | ~30-80 ms | Smallest activations, fewest operations |
| M6 (DS-CNN) | ~80-150 ms | Depthwise separable ops are faster per FLOP due to CMSIS-NN optimization |

Combined with feature extraction (~50-100 ms) and audio capture (1,500 ms), total cycle times should be:
- M5: ~1,580-1,680 ms
- M6: ~1,630-1,750 ms
- M2: ~1,700-1,850 ms

All well under 2 seconds per classification, which is acceptable for real-time use.

---

## 8. Recommended Phase 4 Plan

### 8.1 Quantization Order (Priority)

1. **M6 (DS-CNN) + Condition B** — PTQ then QAT, all 3 tiers. Highest priority as the top deployment candidate.
2. **M2 (2-D CNN) + Condition D** — PTQ then QAT, all 3 tiers. Second priority; will determine whether the tensor arena fits.
3. **M5 (1-D CNN) + Condition B** — PTQ then QAT, all 3 tiers. Lower priority but needed for completeness.

### 8.2 Key Decisions That Phase 4 Will Resolve

1. **Does M2's tensor arena fit in 80 KB?** If not, M2 drops in rank and may need architectural changes or be excluded from deployment.
2. **How much accuracy does M6 lose from quantization?** If PTQ drop is >2%, QAT becomes critical. If QAT doesn't recover the loss, M6's advantage over M2 shrinks.
3. **Is M5's accuracy sufficient post-quantization?** At 0.638 F1, even a 1% drop puts it below the classical SVM baseline (0.699). This may affect whether M5 is worth deploying.
4. **PTQ vs. QAT gap:** The spec's central research question is whether QAT recovers accuracy lost by PTQ. The three architectures will provide three data points for this comparison.

### 8.3 Evaluation Deliverables (from Evaluation Plan)

For each quantized model, Phase 4 should produce:
- Accuracy, F1 macro, F1 weighted on the test set (via TFLite interpreter)
- Per-class precision/recall/F1 (for quantization sensitivity analysis)
- Prediction agreement rate with float32 baseline
- Confidence distribution comparison (float32 vs. int8)
- Model file size (`.tflite`)
- Quantization impact summary table (the central artifact for the research paper)

---

## 9. Summary

The three architectures occupy distinct positions in the accuracy-efficiency trade-off:

| Architecture | Strength | Weakness | Phase 4 Question |
|-------------|----------|----------|-------------------|
| **M6 (DS-CNN)** | Highest accuracy, best hardware acceleration | DS blocks may be quantization-sensitive | Does int8 maintain the lead? |
| **M2 (2-D CNN)** | Consistent accuracy, predictable quantization | Tensor arena may exceed budget | Does it fit in 80 KB? |
| **M5 (1-D CNN)** | Smallest footprint, guaranteed to fit | Accuracy already below SVM baseline | Is it accurate enough to be useful? |

**The expected outcome is that M6 + Condition B remains the deployment champion after quantization**, with M2 + Condition D as the backup if M6 quantizes poorly, and M5 + Condition B as the safe fallback. Phase 4 will confirm or overturn this prediction.
