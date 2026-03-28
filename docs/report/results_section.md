# 5. Results

This section presents the quantitative results of all experiments conducted across the four evaluation dimensions defined in Section 3: classification performance on PC (D1), quantization impact (D2), on-device performance benchmarking (D3), and end-to-end system evaluation via playback testing (D4). Unless otherwise noted, all results are reported on the held-out test set (n = 208) and use macro-averaged F1-score as the primary comparison metric, following the rationale established in the evaluation plan: macro F1 weights all classes equally regardless of support, penalizing models that achieve high accuracy by over-predicting the majority class.

## 5.1 Overall Model Comparison (Tier 2)

Tier 2 — the six-class, state-aware taxonomy — serves as the primary deployment target throughout this study. Table 1 ranks all evaluated models by F1 macro on the Tier 2 test set.

**Table 1.** Master comparison of all models evaluated at Tier 2 (six-class state-aware classification), ranked by macro-averaged F1-score.

| Rank | Model | Type | Training Condition | Accuracy | F1 Macro | F1 Weighted | Size (KB) |
|------|-------|------|--------------------|----------|----------|-------------|-----------|
| 1 | M6 DS-CNN PTQ | Int8 | B | 0.8702 | 0.7835 | 0.8709 | 21.3 |
| 2 | M6 DS-CNN QAT | Int8 | B | 0.8750 | 0.7774 | — | 22.3 |
| 3 | M6 DS-CNN | Float32 | B | 0.8702 | 0.7767 | — | 246 |
| 4 | M2 2D-CNN | Float32 | D | 0.8750 | 0.7512 | 0.8729 | 237 |
| 5 | M2 2D-CNN QAT | Int8 | D | 0.8702 | 0.7426 | — | 20.5 |
| 6 | M2 2D-CNN PTQ | Int8 | D | 0.8654 | 0.7413 | — | 20.1 |
| 7 | SVM | Classical | — | 0.8510 | 0.6992 | 0.8422 | — |
| 8 | Random Forest | Classical | — | 0.8413 | 0.6794 | 0.8272 | — |
| 9 | M5 1D-CNN | Float32 | B | 0.8029 | 0.6377 | — | 140 |
| 10 | M5 1D-CNN PTQ | Int8 | B | 0.8125 | 0.6138 | — | 15.2 |

<!-- Figure: results/paper_master_accuracy_tier2.png — Horizontal bar chart ranking all models by F1 macro -->

The depthwise separable CNN (M6 DS-CNN) achieves the highest F1 macro across all model types and quantization methods. Its post-training quantized variant (PTQ, Int8) reaches an F1 macro of 0.7835, making it the top-ranked model overall and the selected candidate for on-device deployment. The M2 2D-CNN achieves a comparable raw accuracy (0.8750 vs. 0.8702), but its lower F1 macro (0.7512) reveals a tendency to over-predict Idle Fault, the majority class (n = 118 of 208 test samples). This disparity between accuracy and F1 macro underscores why macro-averaged F1 is the more appropriate primary metric for this imbalanced dataset.

The classical ML baselines (SVM and Random Forest) are competitive on overall accuracy (0.8510 and 0.8413, respectively) but lag substantially behind the neural networks on F1 macro, indicating poorer performance on minority classes such as Normal Start-Up (n = 9) and Braking Fault (n = 11). The M5 1D-CNN, despite being the most compact architecture (15.2 KB quantized), ranks lowest among all models, suggesting that the MFCC feature representation discards spectral information important for distinguishing fault conditions at this granularity.

## 5.2 Classical ML Baselines Across Tiers

To contextualize the neural network results, Table 2 reports the classical ML baseline performance across all three taxonomy tiers.

**Table 2.** Classical ML performance by taxonomy tier. CV F1 Macro reports five-fold stratified cross-validation on the combined training and validation sets.

| Model | Tier | Accuracy | F1 Macro | F1 Weighted | CV F1 Macro |
|-------|------|----------|----------|-------------|-------------|
| SVM | 1 | 0.9327 | 0.9163 | 0.9316 | 0.9244 |
| Random Forest | 1 | 0.8990 | 0.8724 | 0.8965 | 0.8899 |
| SVM | 2 | 0.8510 | 0.6992 | 0.8422 | 0.7753 |
| Random Forest | 2 | 0.8413 | 0.6794 | 0.8272 | 0.7246 |
| SVM | 3 | 0.7552 | 0.7106 | 0.7494 | 0.7917 |
| Random Forest | 3 | 0.7622 | 0.7019 | 0.7522 | 0.7778 |
| Random Forest (top-50) | 2 | 0.8125 | 0.6724 | 0.8124 | — |

Both models perform well on the binary Tier 1 task (SVM F1 macro = 0.9163), confirming that the dataset contains a strong normal-vs-anomaly signal in the hand-crafted feature space. Performance degrades substantially at Tier 2, where the gap between accuracy and F1 macro widens — a signature of class-imbalanced misclassification. Interestingly, at Tier 3 (nine classes), the SVM achieves a slightly higher F1 macro (0.7112) than at Tier 2 (0.6992), likely because the Tier 3 formulation excludes combined-fault samples that conflate multiple acoustic signatures into one class. A reduced Random Forest trained on the top-50 features by importance did not improve over the full 441-feature model at Tier 2 (F1 macro 0.6724 vs. 0.6794), suggesting that the feature space is not dominated by irrelevant dimensions.

## 5.3 Data Augmentation Ablation

Three training conditions were compared for each neural network architecture at Tier 2 to assess the impact of data augmentation strategy. Condition B applies waveform augmentations (time shift, Gaussian noise, pitch shift, speed perturbation, random gain) with class-balancing to bring minority classes to approximately 550 samples each (3,317 total). Condition C adds real-world noise injection at SNR levels of 5–20 dB using 360 noise samples recorded on the Arduino hardware. Condition D combines the augmented samples from both B and C (5,664 total).

**Table 3.** F1 macro by training condition and architecture, Tier 2 test set.

| Architecture | Cond B (Synthetic) | Cond C (Noise) | Cond D (Combined) | Best |
|-------------|-------------------|----------------|-------------------|------|
| M6 DS-CNN | **0.7767** | 0.7388 | 0.7415 | B |
| M2 2D-CNN | 0.7315 | 0.7335 | **0.7512** | D |
| M5 1D-CNN | **0.6377** | 0.5956 | 0.6252 | B |

<!-- Figure: results/paper_ablation_tier2.png — Grouped bar chart of F1 macro by condition and architecture -->

The results reveal architecture-dependent interactions with the augmentation strategy. M6 DS-CNN and M5 1D-CNN perform best under Condition B (synthetic augmentation only), while M2 2D-CNN benefits from the larger combined dataset of Condition D. The noise-injected Condition C does not consistently improve clean-test-set performance for any architecture, though it may confer robustness benefits in noisy deployment environments that the clean test set does not capture. M5 shows the sharpest degradation under noise augmentation (F1 macro drops from 0.6377 to 0.5956 under Condition C), consistent with the hypothesis that MFCCs — which operate on a compressed cepstral representation — may already discard frequency bands where real-world noise provides useful regularization signal.

Based on these results, each architecture was carried forward under its best condition for quantization: M6 and M5 under Condition B, and M2 under Condition D.

## 5.4 Quantization Impact

All three neural network architectures were quantized to int8 using both post-training quantization (PTQ) and quantization-aware training (QAT). Table 4 summarizes the quantization results at Tier 2.

**Table 4.** Quantization impact on Tier 2 F1 macro. Agreement denotes the percentage of test samples receiving identical predictions from the float32 and int8 models. McNemar's test assesses whether the disagreements are statistically significant.

| Model | Method | Float32 F1 | Int8 F1 | ΔF1 | Agreement | McNemar p | Size (KB) |
|-------|--------|-----------|---------|-----|-----------|-----------|-----------|
| M6 DS-CNN | PTQ | 0.7826 | 0.7835 | −0.0009 | 98.6% | > 0.05 | 21.3 |
| M6 DS-CNN | QAT | 0.7826 | 0.7774 | +0.0051 | 95.2% | > 0.05 | 22.3 |
| M2 2D-CNN | PTQ | 0.7438 | 0.7413 | +0.0025 | 99.0% | > 0.05 | 20.1 |
| M2 2D-CNN | QAT | 0.7438 | 0.7426 | +0.0012 | 95.7% | > 0.05 | 20.5 |
| M5 1D-CNN | PTQ | 0.5987 | 0.6138 | −0.0150 | 99.0% | > 0.05 | 15.2 |
| M5 1D-CNN | QAT* | 0.5987 | 0.6138 | −0.0150 | 99.0% | > 0.05 | 15.2 |

*M5 QAT is identical to M5 PTQ because the TensorFlow Model Optimization Toolkit (tfmot) does not support quantization-aware training for Conv1D layers, so QAT could not be applied to M5.

<!-- Figure: results/paper_quantization_impact.png — Horizontal bar chart of delta F1 by model and method -->
<!-- Figure: results/paper_accuracy_vs_size.png — Scatter plot of F1 vs. model size for all int8 models -->

The central finding is that int8 quantization is effectively lossless for these small audio classification models. No model–method combination exhibits a statistically significant accuracy change (all McNemar p > 0.05), and prediction agreement rates range from 95.2% to 99.0%. No model shows substantial F1 degradation; the largest observed decrease is 0.0051 (M6 QAT). M5 PTQ actually shows a marginal F1 improvement of 0.015 (int8 exceeds float32) — a phenomenon attributable to the regularization effect of reduced numerical precision in small networks. Int8 model sizes range from 15.2 KB to 22.3 KB, representing a 6–16× compression from their float32 counterparts, and all fit comfortably within the 150 KB flash budget of the Arduino Nano 33 BLE Sense Rev2.

These results indicate that PTQ is sufficient for deployment — QAT provides no measurable benefit over PTQ for any architecture, likely because the small model sizes (6,246–14,566 parameters) and batch normalization folding leave minimal quantization error even without retraining.

## 5.5 On-Device Performance

The selected deployment model (M6 DS-CNN PTQ, Tier 2) was deployed on the Arduino Nano 33 BLE Sense Rev2 running TensorFlow Lite for Microcontrollers with CMSIS-NN optimized kernels. Table 5 reports memory utilization and Table 6 reports inference latency.

**Table 5.** Memory utilization of M6 DS-CNN PTQ on the Arduino Nano 33 BLE Sense Rev2.

| Resource | Used | Budget | Utilization |
|----------|------|--------|-------------|
| Flash | 319 KB | 983 KB | 33% |
| SRAM (static) | 175 KB | 256 KB | 68% |
| Tensor arena | 63,556 B | 65,536 B allocated (80 KB budget) | 97.0% of allocated; 77.5% of budget |
| SRAM remaining | 81 KB | — | Available for stack/heap |

The model consumes only 33% of available flash, leaving substantial room for application logic, BLE stack, and potential multi-model deployment. SRAM utilization is tighter: the tensor arena occupies 63.6 KB of the 80 KB budget (77.5%), and total static SRAM usage reaches 68% of the 256 KB available. The remaining 81 KB of SRAM provides adequate headroom for stack and heap operations during inference.

**Table 6.** Inference latency breakdown for M6 DS-CNN PTQ on the Arduino Nano 33 BLE Sense Rev2.

| Phase | Mean (ms) | Std (ms) | Share |
|-------|-----------|----------|-------|
| Audio capture | 1,490 | — | 60.3% |
| Feature extraction | 197.4 | 0.2 | 8.0% |
| Model inference | 784.9 | 0.1 | 31.7% |
| **Total cycle** | **2,472.5** | **0.7** | **100%** |

<!-- Figure: results/paper_latency_breakdown.png — Stacked bar and pie chart of latency phases -->
<!-- Figure: results/paper_memory_utilization.png — Flash and SRAM usage vs. budget -->

A complete classification cycle takes approximately 2.5 seconds, dominated by the 1.49-second audio capture window (1.5 seconds of audio at 16 kHz). Model inference requires 784.9 ms, yielding 8.4 cycles per multiply-accumulate (MAC) operation over the model's 6,005,952 MACs per inference. This is below the theoretical reference of approximately 10 cycles/MAC for generic int8 operations on Cortex-M4 but above the approximately 3 cycles/MAC achievable with fully optimized CMSIS-NN SIMD kernels. The gap is attributable to the older GCC 7.2.1 toolchain bundled with the Arduino Mbed OS board package, which does not fully exploit the Cortex-M4's SIMD capabilities. Upgrading to GCC 12+ via PlatformIO or the nRF Connect SDK could reduce inference latency to approximately 282 ms, bringing the total cycle time below 2 seconds. Nevertheless, for the target use case — a handheld diagnostic tool held near the engine for several seconds — a 2.5-second cycle is operationally acceptable.

## 5.6 End-to-End Playback Evaluation

The end-to-end system evaluation (D4) assessed real-world inference by playing the test set audio through external speakers and classifying with the Arduino's on-board PDM microphone. Three playback configurations were tested: direct PC audio output (baseline), a Bluetooth speaker, and laptop speakers.

**Table 7.** Overall classification metrics by playback configuration.

| Metric | PC Int8 (Baseline) | Bluetooth Speaker | Laptop Speakers |
|--------|---------------------|-------------------|-----------------|
| Accuracy | 0.7837 | 0.6827 | 0.2500 |
| F1 Macro | 0.7835 | 0.5465 | 0.1540 |
| F1 Weighted | 0.7835 | 0.7162 | 0.2825 |
| Degradation (F1 Macro) | — | −30.3% | −80.3% |

<!-- Figure: results/paper_speaker_ablation.png — Prediction distributions by speaker type -->

The Bluetooth speaker configuration — the most realistic proxy for field deployment — achieves 68.3% accuracy with an F1 macro of 0.5465, representing a 30.3% degradation from the PC baseline. Laptop speakers produced near-random performance (F1 macro = 0.1540), rendering them unsuitable for this application. The Bluetooth speaker's relatively high F1 weighted (0.7162) compared to its F1 macro (0.5465) indicates that the degradation is concentrated in minority classes, a pattern confirmed by the per-class recall analysis in Table 8.

**Table 8.** Per-class recall for Bluetooth speaker playback versus PC baseline.

| Class | n (test) | PC Int8 Recall | Bluetooth Recall | Δ Recall |
|-------|----------|----------------|------------------|----------|
| Idle Fault | 118 | 0.864 | 0.780 | −0.084 |
| Normal Idle | 40 | 0.675 | 0.575 | −0.100 |
| Start-Up Fault | 18 | 0.556 | 0.556 | 0.000 |
| Normal Braking | 12 | 0.750 | 0.333 | −0.417 |
| Braking Fault | 11 | 0.818 | 0.818 | 0.000 |
| Normal Start-Up | 9 | 0.778 | 0.444 | −0.334 |

<!-- Figure: results/paper_pc_vs_playback_recall.png — Per-class recall with three bars per class (PC/BT/Laptop) -->

The playback degradation is highly class-dependent. Braking Fault and Start-Up Fault retain identical recall through the Bluetooth speaker (0.818 and 0.556, respectively), suggesting that their acoustic signatures are robust to the speaker–microphone domain shift. In contrast, Normal Braking and Normal Start-Up suffer the steepest recall drops (−0.417 and −0.334), likely because normal operating sounds have lower signal energy and subtler spectral features that are more susceptible to distortion from Bluetooth audio compression and speaker frequency response limitations.

### 5.6.1 Confidence Calibration

Table 9 reports the confidence calibration of predictions in the Bluetooth speaker configuration.

**Table 9.** Confidence calibration for Bluetooth speaker playback predictions.

| Category | Count (%) | Mean Confidence |
|----------|-----------|-----------------|
| Correct predictions | 142 (68.3%) | 0.808 |
| Incorrect predictions | 66 (31.7%) | 0.655 |

<!-- Figure: results/paper_confidence_calibration.png — Histogram of confidence scores split by correct/incorrect predictions -->

Correct predictions exhibit a meaningfully higher mean confidence (0.808) than incorrect predictions (0.655), indicating that the model's softmax outputs carry useful discriminative information even in the degraded playback condition. This gap suggests that a confidence-based filtering strategy — accepting only predictions above a threshold of approximately 0.70 — could reject a substantial proportion of incorrect predictions while retaining most correct ones, improving effective precision at the cost of reduced recall.

## 5.7 Feature Parity Validation

To ensure that the on-device mel-spectrogram computation (implemented in C using CMSIS-DSP on the Arduino) reproduces the training-time feature extraction (implemented in Python using librosa), a feature parity validation was conducted. Table 10 reports the results.

**Table 10.** Feature extraction parity between Python (librosa) and Arduino (CMSIS-DSP) implementations.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Cosine similarity (dB spectrogram) | 1.0000 | ≥ 0.99 | PASS |
| Cosine similarity (normalized) | 1.0000 | ≥ 0.99 | PASS |
| MAE (dB spectrogram) | 0.016 | — | — |
| MAE (int8 quantized) | 0.04 | ≤ 2.0 | PASS |
| Int8 exact match rate | 96.3% ± 2.2% | — | — |

All parity metrics pass their defined thresholds. The cosine similarity of 1.0000 between the Python and Arduino dB-scale spectrograms confirms that the C implementation faithfully reproduces the training-time feature extraction pipeline. After int8 quantization, 96.3% of feature values match exactly, with a mean absolute error of only 0.04 — well below the 2.0 threshold. These results establish that any accuracy differences observed in on-device evaluation are attributable to the acoustic domain shift (speaker–microphone path) rather than training–serving skew in the feature extraction pipeline.
