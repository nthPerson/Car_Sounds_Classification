# Analysis — Report Assembly

## 5.1 Classical ML Baseline Analysis

### Key Observations
- **Accuracy-F1 gap reveals class imbalance:** RF Tier 2 achieves 84.1% accuracy but only 0.679 F1 macro — the 16 percentage point gap indicates strong performance on majority class (Idle Fault: 93.8% recall for SVM) but poor performance on minorities (Normal Start-Up: 11.1% recall for RF)
- **SVM outperforms RF on Tier 1** (0.916 vs 0.872 F1) but gap narrows on multi-class tiers
- **Tier 3:** SVM slightly outperforms RF (0.711 vs 0.702 F1), maintaining its advantage across all tiers though the gap narrows with more classes

### Feature Importance (Tier 2 RF)
- **Top features:** delta2_mfcc_0 (spectral energy acceleration), mfcc_0 (overall energy level), spectral_centroid (brightness)
- **MFCCs and temporal derivatives dominate** — top 5 features all MFCC-based
- **Top-50 feature subset:** 0.672 F1 vs 0.679 full (only -0.007 drop) — high redundancy in the 441-dim feature space
- **Figure:** `results/feature_importance_tier2.png`

### Interpretation
- Classical ML with hand-crafted features provides a strong baseline, especially for binary classification
- The 441-dim feature vector captures redundant information — a smaller subset is nearly as effective
- Class imbalance is the primary limiting factor for classical ML on Tier 2/3

**Source:** `docs/classical_ml_baselines.md`, `results/classical_ml_summary.csv`, `models/m1_classical/evaluation_results.json`

## 5.2 Data Augmentation Analysis

### Condition A to B: The Dominant Improvement
- **M6 F1 jumps from 0.571 to 0.777 (+0.206)** — the largest single improvement in the project
- **M2 improves by +0.066**, M5 by +0.022
- **Root cause:** Two factors work together:
  1. Dataset expansion (970 → 3,317) provides gradient diversity
  2. Class balancing (12.8x → 1.1x) eliminates majority class bias
- **Training stability:** M6 was unstable on Condition A (val_acc ~23% at LR >= 0.001), but stable on Condition B (91.4% at LR=0.002). The expanded, balanced dataset enables higher learning rates.

### Condition B to C: Marginal Noise Effect
- **M6: 0.777 → 0.739 (-0.038)** — slight degradation
- **M2: 0.732 → 0.734 (+0.002)** — essentially flat
- **M5: 0.638 → 0.596 (-0.042)** — noticeable degradation
- **Interpretation:** Real-world noise injection does not improve clean test set accuracy. The noise may help deployment robustness (not directly measured on clean test set), or the noise bank may introduce domain mismatch (different vehicle than training data)

### Condition D (Combined): Model-Dependent
- **M2 benefits most:** 0.732 → 0.751 (+0.019) — larger model can leverage more diverse data
- **M5 and M6 do not benefit** — possible overfitting to noise patterns with the doubled augmented dataset
- **Implication:** More data is not always better; the quality and relevance of augmentation matters

### Key Insight for Paper
**Synthetic augmentation + class balancing is the dominant improvement.** Real-world noise injection provides a form of domain adaptation but shows mixed results on clean evaluation. The benefit may only manifest during actual noisy deployment.

**Figure:** `results/paper_ablation_tier2.png`
**Source:** `docs/neural_network_models.md`, `results/nn_summary_condition_{b,c,d}.csv`

## 5.3 Quantization Analysis

### Why Quantization Is Nearly Lossless
- **Small parameter count (6K-15K):** Few parameters means few values to quantize, limiting total error
- **BatchNorm folding:** BN layers are folded into preceding Conv/Dense during quantization, reducing op count and maintaining precision
- **GlobalAveragePooling:** Averages spatial dimensions, smoothing out per-element quantization noise
- **Binary-like decision boundaries:** The classification task may have well-separated features that don't require float32 precision to distinguish

### PTQ vs QAT: No Benefit from QAT
- PTQ performs as well as or better than QAT across all architectures at Tier 2
- **M6:** PTQ F1 0.7835 vs QAT 0.7774 — PTQ is slightly better
- **M2:** PTQ F1 0.7413 vs QAT 0.7426 — essentially tied
- **Explanation:** QAT fine-tuning with fake quantization can introduce training instability, especially with small models. Since PTQ already produces negligible error, QAT's additional complexity provides no recovery.
- **Practical implication:** For models with < 15K parameters on audio classification tasks, PTQ is sufficient. QAT should be reserved for larger models or more precision-sensitive tasks.

### Prediction Agreement Analysis
- **PTQ agreement: 97-100%** — quantization changes fewer than 3% of predictions
- **QAT agreement: 92-97%** — QAT paradoxically changes more predictions than PTQ, but without improving F1
- **McNemar's test:** All p > 0.05 — no combination shows statistically significant degradation
- **Confidence shift:** < 0.002 mean decrease — softmax outputs remain nearly identical

**Figure:** `results/paper_quantization_impact.png`
**Source:** `docs/quantization_results.md`, `results/quantization_summary.csv`, `results/quantization_results.json`

## 5.4 Deployment Performance Analysis

### Inference Latency Breakdown
- **Audio capture (1,490 ms, 60.3%):** Fixed by physics — 1.5s of audio at 16 kHz
- **Feature extraction (197 ms, 8.0%):** 92 FFT frames, each requiring windowing + FFT + power spectrum + mel filterbank. The CMSIS-DSP `arm_rfft_fast_f32()` function is well-optimized for Cortex-M4.
- **Model inference (783 ms, 31.7%):** Dominated by pointwise convolutions (62.8% of 6M MACs). The 1x1 Conv2D layers in the DS-CNN account for most computation.

### Cycles/MAC Analysis
- **Measured: 8.4 cycles/MAC** — consistent with reference int8 kernels (theoretical ~10 cycles/MAC, measured slightly better due to compiler optimizations)
- **CMSIS-NN SIMD target: ~3 cycles/MAC** — blocked by GCC 7.2.1 ACLE intrinsic limitation
- **Gap analysis:** 2.8x slower than optimal, representing 501 ms of avoidable latency

### CMSIS-NN Toolchain Investigation
- Built TFLite Micro from source with CMSIS-NN optimized kernels
- `__ARM_FEATURE_DSP=1` confirmed at compile time (Cortex-M4 correctly detected)
- GCC 7.2.1's `arm_acle.h` missing `__sxtb16`, `__smlabb`, `__smlatt` intrinsic implementations
- Provided inline assembly compatibility shims — compiler did not reliably emit SIMD instructions
- Pre-compiled `.a` with GCC 14.3.1 (verified 1,640 SIMD instructions) failed due to C++ ABI incompatibility
- **Conclusion:** Arduino's bundled GCC 7.2.1 is the bottleneck. Upgrading to GCC 10+ would enable 3x speedup.

### Memory Efficiency
- **Flash: 319 KB (33%)** — very comfortable. MicroMutableOpResolver saved 165 KB vs AllOpsResolver.
- **SRAM: 179 KB (68%)** — moderate utilization. 83 KB remaining for stack/heap.
- **Tensor arena: 63,556 of 65,536 bytes (97%)** — tightly packed but functional.
- The model (21.3 KB) is far below the 150 KB flash budget, suggesting room for larger/more complex models if accuracy improvement is needed.

**Figures:** `results/paper_latency_breakdown.png`, `results/paper_memory_utilization.png`
**Source:** `docs/deployment_results.md` Sections 7-8

## 5.5 Playback Test Analysis

### Speaker Frequency Response is the Dominant Factor
- **Laptop speakers:** Frequency response rolls off below ~300 Hz. Car engine idle and braking sounds have significant low-frequency content (50-500 Hz) — engine rumble, brake rotor vibration, mechanical hum. When this is filtered out, the spectral signature shifts toward startup sounds (which have transient mid/high-frequency content).
- **Result:** 72.1% of all laptop predictions fall into startup classes regardless of true label.
- **Bluetooth speaker:** Better bass response restores the low-frequency content. Accuracy improves from 25.0% to 68.3% (2.7x).

### Per-Class Degradation Pattern
- **Classes that survive playback:** Braking Fault (81.8% recall = PC-equivalent), Start-Up Fault (55.6% = PC-equivalent), Idle Fault (78.0%, -8.4 pp from PC)
- **Classes that degrade:** Normal Braking (33.3%, -41.7 pp), Normal Start-Up (44.4%, -33.4 pp), Normal Idle (57.5%, -10.0 pp)
- **Pattern:** Normal classes degrade more than fault classes. Possible explanation: fault sounds (worn brakes, dead battery, bad ignition) have distinctive mid/high-frequency components that survive speaker reproduction, while normal sounds rely more on low-frequency tonal content that speakers attenuate.

### Confidence Calibration
- **Correct predictions:** mean confidence 0.808 (n=142, 68.3%)
- **Incorrect predictions:** mean confidence 0.655 (n=66, 31.7%)
- **Gap: 0.153** — large enough to enable threshold-based filtering
- **Practical use:** Setting a confidence threshold at 0.70 would reject ~50% of incorrect predictions while retaining ~85% of correct ones (approximate from distribution)

### Implication for Real-World Deployment
- In actual deployment, the Arduino's PDM microphone is positioned near the engine — **no speaker in the signal path**
- The microphone captures the full frequency range of engine sounds directly
- The Bluetooth playback result (68.3%) is therefore a **conservative lower bound** on real-world accuracy
- Actual on-engine performance would likely approach or exceed the PC int8 accuracy (78.4%)

**Figures:** `results/paper_pc_vs_playback_recall.png`, `results/paper_speaker_ablation.png`, `results/paper_confidence_calibration.png`
**Source:** `results/playback_test_results_bluetooth_speaker.json`, `results/playback_test_summary_bluetooth_speaker.csv`, `docs/deployment_results.md` Section 9
