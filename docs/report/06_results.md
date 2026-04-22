# Results — Report Assembly

## 6.1 Master Comparison Table — Tier 2 (Primary Deployment Target)

| Rank | Model | Type | Condition | Accuracy | F1 Macro | F1 Weighted | Size (KB) |
|------|-------|------|-----------|----------|----------|-------------|-----------|
| 1 | **M6 DS-CNN PTQ** | **Int8** | **B** | **0.8702** | **0.7835** | **0.8709** | **21.3** |
| 2 | M6 DS-CNN QAT | Int8 | B | 0.8750 | 0.7774 | 0.8720 | 22.3 |
| 3 | M6 DS-CNN | Float32 | B | 0.8702 | 0.7767 | 0.8660 | 246 |
| 4 | M2 2D-CNN | Float32 | D | 0.8750 | 0.7512 | 0.8729 | 237 |
| 5 | M2 2D-CNN QAT | Int8 | D | 0.8702 | 0.7426 | 0.8688 | 20.5 |
| 6 | M2 2D-CNN PTQ | Int8 | D | 0.8510 | 0.7413 | 0.8624 | 20.1 |
| 7 | M1 SVM | Classical | A | 0.8510 | 0.6992 | 0.8422 | — |
| 8 | M1 RF | Classical | A | 0.8413 | 0.6794 | 0.8272 | — |
| 9 | M5 1D-CNN | Float32 | B | 0.8029 | 0.6377 | 0.8100 | 140 |
| 10 | M5 1D-CNN PTQ | Int8 | B | 0.8125 | 0.6138 | 0.8115 | 15.2 |

**Note:** M2 has higher accuracy than M6 but lower F1 macro due to class imbalance — M2 tends to over-predict the majority class (Idle Fault).

**Figure:** `results/paper_master_accuracy_tier2.png`
**Source:** `results/quantization_summary.csv`, `results/nn_summary_condition_{b,d}.csv`, `results/classical_ml_summary.csv`

## 6.2 Classical ML Results (All Tiers)

| Model | Tier | Accuracy | F1 Macro | F1 Weighted | CV F1 Macro |
|-------|------|----------|----------|-------------|-------------|
| RF | 1 | 0.8990 | 0.8724 | 0.8965 | 0.8899 |
| SVM | 1 | 0.9327 | 0.9163 | 0.9316 | 0.9244 |
| RF | 2 | 0.8413 | 0.6794 | 0.8272 | 0.7246 |
| SVM | 2 | 0.8510 | 0.6992 | 0.8422 | 0.7753 |
| RF | 3 | 0.7622 | 0.7021 | 0.7521 | 0.6880 |
| SVM | 3 | 0.7552 | 0.7112 | 0.7463 | 0.7091 |
| RF (top-50) | 2 | 0.8125 | 0.6724 | 0.8124 | — |

**Source:** `results/classical_ml_summary.csv`

## 6.3 Neural Network Ablation (Tier 2 F1 Macro by Condition)

| Architecture | Cond B (Synthetic) | Cond C (Noise) | Cond D (Combined) | Best |
|-------------|-------------------|----------------|-------------------|------|
| M2 2D-CNN | 0.7315 | 0.7335 | **0.7512** | D |
| M5 1D-CNN | **0.6377** | 0.5956 | 0.6252 | B |
| M6 DS-CNN | **0.7767** | 0.7388 | 0.7415 | B |

**Source:** `results/nn_summary_condition_b.csv`, `results/nn_summary_condition_c.csv`, `results/nn_summary_condition_d.csv`
**Figure:** `results/paper_ablation_tier2.png`

## 6.4 Quantization Results (All Tier 2 Models)

| Model | Method | Float32 F1 | Int8 F1 | Delta F1 (F32−Int8) | Agreement | McNemar p | Size (KB) |
|-------|--------|-----------|---------|---------------------|-----------|-----------|-----------|
| M2 | PTQ | 0.7438 | 0.7413 | +0.0025 | 99.0% | > 0.05 | 20.1 |
| M2 | QAT | 0.7438 | 0.7426 | +0.0012 | 95.7% | > 0.05 | 20.5 |
| M5 | PTQ | 0.5987 | 0.6138 | -0.0150 | 99.0% | > 0.05 | 15.2 |
| M5 | QAT* | 0.5987 | 0.6138 | -0.0150 | 99.0% | > 0.05 | 15.2 |
| M6 | PTQ | 0.7826 | 0.7835 | -0.0009 | 98.6% | > 0.05 | 21.3 |
| M6 | QAT | 0.7826 | 0.7774 | +0.0051 | 95.2% | > 0.05 | 22.3 |

*M5 QAT = M5 PTQ (tfmot does not support Conv1D, so QAT row is identical to PTQ)

**Source:** `results/quantization_summary.csv`, `results/quantization_results.json`
**Figure:** `results/paper_quantization_impact.png`, `results/paper_f1_macro_vs_size.png`

## 6.5 On-Device Performance (M6 DS-CNN PTQ)

### Memory Usage

| Resource | Used | Allocated / Budget | Utilization |
|----------|------|-------------------|-------------|
| Flash | 319 KB | 983 KB (budget) | 33% |
| SRAM (static) | 175 KB | 256 KB (budget) | 68% |
| Tensor arena | 63,556 B | 65,536 B (allocated) | 97.0% of allocated; 77.5% of 80 KB budget |
| SRAM remaining | 81 KB | — | For stack/heap |

### Latency

| Phase | Mean (ms) | Std (ms) | Share |
|-------|-----------|----------|-------|
| Audio capture | 1,490 | — | 60.3% |
| Feature extraction | 197.4 | 0.2 | 8.0% |
| Model inference | 784.9 | 0.1 | 31.7% |
| **Total cycle** | **2,472.5** | **0.7** | **100%** |

### Computational Efficiency

| Metric | Value |
|--------|-------|
| Total MACs per inference | 6,005,952 |
| Total cycles (measured) | 50.1 million |
| Cycles per MAC | 8.4 |
| Reference int8 (theoretical) | ~10 cycles/MAC |
| CMSIS-NN SIMD (theoretical) | ~3 cycles/MAC |

**Source:** `docs/deployment_results.md` Sections 7-8
**Figures:** `results/paper_latency_breakdown.png`, `results/paper_memory_utilization.png`

## 6.6 Playback Test Results

### Overall Metrics

| Metric | PC Int8 | Bluetooth Speaker | Laptop Speakers |
|--------|---------|-------------------|-----------------|
| Accuracy | 0.8702 | 0.6827 | 0.2500 |
| F1 Macro | 0.7835 | 0.5465 | 0.1540 |
| F1 Weighted | 0.8709 | 0.7162 | 0.2825 |
| Degradation (F1) | — | -30.3% | -80.3% |

### Per-Class Recall (Bluetooth Speaker vs PC)

| Class | n (test) | PC Int8 | Bluetooth | Delta |
|-------|----------|---------|-----------|-------|
| Normal Braking | 12 | 0.750 | 0.333 | -0.417 |
| Braking Fault | 11 | 0.818 | 0.818 | 0.000 |
| Normal Idle | 40 | 0.675 | 0.575 | -0.100 |
| Idle Fault | 118 | 0.864 | 0.780 | -0.084 |
| Normal Start-Up | 9 | 0.778 | 0.444 | -0.334 |
| Start-Up Fault | 18 | 0.556 | 0.556 | 0.000 |

### Confidence Calibration (Bluetooth)

| Category | Count (%) | Mean Confidence |
|----------|-----------|-----------------|
| Correct predictions | 142 (68.3%) | 0.808 |
| Incorrect predictions | 66 (31.7%) | 0.655 |

**Source:** `results/playback_test_results_bluetooth_speaker.json`, `results/playback_test_summary_bluetooth_speaker.csv`
**Figures:** `results/paper_pc_vs_playback_recall.png`, `results/paper_speaker_ablation.png`, `results/paper_confidence_calibration.png`, `results/cm_playback_tier_2.png`, `results/cm_playback_tier_2_norm.png`

## 6.7 Feature Parity Validation

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Cosine similarity (dB spectrogram) | 1.0000 | >= 0.99 | PASS |
| Cosine similarity (normalized) | 1.0000 | >= 0.99 | PASS |
| MAE (dB spectrogram) | 0.016 | — | — |
| MAE (int8 quantized) | 0.04 | <= 2.0 | PASS |
| Int8 exact match rate | 96.3% +/- 2.2% | — | — |

**Source:** `results/feature_parity_results.json`

## 6.8 Complete Figure Index

| Figure | File | Description |
|--------|------|-------------|
| Master accuracy (Tier 2) | `results/paper_master_accuracy_tier2.png` | All models ranked by F1 macro, horizontal bar chart |
| Ablation (Tier 2) | `results/paper_ablation_tier2.png` | Conditions B/C/D x 3 architectures, grouped bar |
| Quantization impact | `results/paper_quantization_impact.png` | Delta F1 by model/method, horizontal bar |
| F1 macro vs size | `results/paper_f1_macro_vs_size.png` | F1 vs model size scatter, all int8 Tier 2 |
| Latency breakdown | `results/paper_latency_breakdown.png` | Stacked bar + pie chart |
| Memory utilization | `results/paper_memory_utilization.png` | Flash/SRAM usage vs budget |
| PC vs playback recall | `results/paper_pc_vs_playback_recall.png` | Per-class recall, 3 bars (PC/BT/laptop) |
| Speaker ablation | `results/paper_speaker_ablation.png` | Prediction distributions by speaker type |
| Confidence calibration | `results/paper_confidence_calibration.png` | Histogram split by correct/incorrect |
| Feature importance | `results/feature_importance_tier2.png` | Top RF features bar chart |
| Confusion matrices | `results/cm_*.png` | ~50 files: per-model, per-tier, raw + normalized |
| Training curves | `results/training_curves_*.png` | Loss/accuracy per epoch for all NN models |
| ROC curves | `results/roc_*.png` | Tier 1 ROC for all models |
