# Final Validation Report — Paper Section Accuracy Check

**Date:** March 27, 2026
**Method:** 8 parallel validation agents, each cross-referencing one paper section against actual CSV, JSON, and source code files
**Scope:** All factual claims, numerical values, table entries, and comparative statements

---

## Summary

| Section | Status | Critical Issues | Minor Issues |
|---------|--------|----------------|--------------|
| Abstract | Issues Found | 1 (SRAM value) | 2 (model count ambiguity, inference time rounding) |
| Introduction | Issues Found | 0 | 3 (reference year, TinyML latency phrasing, RQ4 not in original spec) |
| Data Sources | **PASS** | 0 | 0 |
| Methodology | Issues Found | 3 (HP search results wrong for all 3 models) | 2 (M2 Block 3 MaxPool, feature ordering) |
| Analysis | Issues Found | 2 (QAT agreement range, accuracy/F1 confusion) | 3 (SRAM, inference rounding, Condition A unverifiable) |
| Results | Issues Found | 8 (Table 1 accuracies, Table 2 Tier 3 CV, Table 4 Float32 baselines) | 5 (SRAM, flash %, latency stds) |
| Discussion | Issues Found | 3 (M6 accuracy as 78.4%, Tier 3 wrong condition, Normal Start-Up F1) | 4 (agreement range, CMSIS-NN estimate, inference rounding, Idle Fault %) |
| Conclusion | Issues Found | 2 (max F1 delta wrong, agreement range) | 3 (2.7x metric, SRAM, inference rounding) |

---

## Cross-Section Recurring Issues

These errors appear in multiple sections and should be fixed consistently:

### Issue A: Accuracy vs F1 Macro Confusion (CRITICAL)

Multiple sections report M6 DS-CNN PTQ Tier 2 "accuracy" as 0.7837 or 78.4%, but the actual int8 accuracy is **0.8702** (87.0%). The value 0.7835/0.7837 is the **F1 macro**, not accuracy.

**Affected sections:** Abstract, Results (Table 1), Analysis (Section 4.5.4), Discussion (Section 7.1.5), Conclusion (Section 8.1)

**Correct values (M6 PTQ Tier 2):**
- Int8 Accuracy: **0.8702**
- Int8 F1 Macro: **0.7835**
- Int8 F1 Weighted: **0.8709**

**Same issue affects other models in Results Table 1:**
- M2 PTQ: accuracy should be 0.8654 (not 0.8510)
- M2 QAT: accuracy should be 0.8702 (not 0.8558)
- M5 PTQ: accuracy should be 0.8125 (not 0.7837)

**Source:** `results/quantization_summary.csv`

### Issue B: SRAM Reporting Inconsistency (MODERATE)

Multiple sections report "179 KB SRAM (68%)" but these values are inconsistent:
- Compiler reports: 178,688 bytes = **174.5 KiB** (using 1024-byte KB, consistent with flash reporting)
- 174.5 / 256 = **68.2%** (matches compiler's "68%")
- The "179 KB" value appears to use decimal KB (178,688 / 1000 = 178.7, rounded to 179)
- 179 / 256 = 69.9%, which does NOT match the compiler's 68%

**Recommendation:** Use **175 KB (68%)** throughout (consistent binary KB rounding, matches compiler output)

**Affected sections:** Abstract, Analysis, Results (Table 5), Conclusion

### Issue C: QAT Prediction Agreement Range (MODERATE)

Multiple sections claim QAT agreement is "92-97%" but the actual range is **92.3% to 99.5%**. M5 QAT produces identical results to M5 PTQ (Conv1D not supported by tfmot), so M5 QAT rows show 99.0-99.5% agreement.

**Recommendation:** State "92-100%" or clarify "92-97% for M2 and M6 (M5 QAT identical to PTQ)"

**Affected sections:** Analysis, Discussion, Conclusion

---

## Per-Section Issues

### 1. Abstract

| # | Issue | Claimed | Actual | Source | Severity |
|---|-------|---------|--------|--------|----------|
| A1 | SRAM value | 179 KB (68%) | 175 KB (68%) | deployment_results.md | Moderate |
| A2 | Model count | "six model architectures" then lists 5 | RF, SVM, 2D-CNN, 1D-CNN, DS-CNN = 5 distinct architectures | — | Minor |
| A3 | Inference time | 783 ms | 784.9 ms mean (playback test) | playback JSON | Minor |

### 2. Introduction

| # | Issue | Claimed | Actual | Source | Severity |
|---|-------|---------|--------|--------|----------|
| I1 | Tsoukas et al. year | 2024 | Sensors vol 25 = 2025 | Citation | Minor |
| I2 | TinyML latency | "0-5 ms" | Physically impossible for 0 ms; project measures 783 ms | evaluation_plan.md | Minor |
| I3 | RQ4 (augmentation) | Listed as research question | Not in original project spec (project_overview.md lists 3 RQs) | project_overview.md | Minor |

### 3. Data Sources

**VALIDATED — no issues found.** All dataset statistics, split sizes, feature parameters, augmentation counts, and noise bank details confirmed correct.

### 4. Methodology

| # | Issue | Claimed | Actual | Source | Severity |
|---|-------|---------|--------|--------|----------|
| M1 | M2 best HP | LR 0.002, dropout 0.3, aggressive SpecAugment | LR 0.002, dropout **0.2**, augmentation **"none"** | hyperparameter_search.json | Critical |
| M2 | M5 best HP | LR 0.002, dropout 0.2, no SpecAugment | LR **0.001**, dropout **0.3**, augmentation **"default"** | hyperparameter_search.json | Critical |
| M3 | M6 best HP | LR 0.0005, default SpecAugment | LR **0.002**, augmentation **"none"** | hyperparameter_search.json | Critical |
| M4 | M2 Block 3 | "each block" has MaxPool | Block 3 has no MaxPool (goes to GlobalAvgPool) | src/models.py | Minor |
| M5 | Feature ordering | contrast before rolloff | rolloff before contrast in code | src/features.py | Minor |

### 5. Analysis

| # | Issue | Claimed | Actual | Source | Severity |
|---|-------|---------|--------|--------|----------|
| AN1 | QAT agreement range | 92-97% | 92.3-99.5% | quantization_summary.csv | Moderate |
| AN2 | "PC-evaluated int8 result of 78.4%" | 78.4% accuracy | 87.0% accuracy; 78.4% is F1 macro | quantization_summary.csv | Critical |
| AN3 | SRAM | 179 KB (68%) | 175 KB (68%) | deployment_results.md | Moderate |
| AN4 | Inference latency | 783 ms | 784.9 ms mean | playback JSON | Minor |
| AN5 | Condition A baseline | M6 F1=0.571 | No condition_a results file exists | — | Minor (unverifiable) |

### 6. Results

| # | Issue | Claimed | Actual | Source | Severity |
|---|-------|---------|--------|--------|----------|
| R1 | Table 1: M6 PTQ accuracy | 0.7837 | **0.8702** | quantization_summary.csv | Critical |
| R2 | Table 1: M6 PTQ F1 weighted | 0.7835 | **0.8709** | quantization_results.json | Critical |
| R3 | Table 1: M2 QAT accuracy | 0.8558 | **0.8702** | quantization_summary.csv | Critical |
| R4 | Table 1: M2 PTQ accuracy | 0.8510 | **0.8654** | quantization_summary.csv | Critical |
| R5 | Table 1: M5 PTQ accuracy | 0.7837 | **0.8125** | quantization_summary.csv | Critical |
| R6 | Table 2: SVM Tier 3 CV F1 | 0.7091 | **0.7917** | classical_ml_summary.csv | Critical |
| R7 | Table 2: RF Tier 3 CV F1 | 0.6880 | **0.7778** | classical_ml_summary.csv | Critical |
| R8 | Table 2: SVM Tier 3 F1 weighted | 0.7463 | **0.7494** | classical_ml_summary.csv | Minor |
| R9 | Table 4: M6 PTQ Float32 F1 | 0.7844 | **0.7826** | quantization_summary.csv | Moderate |
| R10 | Table 4: M6 QAT Float32 F1 | 0.7723 | **0.7826** | quantization_summary.csv | Moderate |
| R11 | Table 4: M2 PTQ Float32 F1 | 0.7388 | **0.7438** | quantization_summary.csv | Moderate |
| R12 | Table 4: M2 QAT Float32 F1 | 0.7414 | **0.7438** | quantization_summary.csv | Moderate |
| R13 | Table 4: M5 PTQ/QAT Float32 F1 | 0.6288 | **0.5987** | quantization_summary.csv | Critical |
| R14 | Table 4 prose: "largest F1 degradation is 0.015 (M5)" | Degradation | Actually an **improvement** (int8 > float32) | quantization_summary.csv | Moderate |
| R15 | Table 5: SRAM | 179 KB / 69.9% | 175 KB / 68% | deployment_results.md | Moderate |
| R16 | Table 5: Flash % | 32.5% | 33% | deployment_results.md | Minor |
| R17 | Table 5: Internal inconsistency | 179 + 83 = 262 KB | Should sum to 256 KB | — | Minor |
| R18 | Table 6: Inference std | 0.2 ms | 0.1 ms (Bluetooth) | playback JSON | Minor |
| R19 | Table 6: Total std | 0.4 ms | 0.7 ms (Bluetooth) | playback JSON | Minor |

### 7. Discussion

| # | Issue | Claimed | Actual | Source | Severity |
|---|-------|---------|--------|--------|----------|
| D1 | M6 "lower accuracy (78.4%)" | 78.4% accuracy | **87.0%** accuracy; 78.4% is F1 | quantization_summary.csv | Critical |
| D2 | Tier 3: "M6 (F1 0.762)" | F1 0.762 (Condition C) | Deployed model is Condition B: F1 **0.715** | quantization_summary.csv | Critical |
| D3 | Normal Start-Up F1 | 0.375 | **0.200** (precision 0.129, recall 0.444) | playback confusion matrix | Critical |
| D4 | QAT agreement range | 92-97% | 92.3-99.5% | quantization_summary.csv | Moderate |
| D5 | CMSIS-NN speedup estimate | ~260 ms | ~282 ms (per deployment_results.md) | deployment_results.md | Minor |
| D6 | Inference latency | 783 ms | 784.9 ms | playback JSON | Minor |
| D7 | Idle Fault training % | 56.7% | 56.9% | preprocessing_config.json | Minor |

### 8. Conclusion & Future Work

| # | Issue | Claimed | Actual | Source | Severity |
|---|-------|---------|--------|--------|----------|
| C1 | Max F1 delta | -0.015 (M5 PTQ) | -0.042 (M6 QAT Tier 1) | quantization_summary.csv | Critical |
| C2 | Agreement range | 95-99% | 92.3-100% | quantization_summary.csv | Moderate |
| C3 | "2.7x improvement in F1" | 2.7x F1 | 2.7x is **accuracy** (68.3/25.0); F1 improvement is 3.55x | playback JSONs | Moderate |
| C4 | SRAM | 179 KB (68%) | 175 KB (68%) | deployment_results.md | Moderate |

---

## Corrections Summary

### Critical Corrections Required (14 items)

1. **Fix accuracy values throughout** — M6 PTQ accuracy is 0.8702, not 0.7837. Update Abstract, Results Table 1, Analysis, Discussion, Conclusion.
2. **Fix Results Table 1** — Correct accuracy for M2 PTQ (0.8654), M2 QAT (0.8702), M5 PTQ (0.8125), M6 PTQ F1 weighted (0.8709).
3. **Fix Results Table 2** — SVM Tier 3 CV F1 macro: 0.7917 (not 0.7091). RF Tier 3 CV F1 macro: 0.7778 (not 0.6880).
4. **Fix Results Table 4** — All Float32 F1 baseline values: M6 = 0.7826, M2 = 0.7438, M5 = 0.5987. Fix M5 "degradation" to "improvement."
5. **Fix Methodology** — M2 best: LR 0.002, dropout 0.2, aug "none". M5 best: LR 0.001, dropout 0.3, aug "default". M6 best: LR 0.002, aug "none".
6. **Fix Discussion** — M6 accuracy is 87.0% not 78.4%. Tier 3 comparison should use Condition B (F1 0.715). Normal Start-Up playback F1 is 0.200.
7. **Fix Conclusion** — Max F1 delta is -0.042 (M6 QAT Tier 1), not -0.015.

### Moderate Corrections Required (10 items)

8. **Standardize SRAM** — Use 175 KB (68%) consistently across all sections.
9. **Fix QAT agreement range** — 92-100% (or 92-97% with M5 caveat) in Analysis, Discussion, Conclusion.
10. **Fix Results Table 4 Float32 baselines** — Use values from quantization_summary.csv.
11. **Fix Conclusion 2.7x** — Clarify this is accuracy improvement, not F1.
12. **Fix Results Table 5** — SRAM internal consistency (175 + 81 = 256 KB).
13. **Fix Discussion CMSIS-NN estimate** — Use 282 ms (from deployment_results.md) not 260 ms.
14. **Fix Results Table 4 M5 direction** — Int8 is better than float32, not worse.

### Minor Issues (12 items)

15-26. Inference rounding (783 vs 785 ms), flash % (32.5 vs 33%), latency stds, reference year, model count wording, feature ordering, M2 Block 3 description, TinyML latency phrasing, RQ4 origin, Tier 3 F1 rounding, Idle Fault percentage, Condition A unverifiable.

---

## Confirmed Correct Across All Sections

The following key claims were verified as correct by all relevant agents:

- Dataset: 1,386 files, 970/208/208 split, 13 conditions, 3 states
- Tier 2 class distribution and 12.8x imbalance ratio
- All augmentation condition sample counts (970, 3317, 3317, 5664)
- Noise bank: 360 samples, 23 categories, 6 groups, operational-state mapping
- Feature extraction params: n_fft=512, hop=256, n_mels=40
- Model parameter counts: M2=14,566, M5=6,246, M6=8,166
- M6 PTQ Tier 2 **F1 macro = 0.7835** (correct everywhere)
- All int8 F1 macro values in quantization tables
- All ablation F1 values (Conditions B, C, D)
- All McNemar p-values > 0.05
- Model sizes: M2 20.1 KB, M5 15.2 KB, M6 21.3 KB
- Flash usage: 319 KB (33%)
- Tensor arena: 63,556 bytes used of 65,536 allocated
- Playback test: Bluetooth 68.3% accuracy, F1 0.5465; Laptop 25.0%, F1 0.154
- All per-class Bluetooth recall values
- Confidence calibration: correct 0.808, incorrect 0.655
- 30.3% F1 degradation (Bluetooth), 80.3% (laptop)
- Augmentation improvement: M6 +0.206 F1 (Condition A to B)
- GCC 7.2.1 CMSIS-NN SIMD limitation
