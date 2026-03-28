#!/usr/bin/env python3
"""Verify that key numbers in the paper section files match source CSV/JSON files.

Checks all critical and moderate corrections from the validation report.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "report"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

errors = []
passes = []


def check(condition, description):
    if condition:
        passes.append(f"  PASS: {description}")
    else:
        errors.append(f"  FAIL: {description}")


def read_section(name):
    return (REPORT / name).read_text()


def load_quant_csv():
    rows = []
    with open(RESULTS / "quantization_summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_classical_csv():
    rows = []
    with open(RESULTS / "classical_ml_summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_hp_json():
    with open(MODELS / "hyperparameter_search.json") as f:
        return json.load(f)


def load_playback_json():
    with open(RESULTS / "playback_test_results_bluetooth_speaker.json") as f:
        return json.load(f)


def find_quant_row(rows, model, tier, method):
    for r in rows:
        if r["model"] == model and r["tier"] == str(tier) and r["method"] == method:
            return r
    return None


def find_classical_row(rows, key):
    for r in rows:
        if r["model"] == key:
            return r
    return None


# ---- Load all sources ----
quant = load_quant_csv()
classical = load_classical_csv()
hp = load_hp_json()
playback = load_playback_json()

# ---- Load all sections ----
abstract = read_section("abstract_section.md")
intro = read_section("introduction_section.md")
methodology = read_section("methodology_section.md")
analysis = read_section("analysis_section.md")
results = read_section("results_section.md")
discussion = read_section("discussion_section.md")
conclusion = read_section("conclusion_and_future_work_section.md")

print("=" * 70)
print("PAPER NUMBER VERIFICATION")
print("=" * 70)

# ======================================================================
# 1. ABSTRACT
# ======================================================================
print("\n--- Abstract ---")

check("175 KB of SRAM (68%)" in abstract,
      "SRAM: 175 KB (68%)")
check("five model architectures" in abstract,
      "Model count: five")
check("785 ms inference" in abstract,
      "Inference time: 785 ms")
check("179 KB" not in abstract,
      "No stale 179 KB")

# ======================================================================
# 2. INTRODUCTION
# ======================================================================
print("\n--- Introduction ---")

check("Tsoukas et al., 2025" in intro,
      "Tsoukas citation year: 2025")
check("Tsoukas et al., 2024" not in intro,
      "No stale Tsoukas 2024")

# ======================================================================
# 3. METHODOLOGY — Hyperparameter search
# ======================================================================
print("\n--- Methodology ---")

# M2 best HP
check(hp["m2"]["best_lr"] == 0.002, "HP JSON: M2 best_lr = 0.002")
check(hp["m2"]["best_dropout"] == 0.2, "HP JSON: M2 best_dropout = 0.2")
check(hp["m2"]["best_augmentation"] == "none", "HP JSON: M2 best_augmentation = none")

check("M2 at learning rate 0.002 with dropout 0.2 and no augmentation" in methodology,
      "Methodology M2 HP matches JSON")

# M5 best HP
check(hp["m5"]["best_lr"] == 0.001, "HP JSON: M5 best_lr = 0.001")
check(hp["m5"]["best_dropout"] == 0.3, "HP JSON: M5 best_dropout = 0.3")
check(hp["m5"]["best_augmentation"] == "default", "HP JSON: M5 best_augmentation = default")

check("M5 at learning rate 0.001 with dropout 0.3 and default augmentation" in methodology,
      "Methodology M5 HP matches JSON")

# M6 best HP
check(hp["m6"]["best_lr"] == 0.002, "HP JSON: M6 best_lr = 0.002")
check(hp["m6"]["best_augmentation"] == "none", "HP JSON: M6 best_augmentation = none")

check("M6 at learning rate 0.002 with no dropout sweep" in methodology,
      "Methodology M6 HP matches JSON")
check("no augmentation" in methodology.split("M6 at learning rate 0.002")[1][:100],
      "Methodology M6 augmentation = none")

# Block 3 MaxPool
check("third block omits max pooling" in methodology,
      "M2 Block 3 no MaxPool noted")

# ======================================================================
# 4. RESULTS — Table 1 accuracy values
# ======================================================================
print("\n--- Results Table 1 ---")

m6_ptq_t2 = find_quant_row(quant, "m6", 2, "ptq")
m2_ptq_t2 = find_quant_row(quant, "m2", 2, "ptq")
m2_qat_t2 = find_quant_row(quant, "m2", 2, "qat")
m5_ptq_t2 = find_quant_row(quant, "m5", 2, "ptq")

check(f"0.8702" in m6_ptq_t2["int8_accuracy"],
      f"CSV: M6 PTQ Tier 2 int8_accuracy = {m6_ptq_t2['int8_accuracy']}")
check("0.8702" in results.split("M6 DS-CNN PTQ")[1][:50],
      "Table 1 M6 PTQ accuracy = 0.8702")

check(f"0.8654" in m2_ptq_t2["int8_accuracy"],
      f"CSV: M2 PTQ Tier 2 int8_accuracy = {m2_ptq_t2['int8_accuracy']}")
check("0.8654" in results.split("M2 2D-CNN PTQ")[1][:50],
      "Table 1 M2 PTQ accuracy = 0.8654")

check(f"0.8702" in m2_qat_t2["int8_accuracy"],
      f"CSV: M2 QAT Tier 2 int8_accuracy = {m2_qat_t2['int8_accuracy']}")
check("0.8702" in results.split("M2 2D-CNN QAT")[1][:50],
      "Table 1 M2 QAT accuracy = 0.8702")

check(f"0.8125" in m5_ptq_t2["int8_accuracy"],
      f"CSV: M5 PTQ Tier 2 int8_accuracy = {m5_ptq_t2['int8_accuracy']}")
check("0.8125" in results.split("M5 1D-CNN PTQ")[1][:50],
      "Table 1 M5 PTQ accuracy = 0.8125")

# ======================================================================
# 5. RESULTS — Table 2 Classical ML
# ======================================================================
print("\n--- Results Table 2 ---")

svm_t3 = find_classical_row(classical, "svm_tier3")
rf_t3 = find_classical_row(classical, "rf_tier3")

check(svm_t3["cv_f1_macro"] == "0.7917",
      f"CSV: SVM Tier 3 CV F1 = {svm_t3['cv_f1_macro']}")
check("0.7917" in results,
      "Table 2 SVM Tier 3 CV F1 = 0.7917")

check(rf_t3["cv_f1_macro"] == "0.7778",
      f"CSV: RF Tier 3 CV F1 = {rf_t3['cv_f1_macro']}")
check("0.7778" in results,
      "Table 2 RF Tier 3 CV F1 = 0.7778")

check(svm_t3["f1_weighted"] == "0.7494",
      f"CSV: SVM Tier 3 F1 weighted = {svm_t3['f1_weighted']}")
check("0.7494" in results,
      "Table 2 SVM Tier 3 F1 weighted = 0.7494")

# ======================================================================
# 6. RESULTS — Table 4 Float32 baselines
# ======================================================================
print("\n--- Results Table 4 ---")

check(m6_ptq_t2["f32_f1_macro"] == "0.7826",
      f"CSV: M6 Tier 2 f32_f1 = {m6_ptq_t2['f32_f1_macro']}")

# Check Table 4 rows directly using pipe-delimited patterns
check("| M6 DS-CNN | PTQ | 0.7826 |" in results,
      "Table 4 M6 PTQ Float32 F1 = 0.7826")
check("| M6 DS-CNN | QAT | 0.7826 |" in results,
      "Table 4 M6 QAT Float32 F1 = 0.7826")
check("| M2 2D-CNN | PTQ | 0.7438 |" in results,
      "Table 4 M2 PTQ Float32 F1 = 0.7438")
check("| M2 2D-CNN | QAT | 0.7438 |" in results,
      "Table 4 M2 QAT Float32 F1 = 0.7438")
check("| M5 1D-CNN | PTQ | 0.5987 |" in results,
      "Table 4 M5 PTQ Float32 F1 = 0.5987")

# Check stale values removed
check("0.7844" not in results,
      "No stale M6 PTQ Float32 F1 (0.7844)")
check("0.7723" not in results,
      "No stale M6 QAT Float32 F1 (0.7723)")
check("0.6288" not in results,
      "No stale M5 Float32 F1 (0.6288)")

# Check prose about degradation
check("largest observed decrease is 0.0051" in results,
      "Table 4 prose: largest decrease is 0.0051 (not 0.015 degradation)")

# ======================================================================
# 7. RESULTS — Table 5 Memory
# ======================================================================
print("\n--- Results Table 5 ---")

check("| SRAM (static) | 175 KB | 256 KB | 68% |" in results,
      "Table 5 SRAM = 175 KB, 68%")
check("| SRAM remaining | 81 KB |" in results,
      "Table 5 SRAM remaining = 81 KB")
check("| Flash | 319 KB | 983 KB | 33% |" in results,
      "Table 5 Flash = 33%")
check("179 KB" not in results,
      "No stale 179 KB in results")

# ======================================================================
# 8. RESULTS — Table 6 Latency
# ======================================================================
print("\n--- Results Table 6 ---")

infer_std = playback["latency"]["infer_ms_std"]
total_std = playback["latency"]["total_ms_std"]
check(round(infer_std, 1) == 0.1,
      f"Playback JSON: infer_ms_std = {infer_std:.3f} rounds to 0.1")
check(round(total_std, 1) == 0.7,
      f"Playback JSON: total_ms_std = {total_std:.3f} rounds to 0.7")

check("| Model inference | 784.9 | 0.1 |" in results,
      "Table 6 inference std = 0.1")
check("| **Total cycle** | **2,472.5** | **0.7** |" in results,
      "Table 6 total std = 0.7")

# ======================================================================
# 9. ANALYSIS
# ======================================================================
print("\n--- Analysis ---")

check("175 KB (68% of 256 KB)" in analysis,
      "Analysis SRAM = 175 KB (68%)")
check("179 KB" not in analysis,
      "No stale 179 KB in analysis")
check("87.0%" in analysis,
      "Analysis: int8 accuracy = 87.0% (not 78.4%)")
check("785 ms" in analysis,
      "Analysis: inference = 785 ms")
check("92–97% for M2 and M6" in analysis,
      "Analysis: QAT agreement range clarified")

# ======================================================================
# 10. DISCUSSION
# ======================================================================
print("\n--- Discussion ---")

check("87.0%" in discussion,
      "Discussion: M6 accuracy = 87.0%")
check("78.4%" not in discussion,
      "No stale 78.4% accuracy in discussion")
check("F1 0.713" in discussion,
      "Discussion: M6 Tier 3 F1 = 0.713 (not 0.762)")
check("F1 0.762" not in discussion,
      "No stale F1 0.762 in discussion")
check("F1 = 0.200" in discussion,
      "Discussion: Normal Start-Up F1 = 0.200 (not 0.375)")
check("F1 = 0.375" not in discussion,
      "No stale Normal Start-Up F1 0.375")
check("282 ms" in discussion,
      "Discussion: CMSIS-NN estimate = 282 ms")
check("260 ms" not in discussion,
      "No stale 260 ms in discussion")
check("56.9%" in discussion,
      "Discussion: Idle Fault = 56.9%")

# ======================================================================
# 11. CONCLUSION
# ======================================================================
print("\n--- Conclusion ---")

check("175 KB of SRAM (68%)" in conclusion,
      "Conclusion SRAM = 175 KB (68%)")
check("179 KB" not in conclusion,
      "No stale 179 KB in conclusion")
check("0.042" in conclusion,
      "Conclusion: max |ΔF1| = 0.042")
check("−0.015" not in conclusion,
      "No stale -0.015 max delta")
check("92% to 100%" in conclusion,
      "Conclusion: agreement range = 92-100%")
check("improvement in accuracy" in conclusion,
      "Conclusion: 2.7x is accuracy (not F1)")
check("282 ms" in conclusion,
      "Conclusion: CMSIS-NN estimate = 282 ms")
check("260 ms" not in conclusion,
      "No stale 260 ms in conclusion")

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n  Passed: {len(passes)}")
print(f"  Failed: {len(errors)}")

if errors:
    print(f"\n{'!' * 70}")
    print("FAILURES:")
    for e in errors:
        print(e)
    print(f"{'!' * 70}")
    sys.exit(1)
else:
    print("\n  All checks passed! Paper numbers match source files.")
    sys.exit(0)
