# Abstract — Report Assembly

## Key Numbers for Abstract (~250 words)

- **Problem:** Real-time car sound classification on a constrained microcontroller
- **Dataset:** Car Diagnostics Dataset (Kaggle) — 1,386 WAV files, 13 conditions across 3 operational states (braking, idle, startup)
- **Taxonomy:** 3 classification tiers: Tier 1 (2-class binary), Tier 2 (6-class state-aware, primary target), Tier 3 (9-class individual faults)
- **Models evaluated:** 6 variants — Random Forest, SVM, 2D-CNN (M2), 1D-CNN (M5), DS-CNN (M6), plus int8 quantized variants (PTQ and QAT)
- **Best model:** M6 DS-CNN PTQ — Tier 2 F1 macro 0.7835, 21.3 KB model size, 8,166 parameters
- **Quantization finding:** Nearly lossless — no model-tier-method combination shows statistically significant accuracy change (all McNemar p > 0.05). PTQ sufficient; QAT provides no benefit.
- **Deployment target:** Arduino Nano 33 BLE Sense Rev2 (nRF52840, Cortex-M4F @ 64 MHz, 1 MB flash, 256 KB SRAM)
- **On-device performance:** 319 KB flash (33%), 179 KB SRAM (68%), 783 ms inference, 197 ms feature extraction, 2,470 ms total cycle
- **Playback test (Bluetooth speaker):** 68.3% accuracy, F1 macro 0.5465, 30.3% degradation from PC evaluation
- **Speaker ablation:** Laptop speakers achieved only 25% accuracy (frequency response rolls off below 300 Hz); Bluetooth speaker: 68.3% (2.7x improvement)
- **Data augmentation:** Synthetic waveform augmentation with class balancing improved M6 F1 from 0.571 to 0.777 (+0.206). Real-world noise injection showed marginal clean-test improvement.
- **Unique contributions:** Real-world Arduino noise collection (360 samples from 2008 VW Jetta), speaker hardware ablation study, CMSIS-NN toolchain investigation, TFLite Micro build from source

## Suggested Abstract Structure
1. One sentence: problem and motivation
2. One sentence: dataset and experimental setup
3. Two sentences: model comparison results (classical ML vs NN, best model)
4. One sentence: quantization finding
5. One sentence: on-device deployment results
6. One sentence: playback test and domain gap finding
7. One sentence: unique contribution (noise collection or speaker ablation)

## Source Files
- `docs/on_device_evaluation_report.md` — Section 10 (Conclusions)
- `results/quantization_summary.csv` — All 18 int8 models
- `results/playback_test_results_bluetooth_speaker.json` — Playback metrics
- `docs/deployment_results.md` — Sections 7-9 (memory, latency, playback)
