# Conclusion & Future Work — Report Assembly

## 8.1 Conclusions (Answering Research Questions)

### RQ1: Can a quantized neural network classify car sounds on a Cortex-M4 MCU?

**Yes.** The M6 DS-CNN PTQ model achieves Tier 2 F1 macro 0.7835 while fitting within all Arduino Nano 33 BLE Sense Rev2 constraints:
- Model size: 21.3 KB (vs 150 KB flash budget)
- Tensor arena: 63.6 KB (vs 80 KB SRAM budget)
- Total flash: 319 KB (33% utilization)
- Total SRAM: 179 KB (68% utilization)
- Inference: 783 ms (2.5s total cycle including 1.5s audio capture)

The complete end-to-end pipeline — PDM audio capture, CMSIS-DSP mel-spectrogram extraction, int8 TFLite Micro inference — runs entirely on the 64 MHz Cortex-M4F processor with no external dependencies.

### RQ2: What is the accuracy cost of int8 quantization for small audio models?

**Negligible.** Across all 15 quantized models (3 architectures x 3 tiers, PTQ and QAT):
- No model-tier-method combination shows a statistically significant accuracy change (McNemar p > 0.05 for all)
- Maximum F1 delta: -0.015 (M5 PTQ Tier 2)
- Prediction agreement: 95-99% (int8 models make the same predictions as float32)
- **PTQ is sufficient** — QAT provides no additional benefit for models with 6K-15K parameters

### RQ3: How does the speaker-microphone domain gap affect real-world deployment?

**The gap is 30.3% (F1 macro) through a Bluetooth speaker, dominated by speaker quality, not the classifier.** Specific findings:
- PC int8 F1: 0.7835 → Bluetooth playback F1: 0.5465 (30.3% degradation)
- Laptop speakers: F1 drops to 0.154 (80.3% degradation) due to missing bass below 300 Hz
- Bluetooth speakers: 2.7x improvement over laptop, restoring low-frequency content
- Two classes (Braking Fault, Start-Up Fault) achieve PC-equivalent recall through Bluetooth
- **Key implication:** In real deployment (mic near engine, no speaker), accuracy should approach PC level

### RQ4: Does real-world noise augmentation improve robustness?

**Mixed results on clean test data, potential benefit for deployment.** Findings:
- Condition B (synthetic augmentation + class balancing) produced the largest improvement: M6 F1 from 0.571 to 0.777 (+0.206)
- Condition C (+ real-world noise injection) showed marginal clean-test effect: M6 0.777 → 0.739 (-0.038)
- Condition D (combined) was best only for M2 (0.751), not M5 or M6
- **The dominant factor is class balancing**, not noise injection. Real-world noise may help in noisy deployment environments, but this benefit was not directly measured.

## 8.2 Limitations

### Dataset Limitations
1. **Small dataset:** 1,386 files total, 970 training, 208 test. Minority classes have very few test samples (Normal Start-Up: 9, Braking Fault: 11), leading to high per-sample variance.
2. **Single source:** Car Diagnostics Dataset from Kaggle — recording conditions, vehicles, and microphones are not documented.
3. **Class imbalance:** 12.8x ratio (Idle Fault: 552 vs Normal Start-Up: 43). Even with augmentation, all augmented copies derive from the same 43 source recordings.

### Noise Collection Limitations
4. **Single vehicle:** All 360 noise samples from one 2008 Volkswagen Jetta. Different vehicles have different acoustic signatures.
5. **Single session:** Collected on one day (March 22, 2026). Weather, traffic, and conditions on that day may not be representative.
6. **No rain/extreme weather:** Missing conditions that could affect deployment.

### Deployment Limitations
7. **GCC 7.2.1 toolchain:** Arduino Mbed OS board package bundles a 2017 compiler that prevents CMSIS-NN SIMD acceleration. Inference is 3x slower than theoretically achievable.
8. **Playback test (not real-engine test):** The playback test introduces a speaker in the signal path that does not exist in real deployment. Results are a lower bound, not a representative measurement.
9. **Single deployed model:** Only M6 PTQ was deployed to Arduino. Multi-model on-device comparison was not performed.

### Methodological Limitations
10. **No Condition A ablation:** Original data (970 samples, no augmentation) was not run as a formal ablation condition with the same evaluation framework. The A→B comparison (+0.206 F1 for M6) cannot be decomposed into dataset expansion vs class balancing effects.
11. **Small test set:** 208 samples with 6 classes — confidence intervals on metrics are wide (especially for classes with < 15 test samples). Single sample misclassification swings recall by 8-11% for minority classes.
12. **No cross-validation on test set:** Test set used once for final evaluation per standard practice, but single-split evaluation introduces variance.
13. **Hyperparameter search scope:** Hyperparameters were tuned on Tier 2 only, then applied to Tiers 1 and 3. Tier-specific optimization was not performed.
14. **TensorFlow version constraint:** TF 2.18.0 required due to cuDNN autotuner issue in TF 2.21 and tfmot compatibility. This may affect reproducibility with newer TF versions.
15. **Training instability on Condition A:** M6 DS-CNN collapsed to majority-class prediction (val_acc ~23%) at learning rates >= 0.001 with original imbalanced data. This instability was resolved by augmentation but represents a methodological constraint — the model's performance on the original dataset cannot be reliably measured.

## 8.3 Future Work

### Short-Term (Direct Extensions)
1. **Toolchain upgrade:** Deploy with PlatformIO or nRF Connect SDK using GCC 12+ to enable CMSIS-NN SIMD acceleration (expected 3x inference speedup: 783 ms → ~260 ms)
2. **On-vehicle field testing:** Test with Arduino positioned near actual engines instead of speaker playback. Would eliminate the speaker domain gap and provide realistic accuracy measurements.
3. **Streaming inference:** Implement overlapping capture windows (e.g., 0.75s hop with 1.5s window) for faster update rate without architectural changes
4. **Confidence thresholding + majority voting:** Implement production filtering (reject low-confidence predictions, vote over 3-4 consecutive inferences) and measure precision-recall trade-off

### Medium-Term (Research Extensions)
5. **Larger dataset:** Collect recordings from multiple vehicles (different makes, models, years, engine types including EVs) to improve generalization
6. **Transfer learning:** Initialize from AudioSet or ESC-50 pretrained features to improve performance with limited task-specific data
7. **Multi-sensor fusion:** Combine audio (PDM mic) with accelerometer/gyroscope data (also on-board the Arduino Nano 33 BLE Sense Rev2) for multi-modal fault detection
8. **Noise-robust evaluation:** Test model on deliberately noisy audio (mix test clips with noise at various SNRs) to quantify the effect of noise augmentation on deployment robustness

### Long-Term (System Extensions)
9. **Federated learning:** Train across multiple vehicles without sharing raw audio, enabling continuous model improvement while preserving privacy
10. **Anomaly detection mode:** Use the model's confidence scores for open-set detection — low confidence on all classes could indicate an unknown fault type
11. **Fleet deployment platform:** Develop a BLE mesh network of Arduino sensors reporting to a central dashboard for fleet-wide vehicle health monitoring
12. **Real-time alerting:** Integrate with vehicle systems (OBD-II, CAN bus) for immediate driver notification when faults are detected

## Source Files
- `docs/on_device_evaluation_report.md` — Sections 9-10 (Constraints, Conclusions)
- `docs/deployment_results.md` — Section 8.3 (CMSIS-NN investigation)
- `docs/noise_data_collection_and_augmentation.md` — Section 10 (Limitations)
- `docs/project_spec/dev_log.md` — Full project history
- `notebooks/07_evaluation.ipynb` — Section 7 (Key Findings Summary)
