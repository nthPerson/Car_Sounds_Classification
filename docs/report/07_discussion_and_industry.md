# Discussion & Industry Applications — Report Assembly

## 7.1 Key Findings Discussion

### Finding 1: Quantization is Free for Small Audio Models
- All 15 int8 models (3 architectures x 3 tiers, PTQ and QAT) show no statistically significant accuracy loss
- McNemar's test p > 0.05 for every combination
- Prediction agreement 95-99%: int8 models make the same decisions as float32
- **Why:** Small parameter count (6K-15K), BatchNorm folding, GlobalAveragePooling all reduce quantization sensitivity
- **Implication for tinyML community:** For models with < 15K parameters on audio classification, PTQ is sufficient — QAT adds complexity without benefit

### Finding 2: Data Augmentation > Architecture for Small Datasets
- Class balancing + synthetic augmentation produced the largest accuracy improvement (+0.206 F1 for M6)
- Architecture differences (M2 vs M5 vs M6) produced smaller improvements (~0.05-0.15 F1 range)
- Real-world noise injection showed marginal clean-test improvement
- **Implication:** For practitioners with small audio datasets, investing in data augmentation and class balancing will yield better returns than trying more complex architectures

### Finding 3: Speaker Frequency Response Dominates Domain Gap
- Laptop speakers (no bass) caused 80% accuracy drop
- Bluetooth speaker (with bass) reduced drop to 30%
- The model and feature extraction are NOT the weak link — the audio reproduction hardware is
- **Implication:** Real-world deployment (direct mic-to-engine) should approach PC-level accuracy since no speaker is in the path

### Finding 4: Classical ML Remains Competitive at Binary Classification
- SVM achieves F1 0.916 on Tier 1 (binary), outperforming all neural networks
- Neural networks win on Tier 2 (M6 0.7835 vs SVM 0.699, +8.4 pp) and Tier 3 (M6 0.762 vs SVM 0.711, +5.1 pp)
- **Implication:** For simple binary screening tasks, classical ML may be sufficient and avoids the complexity of on-device neural inference

### Finding 6: M2 Accuracy-F1 Paradox
- M2 achieves 87.5% accuracy on Tier 2 but only 0.751 F1 macro — a 12.4 pp gap
- M6 has lower accuracy (78.4%) but higher F1 macro (0.784) — chosen as deployment model
- **Explanation:** M2 over-predicts the majority class (Idle Fault, 56.7% of training set), inflating accuracy while performing poorly on minority classes. F1 macro equally weights all classes, revealing this bias.
- **Implication for deployment:** Accuracy alone is misleading for imbalanced datasets. F1 macro is the correct metric for selecting models intended to detect minority fault conditions.

### Finding 7: Training Instability on Small Imbalanced Datasets
- M6 was unstable on Condition A (original 970 samples): val_acc ~23% at LR >= 0.001
- After augmentation + class balancing (Condition B), M6 trained stably at LR=0.002 reaching 91.4% val_acc
- **Explanation:** Severe class imbalance (12.8x) combined with small dataset size causes gradient instability — the model collapses to predicting the majority class. Augmentation provides both diversity and balance.
- **Practical insight:** For practitioners with small, imbalanced audio datasets, data augmentation with class balancing should be the first step before architecture exploration.

### Finding 8: QAT Prediction Agreement Paradox
- QAT changes more predictions than PTQ (92-97% agreement vs 97-100%) despite matching or worse F1
- **Explanation:** QAT fine-tuning with fake quantization introduces gradient noise that shifts decision boundaries. For small models where PTQ already produces negligible error, QAT's perturbation is counterproductive — it changes predictions without improving them.

### Finding 5: DS-CNN is the Optimal Architecture for MCU Deployment
- M6 DS-CNN achieves highest F1 on Tier 2 and 3 despite fewer parameters (8,166) than M2 (14,566)
- Depthwise separable convolutions are explicitly supported by CMSIS-NN optimized kernels
- 21.3 KB model size and 64 KB tensor arena fit comfortably within hardware budgets
- **Note:** The 4x DS-block architecture mirrors ARM's keyword spotting DS-CNN (Zhang et al., 2017), validating the architecture choice for audio tasks on Cortex-M4

## 7.2 Comparison with Related Work

### Contextual References (to be expanded during final paper writing)
- **DS-CNN for keyword spotting** (Zhang et al., 2017): Used DS-CNN on mel-spectrograms for keyword detection on Cortex-M7. Our work applies the same architecture family to car sound diagnostics on Cortex-M4.
- **TinyML audio classification:** Edge Impulse, TensorFlow Lite Micro projects demonstrate audio classification on MCUs, but typically focus on simpler tasks (keyword spotting, anomaly detection) rather than multi-class automotive diagnostics.
- **CMSIS-NN benchmarks** (Lai et al., 2018; ARM, 2021): Report 4.6x speedup with CMSIS-NN SIMD on Cortex-M4. Our investigation confirms the theoretical capability but reveals a practical toolchain limitation (GCC 7.2.1) not documented in the literature.
- **Car sound classification:** Prior work typically uses server-side inference (deep learning on GPU). MCU deployment of car sound classifiers is underexplored.
- **Quantization for small models:** Literature (Jacob et al., 2018; Krishnamoorthi, 2018) focuses on large models (MobileNet, ResNet). Our results on models with < 15K parameters extend the finding that PTQ is sufficient to a much smaller scale.

### How Our Results Compare
- Our M6 DS-CNN (8,166 params, 21.3 KB) is smaller than most published keyword spotting models (~30K-100K params)
- 0.7835 F1 on a 6-class task from 1,386 samples is competitive given the dataset size
- The 783 ms inference (without SIMD) is slower than the ~50-100 ms reported for keyword spotting, but acceptable for our non-real-time use case

## 7.3 Industry Applications

### Predictive Maintenance for Fleet Management
- Fleet operators could equip vehicles with Arduino-based diagnostic sensors
- Continuous monitoring during operation, alerting when fault conditions detected
- Cost: ~$30 per sensor (Arduino Nano 33 BLE Sense Rev2)
- Privacy-preserving: all processing on-device, no audio transmitted

### Consumer Vehicle Health Monitoring
- Integration with OBD-II port for combined sound + engine data
- Smartphone companion app receives classifications via BLE
- Could supplement or replace expensive dealership diagnostics for common faults
- Confidence threshold (0.70) enables "check engine" vs "uncertain" messaging

### Mechanic Diagnostic Tool
- Handheld device for mechanics — point at engine, get immediate classification
- 2.5-second cycle time is fast enough for interactive use
- Majority voting over 3-4 classifications reduces error rate
- Class probabilities provide differential diagnosis (ranked fault likelihood)

### Edge AI for Automotive IoT
- Demonstrates feasibility of edge audio AI in automotive context
- No cloud dependency, low power, low cost
- BLE connectivity enables mesh sensor networks in garages/workshops

## 7.4 Deployment Recommendations

### Confidence-Based Filtering
- Set prediction confidence threshold at 0.70
- Accept predictions above threshold (mean confidence 0.808 for correct predictions)
- Flag predictions below threshold as "uncertain — please retry"
- Estimated effect: reject ~50% of incorrect predictions, retain ~85% of correct ones

### Majority Voting
- Run 3-4 consecutive classifications (7.5-10 seconds total)
- Report the majority class
- Reduces single-inference error rate significantly

### Toolchain Upgrade for Inference Speedup
- Current: GCC 7.2.1 → 783 ms inference (8.4 cycles/MAC)
- Upgrade path 1: PlatformIO with ARM GCC 12+ → ~260 ms inference (3 cycles/MAC)
- Upgrade path 2: nRF Connect SDK with GCC 12+ → same speedup
- Upgrade path 3: Wait for Arduino Mbed OS board package update
- 3x speedup would reduce total cycle from 2.5s to ~1.9s

### Multi-Model Deployment
- M6 DS-CNN PTQ (21.3 KB, 64 KB arena): primary model for devices with sufficient SRAM
- M5 1D-CNN PTQ (15.2 KB, ~25 KB arena): fallback for devices with very limited SRAM
- Model selection can be compile-time via `#define` in `config.h`

## 7.5 Ethical Considerations

### Safety Implications
- **False negatives** (missing a real fault): Could lead to delayed maintenance, potential safety risk
- **False positives** (flagging normal as fault): Could lead to unnecessary repairs, cost burden
- The model should be positioned as a **screening tool**, not a definitive diagnostic
- All predictions should include confidence scores to inform decision-making

### Bias Considerations
- **Vehicle bias:** Training data from Car Diagnostics Dataset (unknown vehicle models); noise bank from a single 2008 VW Jetta
- **Environmental bias:** Noise collected in one geographic location, one day, moderate weather
- **Class bias:** Even after augmentation, minority classes (Normal Start-Up: 43 original samples) are fundamentally limited by source diversity

### Privacy
- All audio processing happens on-device (Cortex-M4 microcontroller)
- No audio data is transmitted, stored, or uploaded to cloud services
- BLE transmission contains only classification results (class label + confidence)
- This is a **privacy by design** architecture

## Source Files
- `docs/on_device_evaluation_report.md` — Section 10 (Conclusions)
- `docs/deployment_results.md` — Section 8.3 (CMSIS-NN investigation)
- `docs/neural_network_models.md` — Augmentation ablation analysis
- `docs/noise_data_collection_and_augmentation.md` — Section 10 (Limitations)
- `notebooks/07_evaluation.ipynb` — Section 7 (Key Findings Summary)
