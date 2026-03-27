# 7. Discussion

This section interprets the experimental results presented in Section 6, situates them within the broader tinyML and automotive diagnostics literature, and discusses their practical implications for edge deployment.

## 7.1 Key Findings

### 7.1.1 Quantization is Effectively Free for Small Audio Models

Perhaps the most practically significant finding of this study is that int8 post-training quantization (PTQ) introduces no statistically significant accuracy loss for any of the fifteen model–tier–method combinations evaluated. McNemar's test returned p > 0.05 in every case, and prediction agreement between float32 and int8 models ranged from 95% to 99%, indicating that the quantized models reproduce nearly identical decision boundaries. This result holds across all three architectures (M2 2D-CNN, M5 1D-CNN, and M6 DS-CNN) and all three classification tiers.

The robustness to quantization is attributable to several structural properties of the models in this study. First, the parameter counts are small—ranging from 6,246 (M5) to 14,566 (M2)—which means fewer weights are subject to rounding error. Second, BatchNorm folding absorbs normalization parameters into the preceding convolutional weights prior to quantization, reducing the number of independently quantized tensors. Third, GlobalAveragePooling aggregates spatial information before the final dense layer, smoothing out localized quantization artifacts.

This finding extends prior quantization research, which has focused predominantly on large-scale models. Jacob et al. (2018) demonstrated that int8 quantization preserves accuracy for models such as MobileNet and Inception on ImageNet-scale tasks, but those architectures contain millions of parameters. The present study confirms that the same conclusion holds at a much smaller scale—models with fewer than 15,000 parameters—where one might expect quantization noise to be proportionally more disruptive. For tinyML practitioners working with audio classification models of similar complexity, PTQ is sufficient and QAT adds training complexity without measurable benefit.

### 7.1.2 Data Augmentation Matters More Than Architecture

The augmentation ablation study revealed that data engineering produced a larger performance gain than architecture selection. The single most impactful intervention across the entire project was the transition from Condition A (original dataset, 970 samples, severe class imbalance at 12.8:1) to Condition B (synthetic waveform augmentation with class balancing, 3,317 samples). For M6 DS-CNN on Tier 2, this change improved F1 macro from 0.571 to 0.777—a gain of 0.206 points. By comparison, the difference between the best and worst neural network architectures on the same tier under matched augmentation conditions was substantially smaller, typically in the range of 0.05 to 0.15 F1 points.

Real-world noise injection (Condition C), which mixed training samples with ambient noise recorded from a 2008 Volkswagen Jetta at signal-to-noise ratios of 5–20 dB, showed marginal effects on the clean test set. Deltas between Conditions B and C were small (±0.01–0.05 F1), consistent with the expectation that noise injection primarily improves robustness to deployment-time acoustic interference rather than performance on clean evaluation data. The combined condition (Condition D, 5,664 samples) was only beneficial for M2 2D-CNN, which achieved its best Tier 2 F1 of 0.751 under this condition. M5 and M6 did not benefit from the additional data volume beyond Condition B.

These results carry a practical implication: for practitioners working with small, imbalanced audio datasets, the highest-return investment is in data augmentation and class balancing rather than in architecture exploration. Class balancing alone resolves gradient instability (discussed in Section 7.1.6) and enables stable training of architectures that would otherwise fail to converge.

### 7.1.3 Speaker Frequency Response Dominates the Domain Gap

The speaker hardware ablation study revealed that the acoustic fidelity of the playback apparatus—not the classifier or the on-device feature extraction pipeline—is the primary source of accuracy degradation in the end-to-end evaluation. Laptop speakers, which roll off sharply below approximately 300 Hz, caused an 80.3% drop in F1 macro (from 0.7835 on PC to 0.154). This occurred because car engine idle and braking sounds contain significant spectral energy in the 50–500 Hz range, and when this low-frequency content is removed by the speaker's transfer function, the resulting mel-spectrogram resembles that of startup sounds—which are dominated by transient mid- and high-frequency content. Indeed, 72.1% of all predictions under laptop playback fell into startup classes regardless of the true label.

A Bluetooth speaker with better bass response partially restored the low-frequency content, improving accuracy from 25.0% to 68.3%—a 2.7× gain—and reducing the F1 degradation to 30.3%. Notably, two fault classes (Braking Fault and Start-Up Fault) achieved PC-equivalent recall through the Bluetooth speaker, and Idle Fault came within 8.4 percentage points of the PC reference.

This finding has an important implication for real-world deployment. In an actual use scenario—where the Arduino's PDM microphone is positioned near an engine with no speaker in the signal path—the full frequency range of the engine's acoustic output would be captured directly. The Bluetooth playback test therefore represents a conservative lower bound on expected real-world accuracy. Direct microphone-to-engine deployment should approach PC-level classification performance.

*[See Figure [X]: Per-class recall comparison across PC evaluation, Bluetooth speaker playback, and laptop speaker playback — `results/paper_pc_vs_playback_recall.png`]*

### 7.1.4 Classical ML Remains Competitive at Binary Classification

An unexpected result was the strong performance of classical machine learning baselines on the binary classification task. On Tier 1, SVM achieved an F1 macro of 0.916, outperforming all neural network architectures. This advantage reversed on the more complex multi-class tasks: on Tier 2, M6 DS-CNN (F1 0.7835) outperformed SVM (F1 0.699) by 8.4 percentage points, and on Tier 3, M6 (F1 0.762) exceeded SVM (F1 0.711) by 5.1 points.

This pattern reflects the well-understood principle that hand-crafted statistical features (in this case, aggregated MFCC statistics) can be highly effective when the classification boundary is simple and the feature space is well-matched to the task. For binary normal-versus-fault screening, the spectral differences are sufficiently large that a linear or RBF kernel can separate the classes effectively. As the number of classes increases and the decision boundaries become more complex, learned hierarchical representations in convolutional architectures provide an advantage. For deployment contexts where only binary screening is needed, classical ML may be preferable because it avoids the complexity of on-device neural inference entirely.

### 7.1.5 The Accuracy–F1 Paradox and Metric Selection

A particularly instructive finding involved the divergence between accuracy and F1 macro for the M2 2D-CNN architecture on Tier 2. M2 achieved the highest accuracy among all models (87.5%) but only an F1 macro of 0.751—a 12.4 percentage-point gap. In contrast, M6 DS-CNN achieved a lower accuracy (78.4%) but a higher F1 macro (0.784), and was selected as the deployment model on this basis.

The explanation lies in the class distribution of the training and test sets. The Idle Fault class accounts for 56.7% of the Tier 2 training set. M2 learned to over-predict this majority class, which inflated its accuracy—since guessing the most common class correctly is rewarded—while performing poorly on minority classes such as Normal Start-Up (43 original samples) and Braking Fault. F1 macro, which assigns equal weight to every class regardless of prevalence, exposed this bias.

This finding underscores a broader methodological point that is especially relevant for fault detection systems: accuracy is a misleading metric when class distributions are imbalanced. Models intended to detect minority fault conditions should be evaluated and selected using F1 macro or a similarly class-balanced metric. Had accuracy been the sole selection criterion, the deployed model would have been one that systematically missed rare but important fault classes.

### 7.1.6 Training Instability on Small Imbalanced Datasets

M6 DS-CNN exhibited severe training instability on Condition A (the original 970-sample dataset with a 12.8:1 class imbalance ratio). At learning rates of 0.001 or higher, validation accuracy collapsed to approximately 23%, indicating that the model had degenerated to predicting only the majority class. After augmentation and class balancing (Condition B), M6 trained stably at a learning rate of 0.002, reaching 91.4% validation accuracy.

This instability arises from the interaction of two factors. Severe class imbalance causes the loss gradient to be dominated by the majority class, which biases parameter updates toward a trivial solution. A small dataset exacerbates this because the model encounters insufficient minority-class diversity to learn meaningful representations, even when class weighting is applied. Augmentation addresses both factors simultaneously: it increases dataset size (providing more gradient diversity) and equalizes class frequencies (removing the gradient bias).

For practitioners encountering training instability with small, imbalanced audio datasets, this result suggests that data augmentation with class balancing should be the first remediation step—before exploring alternative architectures, learning rate schedules, or other hyperparameter adjustments.

### 7.1.7 The QAT Prediction Agreement Paradox

A counterintuitive finding emerged from the quantization-aware training (QAT) results: QAT models exhibited lower prediction agreement with their float32 parents (92–97%) compared to PTQ models (97–100%), despite achieving equal or worse F1 scores. In other words, QAT changed more individual predictions without improving overall classification performance.

This paradox is explained by the mechanism of QAT itself. During QAT fine-tuning, fake quantization nodes inject simulated rounding noise into the forward pass, and the model's weights are updated via backpropagation to compensate. For large models with millions of parameters, this compensation can genuinely recover accuracy lost to quantization. However, for small models where PTQ already produces negligible quantization error, the additional gradient noise from fake quantization is counterproductive—it perturbs decision boundaries that were already well-calibrated, changing predictions without improving them. This finding reinforces the practical recommendation that PTQ is sufficient for audio models in the sub-15K parameter range, and that QAT should not be assumed to be universally beneficial.

### 7.1.8 DS-CNN as the Optimal MCU Architecture

The M6 DS-CNN architecture emerged as the optimal choice for microcontroller deployment across multiple evaluation criteria. It achieved the highest F1 macro on both Tier 2 (0.7835) and Tier 3 (0.762) while using fewer parameters (8,166) than M2 2D-CNN (14,566). As a quantized TFLite model, M6 occupies 21.3 KB of flash and requires a 63.6 KB tensor arena, both well within the hardware constraints of the Arduino Nano 33 BLE Sense Rev2 (983 KB flash, 256 KB SRAM).

The choice of depthwise separable convolutions is also well-aligned with the hardware acceleration ecosystem. The CMSIS-NN library (Lai et al., 2018) provides optimized kernels specifically targeting depthwise separable operations on Cortex-M processors, and these operations benefit disproportionately from SIMD acceleration when the toolchain supports it. The four-block DS-CNN architecture used in this study mirrors the design proposed by Zhang et al. (2017) for keyword spotting on Cortex-M7, validating the architectural pattern for audio classification tasks on Cortex-M4.

*[See Figure [X]: F1 macro versus int8 model size for all quantized models — `results/paper_accuracy_vs_size.png`]*

## 7.2 Comparison with Related Work

The M6 DS-CNN model in this study (8,166 parameters, 21.3 KB int8) is substantially smaller than most published keyword spotting models, which typically range from 30,000 to 100,000 parameters. Zhang et al. (2017) demonstrated DS-CNN architectures achieving 95.4% accuracy on the Google Speech Commands dataset, deployed on a Cortex-M7 microcontroller within approximately 70 KB of memory. Our work applies the same architectural family to a more complex task—six-class automotive sound diagnostics rather than keyword detection—on a lower-tier processor (Cortex-M4 at 64 MHz versus Cortex-M7 at 216 MHz). The resulting F1 macro of 0.7835 on a six-class task trained from only 1,386 total samples is competitive given the dataset constraints.

The inference latency of 783 ms observed in this study is notably slower than the approximately 50–100 ms reported for keyword spotting systems on comparable hardware. This gap is attributable to two factors. First, the mel-spectrogram feature extraction pipeline processes a 1.5-second audio window, which is longer than the typical 1-second window used in keyword spotting. Second, and more significantly, the Arduino Mbed OS board package bundles GCC 7.2.1, a compiler from 2017 that lacks support for ARMv7E-M ACLE intrinsics (specifically `__sxtb16`, `__smlabb`, and `__smlatt`). These intrinsics are required for CMSIS-NN to emit SIMD instructions that would reduce inference from approximately 8.4 cycles per MAC operation to the theoretical 3 cycles per MAC. Lai et al. (2018) reported a 4.6× speedup from CMSIS-NN SIMD optimization on Cortex-M processors; our investigation confirmed the theoretical capability but identified this toolchain limitation—a practical gap not documented in the existing literature. Upgrading to GCC 10 or later would be expected to reduce inference time to approximately 260 ms, bringing the total classification cycle from 2.5 seconds to approximately 1.9 seconds. For the non-real-time diagnostic use case addressed here, the current latency is acceptable.

Existing work on car sound classification has focused primarily on server-side inference using deep learning models running on GPUs. MCU-based deployment of automotive sound classifiers is underexplored in the literature. Similarly, quantization studies (Jacob et al., 2018; Krishnamoorthi, 2018) have concentrated on large-scale models such as MobileNet and ResNet. Our results extend the finding that PTQ is sufficient to models an order of magnitude smaller, addressing a gap in the quantization literature relevant to the tinyML community.

## 7.3 Industry Applications

The system demonstrated in this study suggests several practical deployment pathways.

**Predictive maintenance for fleet management.** Fleet operators could equip vehicles with Arduino-based diagnostic sensors for continuous monitoring during operation, alerting drivers or fleet managers when fault conditions are detected. At a hardware cost of approximately $30 per unit (the retail price of an Arduino Nano 33 BLE Sense Rev2), the system is economically viable for large-scale deployment. Because all audio processing occurs on-device, no audio data is transmitted or stored, making the system inherently privacy-preserving.

**Consumer vehicle health monitoring.** Integration with an OBD-II port for combined sound and engine telemetry data, paired with a smartphone companion application receiving classification results via Bluetooth Low Energy, could supplement or partially replace expensive dealership diagnostics for common faults. A confidence threshold of 0.70 enables differentiated messaging: predictions above the threshold would produce a specific fault indication, while those below it would generate an "uncertain—please retry" message.

**Mechanic diagnostic tool.** A handheld device that a mechanic points at an engine compartment could provide immediate classification with a 2.5-second cycle time—fast enough for interactive use. Majority voting over three to four consecutive classifications would reduce the single-inference error rate, and the class probability vector could serve as a differential diagnosis, ranking fault likelihoods.

**Edge AI for automotive IoT.** More broadly, this work demonstrates the feasibility of edge audio intelligence in automotive contexts: low cost, no cloud dependency, low power consumption, and BLE connectivity that could enable mesh sensor networks in garages and workshops.

## 7.4 Deployment Recommendations

Based on the findings of this study, four strategies are recommended for production deployment.

First, **confidence-based filtering** should be applied with a threshold of approximately 0.70. The well-separated confidence distributions observed in the Bluetooth playback test—mean confidence of 0.808 for correct predictions versus 0.655 for incorrect predictions—indicate that thresholding can reject an estimated 50% of incorrect predictions while retaining approximately 85% of correct ones. Predictions below the threshold should be flagged as uncertain and prompt a retry.

Second, **majority voting** over three to four consecutive classifications (7.5–10 seconds total) would significantly reduce the single-inference error rate. Reporting the majority class across multiple inference cycles is a standard technique for edge classification systems and is straightforward to implement on the device.

Third, a **toolchain upgrade** would unlock a substantial inference speedup. Migrating from the Arduino Mbed OS build system (GCC 7.2.1) to PlatformIO or the nRF Connect SDK with ARM GCC 12 or later would enable CMSIS-NN SIMD acceleration, reducing inference from 783 ms to an estimated 260 ms and total cycle time from 2.5 seconds to approximately 1.9 seconds.

Fourth, a **multi-model deployment strategy** could accommodate devices with varying memory constraints. The M6 DS-CNN PTQ model (21.3 KB, 64 KB tensor arena) is recommended as the primary model for devices with sufficient SRAM. For devices with severely constrained SRAM, M5 1D-CNN PTQ (15.2 KB, approximately 25 KB tensor arena) provides a fallback with reduced accuracy. Model selection can be implemented as a compile-time configuration directive.

*[See Figure [X]: Latency breakdown showing audio capture, feature extraction, and inference phases — `results/paper_latency_breakdown.png`]*

*[See Figure [X]: Flash and SRAM utilization relative to hardware budget — `results/paper_memory_utilization.png`]*

## 7.5 Ethical Considerations

### 7.5.1 Safety Implications

As a screening tool rather than a definitive diagnostic instrument, the classifier must be evaluated in terms of both false negatives and false positives. A false negative—failing to detect a genuine fault—could lead to delayed maintenance and potential safety consequences if the fault is progressive. A false positive—misclassifying normal operation as a fault—could impose unnecessary repair costs and erode user trust. The model should therefore be deployed with clear messaging that it provides preliminary screening, not authoritative diagnosis, and all predictions should be accompanied by confidence scores to support informed decision-making.

### 7.5.2 Bias Considerations

Three forms of bias are present in this study. First, **vehicle bias**: the training data originates from the Car Diagnostics Dataset on Kaggle, for which the vehicle models and recording conditions are not fully documented, and the noise bank was collected from a single 2008 Volkswagen Jetta. Performance on vehicles with substantially different acoustic profiles—such as electric vehicles, diesel trucks, or newer models with advanced sound insulation—is unknown. Second, **environmental bias**: noise data was collected in a single geographic location on a single day under moderate weather conditions. Rain, snow, extreme temperatures, and high-traffic environments were not represented. Third, **class bias**: even after augmentation, minority classes remain fundamentally limited by source diversity. The Normal Start-Up class, which began with only 43 original samples, had the weakest classification performance (F1 = 0.375 on the Bluetooth playback test), reflecting the difficulty of learning robust representations from a narrow sample base.

### 7.5.3 Privacy

The system implements a privacy-by-design architecture. All audio processing—capture, feature extraction, and classification—occurs entirely on the Cortex-M4 microcontroller. No audio data is transmitted, stored, or uploaded to cloud services. Bluetooth Low Energy communication carries only the classification result (a class label and a confidence score). This architecture avoids the privacy concerns associated with cloud-based audio analysis, where raw speech or environmental audio could be intercepted, retained, or repurposed.

## References

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2704–2713.

Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for efficient inference: A whitepaper. *arXiv preprint arXiv:1806.08342*.

Lai, L., Suda, N., & Chandra, V. (2018). CMSIS-NN: Efficient neural network kernels for Arm Cortex-M CPUs. *arXiv preprint arXiv:1801.06601*.

Zhang, Y., Suda, N., Lai, L., & Chandra, V. (2017). Hello Edge: Keyword spotting on microcontrollers. *arXiv preprint arXiv:1711.07128*.
