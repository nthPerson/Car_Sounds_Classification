# 4. Analysis

This section interprets the experimental observations across the four evaluation dimensions defined in Section 3: classification performance (D1), quantization impact (D2), on-device performance (D3), and end-to-end system evaluation (D4). Where the Results section (Section 5) presents the quantitative tables and figures, this section focuses on explaining *why* the observed patterns emerge and what they imply for the viability of TinyML-based automotive sound classification.

## 4.1 Classical ML Baseline Analysis

The classical ML baselines reveal two important characteristics of the classification task. First, SVM consistently outperforms Random Forest across all three taxonomy tiers, achieving F1 macro scores of 0.916 versus 0.872 on Tier 1, 0.699 versus 0.679 on Tier 2, and 0.711 versus 0.702 on Tier 3. The performance gap narrows as the number of classes increases, suggesting that the SVM's superior margin-based decision boundaries offer diminishing returns when the feature space must accommodate more fine-grained distinctions.

Second, and more critically, the gap between accuracy and F1 macro on the multi-class tiers exposes the severity of class imbalance. On Tier 2, Random Forest achieves 84.1% accuracy but only 0.679 F1 macro — a 16 percentage point discrepancy. This gap arises because accuracy is dominated by performance on the majority class (Idle Fault, which accounts for 56.9% of training samples and achieves 93.8% recall under SVM), while F1 macro weighs all classes equally. Minority classes suffer substantially: Normal Start-Up, the smallest class with only 43 original training samples, achieves just 11.1% recall under Random Forest. This pattern confirms that class imbalance, rather than feature quality, is the primary limiting factor for classical ML on this task.

Feature importance analysis of the Tier 2 Random Forest provides insight into which acoustic properties carry the most diagnostic information. The top-ranked features are delta2_mfcc_0 (the second temporal derivative of the zeroth MFCC, capturing spectral energy acceleration), mfcc_0 (overall spectral energy level), and spectral_centroid (spectral brightness). All five of the highest-importance features are MFCC-based, indicating that cepstral representations of the spectral envelope are the dominant discriminative signal. Notably, retraining the Random Forest on only the 50 most important features yields an F1 macro of 0.672, compared to 0.679 with the full 441-dimensional feature vector — a negligible drop of 0.007. This high redundancy suggests that the 441-dimensional space contains substantial colinear information, and a carefully selected subset captures nearly all of the discriminative content. This observation has practical implications for deployment: compact neural networks operating on 13-coefficient MFCCs or 40-band mel spectrograms are not at an inherent disadvantage relative to the richer classical feature set.

[**Figure Placeholder:** Feature importance ranking for the Tier 2 Random Forest classifier, showing the top 20 features by Gini importance. See `results/feature_importance_tier2.png`.]

## 4.2 Data Augmentation Analysis

The data augmentation ablation study yields what is arguably the most practically significant finding of this project: synthetic waveform augmentation combined with class balancing (Condition B) produces the single largest accuracy improvement observed across all experiments.

### 4.2.1 Synthetic Augmentation and Class Balancing (Condition A to B)

Transitioning from the original, imbalanced training set of 970 samples (Condition A) to the synthetically augmented and class-balanced set of 3,317 samples (Condition B) dramatically improves classification performance. The DS-CNN (M6) exhibits the most pronounced gain, with its Tier 2 F1 macro jumping from 0.571 to 0.777 — an improvement of 0.206, the largest single delta observed in any experiment. The 2-D CNN (M2) improves by 0.066, and the 1-D CNN (M5) by 0.022. Two mechanisms work in concert to produce this result. First, the 3.4-fold expansion in training set size provides greater gradient diversity during optimization, exposing the model to a wider range of waveform variations through time shifting, pitch perturbation, speed modification, and Gaussian noise injection. Second, reducing the class imbalance ratio from 12.8:1 to approximately 1.1:1 eliminates the tendency for the loss function to be dominated by the majority class, enabling the model to learn discriminative features for minority classes such as Normal Start-Up and Braking Fault.

The impact on training stability is particularly revealing. Under Condition A, the DS-CNN was highly unstable at learning rates of 0.001 or higher, with validation accuracy oscillating near chance level (approximately 23%). Under Condition B, the same architecture trains stably at a learning rate of 0.002, converging to 91.4% validation accuracy. This behavior is consistent with the well-documented interaction between class imbalance and gradient noise in stochastic optimization: when one class overwhelms the loss landscape, gradient updates become biased and high learning rates amplify this instability. Class balancing smooths the loss surface, permitting more aggressive optimization.

### 4.2.2 Real-World Noise Injection (Condition B to C)

Adding real-world noise injection on top of the synthetic augmentation pipeline (Condition C) produces mixed and generally unfavorable results on the clean test set. The DS-CNN (M6) degrades slightly from 0.777 to 0.739 (−0.038), the 1-D CNN (M5) degrades from 0.638 to 0.596 (−0.042), and the 2-D CNN (M2) remains essentially flat at 0.732 versus 0.734. These results are not surprising when considered carefully. The test set consists of clean recordings without noise overlay, so a training regime that mixes in vehicular noise at SNR levels of 5–20 dB introduces a distributional shift between training and evaluation conditions. Furthermore, the noise bank was collected from a 2008 Volkswagen Jetta, while the training data originates from an unknown set of vehicles in the Kaggle dataset. This potential domain mismatch may cause the model to learn noise-specific features that do not generalize.

It is important to note, however, that the absence of improvement on the clean test set does not rule out improved robustness in noisy deployment environments. The noise injection strategy is designed to simulate the acoustic conditions that the Arduino's on-board PDM microphone would encounter in a real vehicle, and its benefits would be most apparent when evaluating against noisy test samples — a condition not directly measured in this study.

### 4.2.3 Combined Augmentation (Condition D)

The combined augmentation condition (D), which merges all synthetic and noise-injected samples into a training set of 5,664 examples, produces architecture-dependent results. The 2-D CNN (M2) benefits the most, achieving its best Tier 2 F1 macro of 0.751 — an improvement of 0.019 over Condition B. This suggests that the larger model (14,566 parameters) has sufficient capacity to leverage the additional data diversity without overfitting to noise artifacts. In contrast, the DS-CNN (M6) and 1-D CNN (M5) do not benefit from the combined condition; their performance plateaus or slightly degrades relative to Condition B. The smaller architectures (8,166 and 6,246 parameters, respectively) may be saturated at the Condition B data volume, and the doubled dataset introduces noise patterns that these models lack the capacity to disentangle from the diagnostic signal.

This finding carries a practical implication: more data is not universally beneficial. The quality, relevance, and balance of augmented samples matter as much as their quantity, and the optimal augmentation strategy is architecture-dependent.

[**Figure Placeholder:** Augmentation ablation study showing F1 macro by training condition (B, C, D) for each architecture at Tier 2. See `results/paper_ablation_tier2.png`.]

## 4.3 Quantization Analysis

A central contribution of this work is the systematic evaluation of int8 quantization on small audio classification models. The results are unambiguous: for models in the 6,000–15,000 parameter range, post-training quantization (PTQ) is effectively lossless, and quantization-aware training (QAT) provides no additional benefit.

### 4.3.1 Why Quantization Is Nearly Lossless

Several architectural properties of the models under study contribute to the negligible impact of int8 quantization. First, the small parameter counts (6,246 for M5, 8,166 for M6, 14,566 for M2) mean that fewer values are subject to quantization error, limiting the total accumulated distortion. Second, batch normalization layers are folded into their preceding convolutional layers during the quantization process, which reduces the number of arithmetic operations and maintains precision through the fused computation. Third, the use of global average pooling in all three architectures averages across spatial dimensions, which has a smoothing effect that attenuates per-element quantization noise. Finally, the classification task itself may present well-separated decision boundaries in the learned feature space, boundaries that do not require float32 precision to distinguish.

### 4.3.2 PTQ Versus QAT

Across all architectures at Tier 2, PTQ performs as well as or marginally better than QAT. For the DS-CNN (M6), PTQ achieves an F1 macro of 0.7835 compared to 0.7774 for QAT. For the 2-D CNN (M2), the scores are essentially tied at 0.7413 (PTQ) versus 0.7426 (QAT). This result is counterintuitive — QAT inserts fake quantization nodes during training so the model can adapt its weights to the quantization grid, which should in principle recover any accuracy lost to PTQ. However, for models of this size, QAT's additional fine-tuning with simulated quantization noise can introduce training instability. Since PTQ already produces negligible quantization error, there is nothing meaningful for QAT to recover, and the additional optimization complexity risks slightly perturbing an already well-positioned set of weights.

The practical implication is significant: for audio classification models with fewer than approximately 15,000 parameters, PTQ is sufficient. QAT adds pipeline complexity (requiring access to training data and the original training loop) without measurable benefit. QAT should be reserved for larger models or tasks where the classification boundaries are more sensitive to weight precision.

### 4.3.3 Prediction Agreement and Statistical Significance

Prediction agreement analysis confirms the quantitative picture. PTQ models agree with their float32 counterparts on 97–100% of individual predictions, meaning quantization changes the predicted class for fewer than 3% of test samples. Paradoxically, QAT models show lower agreement (92–97%) despite not improving F1, suggesting that QAT redistributes errors rather than eliminating them. McNemar's test applied to all float32-versus-int8 comparisons yields p-values exceeding 0.05 in every case, confirming that no model-quantization combination produces a statistically significant change in classification performance. Mean confidence shift (measured as the change in softmax output magnitude) is less than 0.002, indicating that the probabilistic outputs remain nearly identical after quantization.

[**Figure Placeholder:** Quantization impact on F1 macro, showing the delta between float32 and int8 variants (PTQ and QAT) for each architecture. See `results/paper_quantization_impact.png`.]

## 4.4 Deployment Performance Analysis

Deploying the M6 DS-CNN PTQ model on the Arduino Nano 33 BLE Sense Rev2 reveals that the system is functional and fits within all hardware constraints, but that inference latency is limited by toolchain deficiencies rather than by the hardware itself.

### 4.4.1 Inference Latency

The total classification cycle takes approximately 2,470 ms, decomposed into three phases. Audio capture accounts for 1,490 ms (60.3% of the cycle), which is fixed by the physical requirement to record 1.5 seconds of audio at 16 kHz. Feature extraction — computing a 40-band log-mel spectrogram from 92 overlapping FFT frames — requires 197 ms (8.0%), with the bulk of the computation handled by the CMSIS-DSP `arm_rfft_fast_f32()` function, which is well-optimized for the Cortex-M4's floating-point unit. Model inference consumes the remaining 783 ms (31.7%), dominated by the pointwise (1×1) convolutions in the DS-CNN's four depthwise separable blocks, which account for 62.8% of the model's approximately 6 million multiply-accumulate (MAC) operations.

At 783 ms of inference on a 64 MHz clock, the measured throughput is 8.4 cycles per MAC. This is consistent with published benchmarks for reference int8 kernels without SIMD acceleration (theoretical baseline of approximately 10 cycles/MAC), with the slight improvement attributable to compiler-level optimizations.

### 4.4.2 The CMSIS-NN SIMD Gap

A substantial investigation was conducted to enable CMSIS-NN's SIMD-optimized kernels, which leverage Cortex-M4 DSP extension instructions (such as `SMLAD`, `SXTB16`, and `PKHBT`) to process two int8 multiplications per clock cycle. ARM's published benchmarks indicate a target of approximately 3 cycles/MAC with full SIMD utilization, which would reduce inference time from 783 ms to roughly 282 ms.

TensorFlow Lite Micro was built from source with CMSIS-NN optimized kernels, and the Cortex-M4 target was correctly detected at compile time (`__ARM_FEATURE_DSP=1`). However, the Arduino Mbed OS board package bundles GCC 7.2.1 (released in 2017), whose `arm_acle.h` header lacks implementations for several critical ARM C Language Extension (ACLE) intrinsics, including `__sxtb16`, `__smlabb`, and `__smlatt`. Compatibility shims using inline assembly were provided, but the compiler did not reliably emit the corresponding SIMD instructions. A secondary approach — pre-compiling the TFLite Micro static library with GCC 14.3.1 (which correctly emitted 1,640 verified SIMD instructions) and linking it into the Arduino sketch — failed due to C++ ABI incompatibilities between the two compiler versions.

The measured gap is 2.8× slower than what full SIMD acceleration would achieve, corresponding to approximately 501 ms of avoidable latency per inference cycle. Upgrading to GCC 10 or later, via PlatformIO or the nRF Connect SDK, would unlock a roughly 3× speedup and reduce the total cycle time from 2.5 seconds to approximately 1.7 seconds.

### 4.4.3 Memory Utilization

The deployed system makes comfortable use of the Arduino's memory resources. Flash utilization stands at 319 KB (33% of the 983 KB usable), well below the budget. The use of `MicroMutableOpResolver` with only five registered operators (Conv2D, DepthwiseConv2D, FullyConnected, Mean, and Softmax) saved 165 KB of flash compared to `AllOpsResolver`, reducing usage from 484 KB to 319 KB. SRAM utilization is 179 KB (68% of 256 KB), leaving 83 KB for stack and heap — a comfortable margin. The TFLite tensor arena, allocated at 65,536 bytes, is 97% utilized (63,556 bytes consumed), indicating tight but functional packing. The quantized model itself occupies only 21.3 KB, far below the 150 KB flash budget, which suggests that substantially larger or more complex models could be accommodated if accuracy improvements were needed.

[**Figure Placeholder:** Latency breakdown showing the proportion of time spent in audio capture, feature extraction, and model inference. See `results/paper_latency_breakdown.png`.]

[**Figure Placeholder:** Memory utilization showing flash and SRAM usage relative to hardware budgets. See `results/paper_memory_utilization.png`.]

## 4.5 Playback Test Analysis

The playback test evaluates the complete end-to-end system by playing all 208 test clips through an external speaker while the Arduino captures and classifies the audio via its on-board PDM microphone. This test introduces the speaker-microphone domain gap — the acoustic distortion introduced by the playback chain — which is the most practically relevant evaluation for assessing deployment readiness.

### 4.5.1 Speaker Frequency Response as the Dominant Factor

The most striking finding from the playback evaluation is the overwhelming influence of speaker frequency response on classification accuracy. When test clips are played through laptop speakers, overall accuracy drops to 25.0%, with 72.1% of all predictions falling into startup-related classes regardless of the true label. This occurs because laptop speakers exhibit a steep frequency response roll-off below approximately 300 Hz, attenuating the low-frequency content (50–500 Hz) that characterizes engine idle and braking sounds — including engine rumble, brake rotor vibration, and mechanical hum. With the low-frequency signature removed, the remaining mid- and high-frequency spectral content more closely resembles the transient characteristics of startup sounds, causing systematic misclassification.

Switching to a Bluetooth speaker with improved bass response restores much of the low-frequency content and raises accuracy from 25.0% to 68.3% — a 2.7-fold improvement. This dramatic difference from a single hardware change underscores that the speaker, not the model, is the primary bottleneck in the playback evaluation paradigm.

### 4.5.2 Per-Class Degradation Patterns

The playback-induced accuracy degradation is not uniform across classes. Fault classes are relatively robust: Braking Fault achieves 81.8% recall (equivalent to the PC baseline), Start-Up Fault achieves 55.6% (also PC-equivalent), and Idle Fault achieves 78.0% (a modest 8.4 percentage point drop from the PC result). Normal classes, by contrast, degrade substantially: Normal Braking drops to 33.3% recall (−41.7 pp), Normal Start-Up to 44.4% (−33.4 pp), and Normal Idle to 57.5% (−10.0 pp).

A plausible explanation for this asymmetry is that fault sounds — worn brakes, a dead battery cranking, poor ignition — tend to produce distinctive mid- and high-frequency components (scraping, clicking, sputtering) that survive speaker reproduction with relatively little distortion. Normal sounds, by contrast, are characterized by smoother, lower-frequency tonal content (steady engine hum, smooth brake engagement) that is more vulnerable to speaker-induced attenuation. This interpretation aligns with the spectral analysis of the feature importance results in Section 4.1, which identified spectral energy and brightness as the dominant discriminative features.

### 4.5.3 Confidence Calibration

An encouraging finding is that the model's softmax confidence scores provide a useful signal for distinguishing correct from incorrect predictions in the playback setting. Correct predictions (n = 142, 68.3% of the test set) carry a mean confidence of 0.808, while incorrect predictions (n = 66, 31.7%) carry a mean confidence of 0.655 — a gap of 0.153. This separation is large enough to support threshold-based filtering: setting a confidence threshold at 0.70 would reject approximately 50% of incorrect predictions while retaining roughly 85% of correct ones. In a practical deployment, this mechanism could be combined with majority voting across consecutive inference cycles to further reduce the effective error rate.

### 4.5.4 Implications for Real-World Deployment

It is essential to note that the playback test represents a *conservative lower bound* on real-world performance, not a ceiling. In actual deployment, the Arduino's PDM microphone would be positioned near the vehicle's engine compartment, capturing the full frequency range of engine sounds directly — with no speaker in the signal path. The speaker-microphone domain gap that dominates the playback evaluation would be entirely absent. The Bluetooth speaker result of 68.3% accuracy therefore understates the performance that would be expected in practice. On-device accuracy would likely approach or exceed the PC-evaluated int8 result of 78.4% (M6 PTQ, Tier 2), since the on-device feature extraction pipeline has been validated to match the Python training pipeline with near-perfect fidelity (cosine similarity of 1.0000 across all 24 validation clips).

[**Figure Placeholder:** Per-class recall comparison between PC evaluation, Bluetooth speaker playback, and laptop speaker playback. See `results/paper_pc_vs_playback_recall.png`.]

[**Figure Placeholder:** Prediction distribution by speaker type, showing the shift toward startup classes under laptop playback. See `results/paper_speaker_ablation.png`.]

[**Figure Placeholder:** Confidence calibration histogram comparing the distribution of softmax confidence scores for correct versus incorrect predictions in the Bluetooth playback test. See `results/paper_confidence_calibration.png`.]
