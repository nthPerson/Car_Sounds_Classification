# 2. Data Sources

## 2.1 Source Dataset

This study uses the Car Diagnostics Dataset published on Kaggle by Malak Ragaie, which contains 1,386 WAV recordings of automobile sounds spanning three operational states: braking, idle, and startup. The dataset captures 13 distinct conditions, including normal operation and various fault modes. Braking samples include normal braking and worn brakes. Idle samples encompass normal engine idle, low oil level, power steering fault, serpentine belt fault, and four combined-fault conditions (pairwise and triple combinations of the three single faults). Startup samples cover normal startup, bad ignition, and dead battery. The raw recordings exhibit heterogeneous sample rates (24 kHz, 44.1 kHz, and 48 kHz), and their durations differ slightly by operational state: braking and idle clips are approximately 1.5 seconds, while startup clips are approximately 1.7 seconds.

## 2.2 Classification Taxonomy

To support evaluation at multiple levels of diagnostic granularity, three hierarchical classification tiers were defined over the 13 source conditions.

**Tier 1 (Binary)** collapses all conditions into two classes — Normal versus Anomaly — providing a coarse screening objective. **Tier 2 (State-Aware, 6 classes)** serves as the primary deployment target and distinguishes normal operation from fault conditions within each operational state: Normal Braking, Braking Fault, Normal Idle, Idle Fault, Normal Start-Up, and Start-Up Fault. Each fault class aggregates all fault conditions for its state; for example, Idle Fault includes low oil, power steering fault, serpentine belt fault, and the combined-fault recordings. **Tier 3 (Individual Faults, 9 classes)** assigns a unique label to each single-fault condition (e.g., Worn Brakes, Low Oil, Bad Ignition), enabling fine-grained diagnosis. Because the combined-fault idle recordings do not map to a single fault, they are excluded from Tier 3 evaluation (assigned a sentinel label of −1) but are retained in Tiers 1 and 2, where they contribute to the Idle Fault aggregate class.

## 2.3 Train, Validation, and Test Splits

The 1,386 recordings were partitioned into training, validation, and test sets at a 70/15/15 ratio using stratified random sampling over the 13 source conditions, with a fixed random seed of 42 for reproducibility. This stratification ensures that every condition — including minority classes — appears in all three sets. The resulting splits contain 970 training samples, 208 validation samples, and 208 test samples. The validation set was used for hyperparameter tuning and early stopping during neural network training; the test set was reserved for final evaluation and used only once per model.

The Tier 2 class distribution in the training set reveals substantial imbalance. Idle Fault, which aggregates six underlying conditions, dominates with 552 samples (56.9%), while Normal Start-Up constitutes the smallest class with just 43 samples (4.4%). The resulting imbalance ratio of 12.8:1 presents a significant challenge for classifier training and motivated both class-weighted loss functions and the data augmentation strategy described in Section 2.6. The remaining training-set class counts are Normal Idle (185, 19.1%), Start-Up Fault (83, 8.6%), Normal Braking (54, 5.6%), and Braking Fault (53, 5.5%). Validation and test distributions mirror these proportions due to the stratified split.

`[Figure 2.1: Tier 2 class distribution across training, validation, and test sets — placeholder]`

## 2.4 Data Preprocessing

All audio files underwent a four-step standardization pipeline to ensure uniform input dimensions across the dataset and compatibility with the deployment target hardware.

First, every recording was resampled to 16 kHz mono, matching the sample rate of the MP34DT06JTR PDM microphone on the Arduino Nano 33 BLE Sense Rev2 deployment board. Second, each waveform was peak-normalized by dividing by its maximum absolute amplitude, scaling all signals to the range [−1, 1]. Third, durations were standardized to exactly 1.5 seconds (24,000 samples): clips shorter than 1.5 seconds were zero-padded, while the 1.7-second startup recordings were center-cropped to 1.5 seconds, preserving the acoustically informative central portion of the ignition event. These preprocessing parameters were serialized to a configuration file to ensure reproducibility across pipeline stages.

## 2.5 Feature Extraction

Three complementary feature representations were extracted from the preprocessed audio to serve different model architectures.

**Log-Mel Spectrograms.** For the 2-D Convolutional Neural Network (M2) and Depthwise Separable CNN (M6), log-mel spectrograms were computed using a 512-point FFT with a hop length of 256 samples and a 40-band mel filterbank spanning 20 Hz to 8,000 Hz. The resulting power spectrograms were converted to decibel scale using a reference-maximum normalization with an 80 dB dynamic range floor. Each spectrogram has shape (40, 92) — 40 mel bands by 92 time frames — and was expanded to (40, 92, 1) for compatibility with 2-D convolutional input layers. Per-band z-score normalization was applied using mean and standard deviation statistics computed exclusively from the training set, ensuring no information leakage from the validation or test partitions.

**Mel-Frequency Cepstral Coefficients (MFCCs).** For the 1-D CNN (M5), 13 MFCCs were extracted using the same FFT and mel filterbank parameters as the spectrogram pipeline, yielding feature matrices of shape (92, 13) — 92 time frames by 13 coefficients. Per-coefficient z-score normalization was again computed from training-set statistics only.

**Classical ML Feature Vectors.** For the Random Forest and SVM baselines (M1), a 441-dimensional feature vector was computed for each clip. Sixty-three frame-level features were first extracted: 13 MFCCs, 13 delta MFCCs, 13 delta-delta MFCCs, spectral centroid, spectral bandwidth, spectral rolloff (85th percentile), spectral contrast across 7 sub-bands, RMS energy, zero-crossing rate, and 12 chroma bins. Seven aggregation statistics (mean, standard deviation, minimum, maximum, median, skewness, and kurtosis) were then computed over the time axis for each of the 63 frame-level features, yielding 63 × 7 = 441 dimensions per sample.

All normalization statistics were computed separately for each training condition (A through D, described below) to account for distributional shifts introduced by augmentation. Separate statistics files were maintained and applied consistently to validation and test sets at inference time.

`[Figure 2.2: Example log-mel spectrograms for each Tier 2 class — placeholder]`

## 2.6 Data Augmentation

The severe class imbalance in the training set (12.8:1 ratio) and the modest dataset size (970 training samples) necessitated a data augmentation strategy with two objectives: balancing class frequencies and increasing the effective training-set diversity. An ablation study across four training conditions was designed to isolate the contributions of synthetic augmentation and real-world noise injection.

**Condition A (Original)** uses the unaugmented 970-sample training set with its natural class imbalance. **Condition B (Synthetic Augmentation)** applies five waveform-domain augmentations — time shift (±100 ms, 50% probability), additive Gaussian noise (SNR 5–25 dB, 50%), pitch shift (±2 semitones, 30%), speed perturbation (0.9–1.1× rate, 30%), and random gain (0.7–1.3× amplitude, 50%) — with per-class multipliers calibrated to bring every Tier 2 class to approximately 550 samples, matching the largest class (Idle Fault at 552). The resulting balanced training set contains 3,317 samples. **Condition C (Noise-Augmented)** applies the same waveform augmentations as Condition B, then additionally injects real-world noise (described in Section 2.7) into 100% of the augmented samples at signal-to-noise ratios uniformly sampled from {5, 10, 15, 20} dB, also yielding 3,317 samples. **Condition D (Combined)** merges the augmented samples from Conditions B and C, producing 5,664 training samples. To avoid duplicating the 970 original recordings, Condition D retains the full Condition B set (970 originals plus 2,347 synthetic copies) and appends only the 2,347 augmented-only copies from Condition C.

All augmentations were applied exclusively to the training set. The validation and test sets remained unaugmented throughout, ensuring fair comparison across conditions. Each augmented copy used a unique random seed for reproducibility.

`[Figure 2.3: Per-class sample counts across training conditions A through D — placeholder]`

## 2.7 Real-World Noise Bank

A distinguishing element of this study's data pipeline is the collection of a purpose-built noise bank recorded with the same hardware targeted for deployment. The Car Diagnostics Dataset was produced under controlled recording conditions, but the deployment environment — an Arduino microcontroller mounted in or near a vehicle — will encounter engine rumble, road noise, wind, HVAC airflow, and ambient urban sound. Bridging this domain gap required noise samples that capture not only the spectral characteristics of real automotive environments but also the frequency response, noise floor, and quantization artifacts of the deployment microphone itself.

**Collection hardware and protocol.** All noise samples were recorded on March 22, 2026, using an Arduino Nano 33 BLE Sense Rev2 with its on-board MP34DT06JTR PDM MEMS microphone — the identical hardware designated for inference. The Arduino was positioned either under the hood (near the engine bay) or inside the cabin of a 2008 Volkswagen Jetta, depending on the noise category. A custom Arduino sketch captured 1.5-second clips at 16 kHz and transmitted each as raw 16-bit integer binary data over a serial connection at 115,200 baud to a companion Python receiver script running on a laptop, which saved each clip as a WAV file and logged metadata (timestamp, label, sample rate, sample count, duration, and RMS energy) to a structured collection log.

**Noise categories.** A total of 360 noise samples were collected across 23 source labels, which were subsequently grouped into six acoustic categories for augmentation purposes. Engine Mechanical (103 samples) captures ambient under-hood sound, cold and warm engine idle, varying RPM, and cooling fan activity. HVAC (30 samples) spans three fan speed settings recorded under the hood. Road/Driving (60 samples) covers low-speed residential, moderate-speed city, and rough-road driving, all recorded under the hood. Cabin/Driving (60 samples) records interior cabin sound during city and highway driving with windows open and closed. Wind (20 samples) includes windy-day ambient and windy driving conditions. Ambient/Stationary (87 samples) encompasses parking lot, quiet residential, stoplight, busy street, and busy highway recordings from inside the cabin.

**Quality validation.** All 360 samples passed automated validation checks: every file confirmed a sample rate of exactly 16,000 Hz and a duration of exactly 1.5 seconds, with no clipping detected and no silent recordings (all clips exceeded an RMS energy threshold of 0.001).

**Operational-state-aware mixing.** During noise injection (Condition C), noise samples were not mixed randomly but were selected according to an operational-state relevance mapping. Braking samples received noise from the Engine Mechanical, Road/Driving, Cabin/Driving, Wind, and HVAC groups. Idle samples were mixed with Engine Mechanical, HVAC, Ambient/Stationary, and Wind noise. Startup samples received Ambient/Stationary, Wind, and HVAC noise. This mapping ensures that each training sample encounters noise types realistic for its operational context. The signal-to-noise ratio for each mixture was uniformly sampled from {5, 10, 15, 20} dB, producing a range from substantially corrupted (5 dB) to nearly clean (20 dB) conditions.

`[Figure 2.4: Spectrograms of representative noise samples from each of the six acoustic categories — placeholder]`

## 2.8 Summary of Data Pipeline

The complete data pipeline transforms 1,386 heterogeneous WAV recordings into standardized, feature-extracted, and optionally augmented training inputs across four experimental conditions. Preprocessing normalizes all audio to 16 kHz mono, 1.5-second clips. Three feature representations (log-mel spectrograms, MFCCs, and 441-dimensional classical feature vectors) serve the five model architectures evaluated in this study. Data augmentation addresses the 12.8:1 class imbalance through targeted oversampling with waveform perturbations, while a purpose-built noise bank of 360 real-world automotive recordings — collected with the deployment hardware itself — enables domain-realistic noise injection. Strict separation of training, validation, and test partitions is maintained at every stage: normalization statistics are computed from training data only, augmentation is applied to training data only, and the test set is used exactly once per model for final evaluation.
