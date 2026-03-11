# Quantization & Deployment Specification

**Parent document:** [project_overview.md](project_overview.md)
**Depends on:** [model_architecture_spec.md](model_architecture_spec.md), [data_preprocessing_spec.md](data_preprocessing_spec.md)
**Date:** March 10, 2026

---

## Table of Contents

1. [Quantization Overview](#1-quantization-overview)
2. [Post-Training Quantization (PTQ)](#2-post-training-quantization-ptq)
3. [Quantization-Aware Training (QAT)](#3-quantization-aware-training-qat)
4. [TFLite Model Conversion Pipeline](#4-tflite-model-conversion-pipeline)
5. [Model-to-C-Array Conversion](#5-model-to-c-array-conversion)
6. [Arduino Sketch Architecture](#6-arduino-sketch-architecture)
7. [On-Device Audio Capture](#7-on-device-audio-capture)
8. [On-Device Feature Extraction](#8-on-device-feature-extraction)
9. [On-Device Inference with TFLite Micro](#9-on-device-inference-with-tflite-micro)
10. [Output and Communication](#10-output-and-communication)
11. [Memory Layout and Optimization](#11-memory-layout-and-optimization)
12. [Arduino IDE Setup and Dependencies](#12-arduino-ide-setup-and-dependencies)
13. [Flashing and Debugging](#13-flashing-and-debugging)

---

## 1. Quantization Overview

### 1.1 What Is Quantization?

Quantization converts a neural network's weights and activations from 32-bit floating-point (float32) to lower-precision representations — in our case, 8-bit integers (int8). This provides three benefits critical for microcontroller deployment:

1. **Size reduction:** ~4× smaller model (each weight goes from 4 bytes to 1 byte)
2. **Speed improvement:** ~2–4× faster inference on Cortex-M4 (CMSIS-NN int8 kernels use SIMD to process 4 values per instruction)
3. **Memory reduction:** Intermediate activations are also int8, reducing tensor arena SRAM usage by ~4×

### 1.2 Quantization Scheme: Full Integer (Int8)

We use **full-integer quantization** where both weights AND activations are quantized to int8. This is required for:
- Maximum CMSIS-NN acceleration on Cortex-M4
- Compatibility with the TFLite Micro int8 kernel implementations
- Minimum memory footprint

The quantization mapping for each tensor is:

```
real_value = (int8_value - zero_point) × scale
```

Where `scale` (float32) and `zero_point` (int8) are per-tensor or per-channel quantization parameters computed during the conversion process.

### 1.3 Per-Channel vs. Per-Tensor Quantization

- **Per-tensor:** A single scale and zero_point for all values in a weight tensor. Simpler but less accurate.
- **Per-channel:** Separate scale and zero_point for each output channel of a convolution. More accurate because different channels can have very different weight distributions.

TFLite's default for Conv2D weights is **per-channel quantization**, which we will use. Activations use **per-tensor quantization** (since per-channel activation quantization has minimal benefit and higher runtime cost).

---

## 2. Post-Training Quantization (PTQ)

### 2.1 Method

PTQ converts a pre-trained float32 model to int8 WITHOUT retraining. It requires a **representative dataset** — a small subset of training data (~100–300 samples) used to calibrate the activation quantization ranges. The calibration process runs inference on each representative sample and records the min/max values of every activation tensor. These ranges are then used to compute the quantization parameters (scale, zero_point) for each activation.

### 2.2 Representative Dataset

```python
def representative_data_gen():
    """Generator that yields calibration samples for PTQ."""
    # Use a random subset of the training set
    indices = np.random.choice(len(X_train), size=200, replace=False)
    for idx in indices:
        sample = X_train[idx:idx+1].astype(np.float32)
        yield [sample]
```

**Requirements:**
- Must include samples from ALL classes (stratified sampling recommended)
- Must use the SAME preprocessing (normalization, feature extraction) as training
- 200 samples is typically sufficient; more samples give marginally better calibration but with diminishing returns
- The generator must yield one sample at a time (batch size 1)

### 2.3 Conversion Code

```python
import tensorflow as tf

# Load the trained float32 model
model = tf.keras.models.load_model('models/m2_cnn2d_float32/tier2_best.h5')

# Convert to TFLite with full-integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save
with open('models/m3_cnn2d_int8_ptq/tier2.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Int8 PTQ model size: {len(tflite_model) / 1024:.1f} KB")
```

### 2.4 Key Converter Settings Explained

| Setting | Value | Purpose |
|---------|-------|---------|
| `optimizations = [DEFAULT]` | Enables quantization | Tells the converter to quantize weights and activations |
| `representative_dataset` | Calibration generator | Provides data for activation range calibration |
| `supported_ops = [TFLITE_BUILTINS_INT8]` | Int8 only | Ensures ALL ops are quantized to int8 (no float32 fallback) |
| `inference_input_type = tf.int8` | Int8 input | Model expects int8 input (not float32). The input quantization scale/zero_point are stored in the model metadata. |
| `inference_output_type = tf.int8` | Int8 output | Model produces int8 output. Dequantize to float32 on the host for probability interpretation. |

### 2.5 Expected Impact

| Metric | Float32 Baseline | Int8 PTQ |
|--------|-----------------|----------|
| Model size | ~57 KB (2-D CNN) | ~16 KB |
| Accuracy drop | — | 0.5–2% typical |
| Inference speed (Cortex-M4) | 1× | 2–4× faster |

PTQ is the simplest quantization path and usually sufficient. If the accuracy drop exceeds 2%, escalate to QAT (Section 3).

---

## 3. Quantization-Aware Training (QAT)

### 3.1 Method

QAT inserts **fake quantization nodes** into the training graph. During forward passes, weights and activations are quantized to int8 and then dequantized back to float32 (simulating the precision loss). During backward passes, gradients flow through these nodes using the **straight-through estimator** (gradients are passed through as if the quantization didn't happen). This allows the model to learn to compensate for quantization effects during training.

### 3.2 Implementation

```python
import tensorflow_model_optimization as tfmot

# Start from the trained float32 model
model = tf.keras.models.load_model('models/m2_cnn2d_float32/tier2_best.h5')

# Apply QAT to the entire model
q_aware_model = tfmot.quantization.keras.quantize_model(model)

# Compile with the same settings as original training
q_aware_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune for a smaller number of epochs (the model is already trained)
q_aware_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,        # Fewer epochs — just fine-tuning, not training from scratch
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)
    ]
)
```

### 3.3 Converting QAT Model to Int8 TFLite

After QAT fine-tuning, the model already has quantization-aware weights. The conversion to int8 TFLite is simpler because the quantization ranges are already learned:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Note: representative_dataset is OPTIONAL for QAT models (ranges are already learned)
# But providing it can further refine activation ranges:
converter.representative_dataset = representative_data_gen

tflite_qat_model = converter.convert()

with open('models/m4_cnn2d_int8_qat/tier2.tflite', 'wb') as f:
    f.write(tflite_qat_model)
```

### 3.4 QAT Training Tips

| Consideration | Recommendation |
|--------------|---------------|
| **Starting point** | Always start QAT from a pre-trained float32 model (not from scratch). QAT is fine-tuning, not full training. |
| **Learning rate** | Use 10–100× lower learning rate than original training (e.g., 0.0001 if original was 0.001). The model is already close to optimal; QAT just needs to adjust for quantization effects. |
| **Epochs** | 10–30 epochs is usually sufficient. Monitor validation loss closely. |
| **Augmentation** | Continue using the same augmentation as original training. |
| **Batch normalization** | `tfmot.quantization.keras.quantize_model` handles BN folding automatically. BN layers are folded into preceding Conv layers during quantization. |

### 3.5 Expected Impact

| Metric | Float32 Baseline | Int8 PTQ | Int8 QAT |
|--------|-----------------|----------|----------|
| Model size | ~57 KB | ~16 KB | ~16 KB |
| Accuracy drop | — | 0.5–2% | **0–1%** |
| Inference speed | 1× | 2–4× | 2–4× |

QAT typically recovers 0.5–1% accuracy compared to PTQ, closing most of the gap to the float32 baseline.

---

## 4. TFLite Model Conversion Pipeline

### 4.1 Full Conversion Flow

```
Trained Keras model (.h5 or SavedModel)
        │
        ├──────────────────────────────────────────┐
        │                                          │
        ▼                                          ▼
   Float32 TFLite                           QAT Fine-Tuned Model
   (for size comparison only)                      │
        │                                          │
        ▼                                          ▼
   Int8 PTQ TFLite                          Int8 QAT TFLite
   (converter + representative data)        (converter)
        │                                          │
        ├──────────────────────────────────────────┤
        │                                          │
        ▼                                          ▼
   PC Evaluation                            PC Evaluation
   (accuracy, F1, confusion matrix)         (accuracy, F1, confusion matrix)
        │                                          │
        ├──────────────────────────────────────────┤
        │                                          │
        ▼                                          ▼
   Select best int8 model ──────────────────────────
        │
        ▼
   xxd -i model.tflite > model_data.h   (C byte array)
        │
        ▼
   Arduino sketch (includes model_data.h)
        │
        ▼
   Flash to Arduino Nano 33 BLE Sense Rev2
```

### 4.2 Validation on PC Before Deployment

Before deploying any int8 model to the Arduino, validate it on the PC using the TFLite interpreter:

```python
def evaluate_tflite_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a TFLite model on the test set."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input quantization parameters
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]

    predictions = []
    for i in range(len(X_test)):
        # Quantize input to int8
        sample = X_test[i:i+1].astype(np.float32)
        quantized_input = (sample / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], quantized_input)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output))

    predictions = np.array(predictions)
    accuracy = np.mean(predictions == y_test)
    return {'accuracy': accuracy, 'predictions': predictions}
```

This PC-based TFLite evaluation gives the exact same results as on-device inference (the TFLite interpreter is bit-exact). It's much faster to iterate on PC than to flash the Arduino for each test.

---

## 5. Model-to-C-Array Conversion

### 5.1 Using xxd

The TFLite model file must be converted to a C byte array that can be compiled into the Arduino sketch:

```bash
xxd -i model.tflite > model_data.h
```

This produces a header file like:

```c
// model_data.h (auto-generated)
unsigned char model_tflite[] = {
  0x20, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
unsigned int model_tflite_len = 16384;
```

### 5.2 Aligning the Model Data

For optimal memory access on Cortex-M4, align the model data to a 16-byte boundary:

```c
// model_data.h (modified)
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

alignas(16) const unsigned char model_tflite[] = {
  0x20, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
const unsigned int model_tflite_len = 16384;

#endif // MODEL_DATA_H
```

### 5.3 Storing in Flash (PROGMEM)

On the nRF52840, the model data is automatically stored in flash (not SRAM) when declared as `const`. No explicit `PROGMEM` attribute is needed (unlike AVR-based Arduinos). The TFLite Micro interpreter reads model data directly from flash.

---

## 6. Arduino Sketch Architecture

### 6.1 High-Level Structure

```
car_sound_classifier/
├── car_sound_classifier.ino    # Main sketch
├── model_data.h                # Quantized model as C byte array
├── feature_extraction.h/.cpp   # Mel-spectrogram computation
├── mel_filterbank.h            # Pre-computed mel filterbank coefficients
└── config.h                    # Configuration constants
```

### 6.2 Main Loop (Pseudocode)

```c
#include <PDM.h>
#include <TensorFlowLite.h>
#include "model_data.h"
#include "feature_extraction.h"
#include "config.h"

// ---- Global buffers ----
int16_t audio_buffer[AUDIO_SAMPLES];   // 24,000 samples × 2 bytes = 48 KB
float mel_spectrogram[N_MELS * N_FRAMES];  // 40 × 92 = 3,680 floats = ~14.7 KB
int8_t model_input[N_MELS * N_FRAMES];    // 3,680 bytes = ~3.6 KB
uint8_t tensor_arena[TENSOR_ARENA_SIZE];   // Pre-allocated SRAM for TFLite

// ---- TFLite Micro objects ----
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;

void setup() {
    Serial.begin(115200);

    // Initialize PDM microphone
    PDM.onReceive(onPDMdata);
    PDM.begin(1, SAMPLE_RATE);  // 1 channel, 16 kHz

    // Load model from flash
    model = tflite::GetModel(model_tflite);

    // Create interpreter with only the ops needed by our model
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();

    Serial.println("Car Sound Classifier ready.");
}

void loop() {
    // 1. Capture 1.5 seconds of audio
    capture_audio(audio_buffer, AUDIO_SAMPLES);

    // 2. Compute mel-spectrogram
    compute_mel_spectrogram(audio_buffer, AUDIO_SAMPLES,
                            mel_spectrogram, N_MELS, N_FRAMES);

    // 3. Quantize mel-spectrogram to int8 for model input
    quantize_features(mel_spectrogram, model_input, N_MELS * N_FRAMES,
                      INPUT_SCALE, INPUT_ZERO_POINT);

    // 4. Copy to model input tensor
    int8_t* input_tensor = interpreter->input(0)->data.int8;
    memcpy(input_tensor, model_input, N_MELS * N_FRAMES);

    // 5. Run inference
    unsigned long t_start = micros();
    TfLiteStatus status = interpreter->Invoke();
    unsigned long t_inference = micros() - t_start;

    if (status != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }

    // 6. Read output and find predicted class
    int8_t* output = interpreter->output(0)->data.int8;
    float output_scale = interpreter->output(0)->params.scale;
    int output_zero_point = interpreter->output(0)->params.zero_point;

    int predicted_class = 0;
    float max_score = -1e9;
    for (int i = 0; i < NUM_CLASSES; i++) {
        float score = (output[i] - output_zero_point) * output_scale;
        if (score > max_score) {
            max_score = score;
            predicted_class = i;
        }
    }

    // 7. Output result
    Serial.print("Predicted: ");
    Serial.print(CLASS_NAMES[predicted_class]);
    Serial.print(" (confidence: ");
    Serial.print(max_score, 3);
    Serial.print(", inference time: ");
    Serial.print(t_inference / 1000.0, 1);
    Serial.println(" ms)");
}
```

### 6.3 Configuration Constants (`config.h`)

```c
#ifndef CONFIG_H
#define CONFIG_H

// Audio capture
#define SAMPLE_RATE       16000
#define AUDIO_DURATION_S  1.5f
#define AUDIO_SAMPLES     ((int)(SAMPLE_RATE * AUDIO_DURATION_S))  // 24000

// Feature extraction
#define N_FFT             512
#define HOP_LENGTH        256
#define N_MELS            40
#define N_FRAMES          92    // floor((24000 - 512) / 256) + 1
#define MEL_FMIN          20.0f
#define MEL_FMAX          8000.0f

// Model
#define NUM_CLASSES       6     // Tier 2
#define TENSOR_ARENA_SIZE 81920 // 80 KB — adjust based on profiling
#define INPUT_SCALE       0.00392157f  // Placeholder — actual value from TFLite model
#define INPUT_ZERO_POINT  -128         // Placeholder — actual value from TFLite model

// Class names
const char* CLASS_NAMES[] = {
    "Normal Braking",
    "Braking Fault",
    "Normal Idle",
    "Idle Fault",
    "Normal Start-Up",
    "Start-Up Fault"
};

#endif // CONFIG_H
```

---

## 7. On-Device Audio Capture

### 7.1 PDM Library Usage

The Arduino `PDM` library handles all the low-level details of the MP34DT06JTR MEMS microphone:
- Configures the PDM clock and data pins
- Runs a decimation filter on the 1-bit PDM bitstream
- Delivers 16-bit signed PCM samples via an interrupt-driven callback

### 7.2 Capture Strategy: Block Capture (Simple)

For the initial implementation, use a simple block-capture approach:

```c
volatile bool audio_ready = false;
volatile int samples_read = 0;

void onPDMdata() {
    int bytes_available = PDM.available();
    int samples = bytes_available / 2;  // 16-bit samples = 2 bytes each

    PDM.read(&audio_buffer[samples_read], bytes_available);
    samples_read += samples;

    if (samples_read >= AUDIO_SAMPLES) {
        audio_ready = true;
        samples_read = AUDIO_SAMPLES;  // Cap to prevent overflow
    }
}

void capture_audio(int16_t* buffer, int num_samples) {
    audio_ready = false;
    samples_read = 0;

    // Wait for buffer to fill (blocking)
    while (!audio_ready) {
        // At 16 kHz, filling 24,000 samples takes 1.5 seconds
        delay(10);
    }

    // Stop PDM to prevent buffer overwrite during processing
    PDM.end();
}
```

**Timing:** Audio capture takes exactly 1.5 seconds (real-time). This is the dominant latency in each inference cycle. Feature extraction and inference add ~100–300 ms on top.

### 7.3 Future Enhancement: Sliding Window (Continuous)

For near-real-time operation, a sliding window approach captures audio continuously and processes overlapping windows:

1. Use a **circular buffer** (ring buffer) of 1.5 s
2. After filling the buffer, extract features and run inference
3. While inference is running, continue capturing into the buffer
4. Shift the window forward by 0.75 s (50% overlap) and repeat

This produces a classification every ~0.75 s but is significantly more complex to implement (requires double-buffering or ring buffer management without blocking). **Defer to Phase 5 if time permits.**

---

## 8. On-Device Feature Extraction

### 8.1 Overview

On-device feature extraction must replicate the training pipeline's mel-spectrogram computation. The steps are:

1. **Apply Hann window** to each frame of the audio signal
2. **Compute FFT** (real FFT, size 512) for each frame
3. **Compute power spectrum** (magnitude squared)
4. **Apply mel filterbank** (40 mel filters)
5. **Apply log compression** (log10 or natural log + small epsilon)

### 8.2 Implementation Strategy: Float32 with CMSIS-DSP

We compute features in float32 on the Cortex-M4F, leveraging the hardware FPU and CMSIS-DSP library:

```c
#include "arm_math.h"     // CMSIS-DSP
#include "arm_const_structs.h"

// Pre-computed constants (stored in flash)
static const float hann_window[N_FFT];        // 512 Hann window coefficients
static const float mel_filterbank[N_MELS][N_FFT/2 + 1]; // 40 × 257 mel filter weights

void compute_mel_spectrogram(const int16_t* audio, int audio_len,
                             float* output, int n_mels, int n_frames) {
    float frame[N_FFT];
    float fft_output[N_FFT];       // Complex output (interleaved real/imag)
    float power_spectrum[N_FFT/2 + 1];

    arm_rfft_fast_instance_f32 fft_instance;
    arm_rfft_fast_init_f32(&fft_instance, N_FFT);

    for (int t = 0; t < n_frames; t++) {
        int offset = t * HOP_LENGTH;

        // 1. Extract frame and convert int16 → float32
        for (int i = 0; i < N_FFT; i++) {
            if (offset + i < audio_len) {
                frame[i] = (float)audio[offset + i] / 32768.0f;
            } else {
                frame[i] = 0.0f;  // Zero-pad if needed
            }
        }

        // 2. Apply Hann window
        arm_mult_f32(frame, hann_window, frame, N_FFT);

        // 3. Compute real FFT
        arm_rfft_fast_f32(&fft_instance, frame, fft_output, 0);

        // 4. Compute power spectrum (magnitude squared)
        // fft_output is interleaved [re0, im0, re1, im1, ...]
        arm_cmplx_mag_squared_f32(fft_output, power_spectrum, N_FFT / 2);
        // Handle DC and Nyquist bins specially
        power_spectrum[0] = fft_output[0] * fft_output[0];
        power_spectrum[N_FFT/2] = fft_output[1] * fft_output[1];

        // 5. Apply mel filterbank (matrix-vector multiply)
        for (int m = 0; m < N_MELS; m++) {
            float mel_energy = 0.0f;
            arm_dot_prod_f32(mel_filterbank[m], power_spectrum,
                           N_FFT/2 + 1, &mel_energy);
            // 6. Log compression
            output[m * n_frames + t] = log10f(mel_energy + 1e-10f);
        }
    }
}
```

### 8.3 Pre-Computed Mel Filterbank

The mel filterbank matrix is computed on the PC (using librosa's `mel` function) and embedded as a constant array in the Arduino sketch. This ensures exact parity with the training pipeline's filterbank.

```python
# Generate mel filterbank on PC
import librosa

mel_filterbank = librosa.filters.mel(
    sr=16000, n_fft=512, n_mels=40, fmin=20, fmax=8000
)
# mel_filterbank shape: (40, 257)

# Export to C header
with open('mel_filterbank.h', 'w') as f:
    f.write('#ifndef MEL_FILTERBANK_H\n#define MEL_FILTERBANK_H\n\n')
    f.write(f'#define MEL_FILTER_ROWS {mel_filterbank.shape[0]}\n')
    f.write(f'#define MEL_FILTER_COLS {mel_filterbank.shape[1]}\n\n')
    f.write('const float mel_filterbank[MEL_FILTER_ROWS][MEL_FILTER_COLS] = {\n')
    for row in mel_filterbank:
        f.write('  {' + ', '.join(f'{v:.8f}' for v in row) + '},\n')
    f.write('};\n\n')
    f.write('#endif\n')
```

### 8.4 Memory Usage for Feature Extraction

| Buffer | Size | Notes |
|--------|------|-------|
| `frame[512]` (float32) | 2 KB | Reused for each frame |
| `fft_output[512]` (float32) | 2 KB | Complex output, reused |
| `power_spectrum[257]` (float32) | ~1 KB | Reused |
| `mel_filterbank[40][257]` (float32, in flash) | ~40 KB flash | Stored in flash, not SRAM |
| `hann_window[512]` (float32, in flash) | 2 KB flash | Stored in flash |
| `mel_spectrogram[40×92]` (float32) | ~14.7 KB SRAM | Full spectrogram output |
| **Total SRAM for feature extraction** | **~20 KB** | |

### 8.5 Optimization: Sparse Mel Filterbank

The mel filterbank matrix is very sparse — each mel filter is a triangular window that only overlaps a small number of FFT bins. Instead of a dense matrix multiply (40 × 257 = 10,280 multiply-adds per frame), we can store only the non-zero filter weights and their indices:

```c
typedef struct {
    int start_bin;     // First non-zero FFT bin
    int num_weights;   // Number of non-zero weights
    const float* weights;  // Pointer to weight array (in flash)
} MelFilter;

MelFilter mel_filters[N_MELS];
```

This reduces the mel filterbank application from O(n_mels × n_fft/2) to O(n_mels × avg_filter_width), which is typically 5–10× faster. The non-zero weights array is much smaller (~500 floats instead of 10,280), saving ~39 KB of flash.

---

## 9. On-Device Inference with TFLite Micro

### 9.1 TFLite Micro Setup

```c
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Only register the ops used by our model (minimizes flash usage)
static tflite::MicroMutableOpResolver<8> resolver;

void setup_tflite() {
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();  // Only if using DS-CNN
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();    // Only if using GlobalAvgPool (mapped to AvgPool)
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();         // For input quantization

    static tflite::MicroInterpreter static_interpreter(
        tflite::GetModel(model_tflite),
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE
    );
    interpreter = &static_interpreter;

    TfLiteStatus status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed.");
        Serial.print("Required arena size may exceed ");
        Serial.print(TENSOR_ARENA_SIZE);
        Serial.println(" bytes.");
        while (1);  // Halt
    }

    // Print model info
    TfLiteTensor* input = interpreter->input(0);
    Serial.print("Input shape: ");
    for (int i = 0; i < input->dims->size; i++) {
        Serial.print(input->dims->data[i]);
        if (i < input->dims->size - 1) Serial.print(" x ");
    }
    Serial.println();
    Serial.print("Input type: ");
    Serial.println(input->type == kTfLiteInt8 ? "int8" : "other");
    Serial.print("Input scale: ");
    Serial.println(input->params.scale, 8);
    Serial.print("Input zero_point: ");
    Serial.println(input->params.zero_point);
}
```

### 9.2 Input Quantization

The model expects int8 input. We must convert the float32 mel-spectrogram to int8 using the model's input quantization parameters:

```c
void quantize_features(const float* features, int8_t* output,
                       int num_values, float scale, int zero_point) {
    for (int i = 0; i < num_values; i++) {
        int32_t quantized = (int32_t)roundf(features[i] / scale) + zero_point;
        // Clamp to int8 range
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        output[i] = (int8_t)quantized;
    }
}
```

The `scale` and `zero_point` values are read from the TFLite model at setup time:

```c
float input_scale = interpreter->input(0)->params.scale;
int input_zero_point = interpreter->input(0)->params.zero_point;
```

### 9.3 Output Dequantization

The model outputs int8 softmax probabilities. To get human-readable confidence scores:

```c
void dequantize_output(const int8_t* output, float* probabilities,
                       int num_classes, float scale, int zero_point) {
    for (int i = 0; i < num_classes; i++) {
        probabilities[i] = (output[i] - zero_point) * scale;
    }
}
```

### 9.4 CMSIS-NN Integration

TFLite Micro automatically delegates to CMSIS-NN optimized kernels when:
1. The target is a Cortex-M4 or Cortex-M7
2. The model uses int8 quantization
3. The `CMSIS_NN` build flag is enabled

With the Arduino IDE + `Arduino_TensorFlowLite` library, CMSIS-NN integration is typically automatic. Verify by checking the compile output for CMSIS-NN function calls (e.g., `arm_convolve_s8`, `arm_depthwise_conv_s8`).

---

## 10. Output and Communication

### 10.1 Serial Output (Debug/Evaluation)

During development and on-device evaluation, output results over Serial (USB):

```
Car Sound Classifier ready.
Model: Compact CNN (Tier 2, Int8 QAT)
Input: 40x92x1, Output: 6 classes
Tensor arena: 73,728 bytes allocated

--- Inference #1 ---
Predicted: Idle Fault (confidence: 0.847)
Inference time: 123.4 ms
Total cycle time: 1623.4 ms

--- Inference #2 ---
Predicted: Normal Idle (confidence: 0.932)
Inference time: 122.8 ms
Total cycle time: 1622.8 ms
```

### 10.2 BLE Output (Stretch Goal)

If time permits, transmit results over Bluetooth Low Energy using the `ArduinoBLE` library:

```c
#include <ArduinoBLE.h>

BLEService classifierService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLEByteCharacteristic predictedClass("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);
BLEFloatCharacteristic confidence("19B10002-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);
```

**Note:** BLE adds ~20–40 KB to flash usage and ~10–15 KB to SRAM usage. This further tightens the memory budget. Only enable BLE after confirming that inference works correctly with Serial output.

---

## 11. Memory Layout and Optimization

### 11.1 Complete SRAM Budget (Tier 2, 2-D CNN)

| Component | SRAM Usage | Notes |
|-----------|-----------|-------|
| Mbed OS + Arduino core + stack | ~40 KB | Includes interrupt handlers, heap |
| Audio buffer (`int16_t[24000]`) | 48 KB | Raw PDM samples |
| Feature extraction workspace | ~20 KB | FFT buffers, mel output (float32) |
| Model input (int8) | ~3.6 KB | 40 × 92 × 1 |
| TFLite tensor arena | ~75 KB | Model activations during inference |
| TFLite interpreter overhead | ~10 KB | Op resolver, tensor metadata |
| Serial/misc buffers | ~5 KB | |
| **Total** | **~202 KB** | Out of 256 KB available |
| **Headroom** | **~54 KB** | Sufficient margin |

### 11.2 Memory Optimization Techniques

If the budget is too tight:

1. **Reduce audio buffer:** Use 1.0 s window instead of 1.5 s → saves 16 KB SRAM
2. **Reduce mel bands:** Use 32 instead of 40 → reduces tensor arena by ~15%
3. **Reuse audio buffer for features:** After feature extraction is complete, the audio buffer is no longer needed. By structuring the code carefully, we can overlay the mel-spectrogram output onto part of the audio buffer memory. (Requires careful pointer management; fragile but effective.)
4. **Use the 1-D CNN (M5):** Tensor arena drops from ~75 KB to ~20 KB, saving ~55 KB
5. **Reduce DS-CNN channels:** Already addressed in [model_architecture_spec.md](model_architecture_spec.md)

### 11.3 Flash Budget

| Component | Flash Usage | Notes |
|-----------|------------|-------|
| Mbed OS + Arduino core | ~300 KB | |
| PDM library | ~15 KB | |
| TFLite Micro runtime + CMSIS-NN | ~80 KB | With MicroMutableOpResolver (only needed ops) |
| Feature extraction code | ~20 KB | FFT, mel filterbank application |
| Mel filterbank data (sparse) | ~5 KB | Sparse representation (~500 floats) |
| Hann window data | 2 KB | 512 floats |
| Application code + config | ~10 KB | |
| Int8 model (2-D CNN, Tier 2) | ~16 KB | |
| **Total** | **~448 KB** | Out of 1,024 KB available |
| **Headroom** | **~576 KB** | Very generous |

Flash is NOT the bottleneck. SRAM is the limiting factor.

---

## 12. Arduino IDE Setup and Dependencies

### 12.1 Board Support

1. Open Arduino IDE (2.x recommended)
2. Go to **Tools → Board → Boards Manager**
3. Search for "Arduino Mbed OS Nano Boards" and install (includes nRF52840 support)
4. Select **Tools → Board → Arduino Mbed OS Nano Boards → Arduino Nano 33 BLE**

### 12.2 Required Libraries

| Library | Version | Purpose | Install Method |
|---------|---------|---------|---------------|
| `Arduino_TensorFlowLite` | ≥ 2.4.0 | TFLite Micro runtime + CMSIS-NN | Library Manager or GitHub |
| `PDM` | Built-in | PDM microphone driver | Pre-installed with board support |
| `ArduinoBLE` | ≥ 1.3.0 | Bluetooth Low Energy (stretch goal) | Library Manager |

**Important note on TFLite library:** The `Arduino_TensorFlowLite` library may be outdated in the Library Manager. The most reliable source is building from the [tflite-micro](https://github.com/tensorflow/tflite-micro) repository and generating an Arduino-compatible library using:

```bash
# From the tflite-micro repo:
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
    --makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn" \
    /output/path
```

This ensures CMSIS-NN integration and the latest op kernels.

### 12.3 Compilation Settings

- **Optimization level:** The Arduino IDE compiles with `-O2` by default for Mbed OS targets. This is appropriate for production.
- **Debug builds:** For development, adding `-DTFLITE_MCU_DEBUG_LOG` enables TFLite Micro debug output via `Serial`.

### 12.4 Verifying Compilation Output

After compiling, the Arduino IDE reports flash and SRAM usage:

```
Sketch uses 448,512 bytes (43%) of program storage space. Maximum is 1,048,576 bytes.
Global variables use 201,728 bytes (78%) of dynamic memory. Maximum is 262,144 bytes.
```

The "Global variables" number is the static SRAM allocation — the tensor arena is typically statically allocated. If this exceeds ~220 KB (leaving too little for the stack and heap), the sketch needs optimization.

---

## 13. Flashing and Debugging

### 13.1 Flashing

1. Connect the Arduino Nano 33 BLE Sense Rev2 via Micro-USB
2. Select the correct port in **Tools → Port**
3. Click **Upload** (or Ctrl+U)
4. The board resets automatically after flashing

If the board becomes unresponsive, double-tap the reset button to enter bootloader mode (the built-in LED pulses). Re-select the port and upload.

### 13.2 Serial Monitor

Open **Tools → Serial Monitor** (or Ctrl+Shift+M). Set baud rate to 115200. The sketch outputs:
- Model information at startup
- Classification results for each inference cycle
- Inference latency measurements

### 13.3 Common Issues

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| `AllocateTensors() failed` | Tensor arena too small | Increase `TENSOR_ARENA_SIZE`, or use a smaller model |
| Very slow inference (>1 second) | CMSIS-NN not enabled, or float32 fallback | Verify int8 model; check CMSIS-NN in build |
| No audio captured | PDM library issue | Check `PDM.begin()` return value; verify board revision |
| Wrong classifications | Training-serving skew | Compare on-device spectrograms with PC spectrograms |
| Sketch too large (>1 MB) | Too many TFLite ops registered | Use `MicroMutableOpResolver` with only needed ops |
| SRAM overflow (>256 KB) | Tensor arena + buffers too large | Reduce model size, reduce audio buffer, reduce mel bands |
