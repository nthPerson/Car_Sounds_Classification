/*
 * Car Sound Classifier — Arduino Nano 33 BLE Sense Rev2
 *
 * Captures 1.5s of audio via the on-board PDM microphone, computes a
 * log-mel spectrogram, runs TFLite Micro int8 inference, and prints the
 * classification result over Serial.
 *
 * Hardware: Arduino Nano 33 BLE Sense Rev2 (nRF52840, Cortex-M4F @ 64 MHz)
 * Model:    DS-CNN (M6) Tier 2 PTQ — 6-class car sound classification
 *
 * Setup:
 *   1. Install "Arduino Mbed OS Nano Boards" via Boards Manager
 *   2. Install "TensorFlowLite_CMSIS_NN" library (custom build with
 *      CMSIS-NN optimized kernels for Cortex-M4 SIMD acceleration)
 *   3. Select Board: Arduino Nano 33 BLE
 *   4. Upload this sketch
 *   5. Open Serial Monitor at 115200 baud
 *
 * Serial Commands:
 *   "READY" — Trigger one capture-classify cycle (used by playback_test.py)
 *   "DUMP"  — Trigger one capture-classify cycle AND stream the raw
 *             log-mel spectrogram (40×92 floats, dB) over Serial. Used by
 *             src/compare_mel_spectrograms.py.
 *   "AUTO"  — Continuous classification.
 *   "STOP"  — Stop continuous classification.
 */

#include <PDM.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h>

#include "config.h"
#include "model_data.h"
#include "feature_extraction.h"

// ─── Buffers ────────────────────────────────────────────────────────────

// Audio capture buffer (48 KB)
int16_t audioBuffer[AUDIO_SAMPLES];
volatile int samplesRead = 0;
volatile bool captureComplete = false;

// Feature buffers
float melSpectrogram[SPECTROGRAM_SIZE];      // 14.7 KB
int8_t modelInput[SPECTROGRAM_SIZE];         // 3.6 KB

// Tensor arena for TFLite Micro
alignas(16) uint8_t tensorArena[TENSOR_ARENA_SIZE];

// TFLite globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputTensor = nullptr;
TfLiteTensor* outputTensor = nullptr;

float inputScale;
int inputZeroPoint;
float outputScale;
int outputZeroPoint;

int inferenceCount = 0;

// Serial input buffer for commands
char serialBuffer[64];
int serialPos = 0;

// ─── PDM Callback ───────────────────────────────────────────────────────

void onPDMdata() {
    int bytesAvailable = PDM.available();
    int samples = bytesAvailable / sizeof(int16_t);

    if (samplesRead + samples > AUDIO_SAMPLES) {
        samples = AUDIO_SAMPLES - samplesRead;
    }

    if (samples > 0) {
        PDM.read(&audioBuffer[samplesRead], samples * sizeof(int16_t));
        samplesRead += samples;
    }

    if (samplesRead >= AUDIO_SAMPLES) {
        captureComplete = true;
    }
}

// ─── Audio Capture ──────────────────────────────────────────────────────

bool captureClip() {
    captureComplete = false;
    samplesRead = 0;

    PDM.onReceive(onPDMdata);
    PDM.setGain(PDM_GAIN);
    if (!PDM.begin(NUM_CHANNELS, SAMPLE_RATE)) {
        Serial.println("ERROR: PDM.begin() failed");
        return false;
    }

    // Wait for buffer to fill (1.5 seconds of real-time audio)
    while (!captureComplete) {
        delay(10);
    }

    PDM.end();
    return true;
}

// ─── Mel-spectrogram dump (for compare_mel_spectrograms.py) ─────────────
//
// Streams the 40×92 log-mel spectrogram as text. Format:
//   MELDUMP_START|<n_mels>|<n_frames>
//   <row_0: n_frames floats, comma-separated>
//   ...
//   <row_(n_mels-1): n_frames floats, comma-separated>
//   MELDUMP_END
//
// Values are in dB (post librosa-style power_to_db, ref=np.max, top_db=80),
// matching the output of `compute_mel_spectrogram` BEFORE z-score
// normalization. This makes them directly comparable to a librosa
// spectrogram computed in Python.
void dumpMelSpectrogram(const float* mel) {
    Serial.print("MELDUMP_START|");
    Serial.print(N_MELS);
    Serial.print("|");
    Serial.println(N_FRAMES);
    for (int b = 0; b < N_MELS; b++) {
        for (int f = 0; f < N_FRAMES; f++) {
            Serial.print(mel[b * N_FRAMES + f], 3);
            if (f < N_FRAMES - 1) Serial.print(",");
        }
        Serial.println();
    }
    Serial.println("MELDUMP_END");
}

// ─── Inference ──────────────────────────────────────────────────────────

void runInference(bool dumpMel) {
    unsigned long t_total_start = micros();

    // 1. Capture audio
    Serial.println("Listening...");
    unsigned long t_capture_start = micros();
    if (!captureClip()) {
        Serial.println("ERROR: Audio capture failed");
        return;
    }
    float captureMs = (micros() - t_capture_start) / 1000.0f;

    // 2. Feature extraction
    unsigned long t_feat_start = micros();
    compute_mel_spectrogram(audioBuffer, AUDIO_SAMPLES, melSpectrogram);
    if (dumpMel) {
        dumpMelSpectrogram(melSpectrogram);
    }
    normalize_spectrogram(melSpectrogram);
    quantize_input(melSpectrogram, modelInput, inputScale, inputZeroPoint);
    float featMs = (micros() - t_feat_start) / 1000.0f;

    // 3. Copy input to model tensor
    memcpy(inputTensor->data.int8, modelInput, SPECTROGRAM_SIZE);

    // 4. Run inference
    unsigned long t_infer_start = micros();
    TfLiteStatus invokeStatus = interpreter->Invoke();
    float inferMs = (micros() - t_infer_start) / 1000.0f;

    if (invokeStatus != kTfLiteOk) {
        Serial.println("ERROR: Invoke() failed");
        return;
    }

    // 5. Read output and find prediction
    float probabilities[NUM_CLASSES];
    int bestClass = 0;
    float bestProb = -1.0f;

    for (int i = 0; i < NUM_CLASSES; i++) {
        int8_t rawOutput = outputTensor->data.int8[i];
        probabilities[i] = (rawOutput - outputZeroPoint) * outputScale;
        if (probabilities[i] > bestProb) {
            bestProb = probabilities[i];
            bestClass = i;
        }
    }

    float totalMs = (micros() - t_total_start) / 1000.0f;
    inferenceCount++;

    // 6. Print human-readable result
    Serial.print("--- Inference #");
    Serial.print(inferenceCount);
    Serial.println(" ---");
    Serial.print("Predicted: ");
    Serial.print(CLASS_NAMES[bestClass]);
    Serial.print(" (confidence: ");
    Serial.print(bestProb, 3);
    Serial.println(")");
    Serial.print("Timing: capture=");
    Serial.print(captureMs, 1);
    Serial.print("ms  feat=");
    Serial.print(featMs, 1);
    Serial.print("ms  infer=");
    Serial.print(inferMs, 1);
    Serial.print("ms  total=");
    Serial.print(totalMs, 1);
    Serial.println("ms");

    // 7. Print machine-readable result (for playback_test.py)
    Serial.print("RESULT|");
    Serial.print(inferenceCount);
    Serial.print("|");
    Serial.print(bestClass);
    Serial.print("|");
    Serial.print(bestProb, 4);
    Serial.print("|");
    Serial.print(featMs, 1);
    Serial.print("|");
    Serial.print(inferMs, 1);
    Serial.print("|");
    Serial.print(totalMs, 1);
    Serial.print("|");
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.print(probabilities[i], 4);
        if (i < NUM_CLASSES - 1) Serial.print(",");
    }
    Serial.println();
    Serial.println();
}

// ─── Setup ──────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) {
        delay(10);  // Wait for Serial Monitor
    }

    // Route TFLite Micro debug log to Arduino Serial
    RegisterDebugLogCallback([](const char* s) { Serial.print(s); });

    Serial.println("============================================================");
    Serial.println("  Car Sound Classifier — Arduino Nano 33 BLE Sense Rev2");
    Serial.println("============================================================");
    Serial.print("Model: ");
    Serial.println(MODEL_NAME);

    // CMSIS-NN SIMD diagnostic — verify compile-time feature detection
    Serial.print("CMSIS-NN SIMD: ");
#if defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1)
    Serial.println("ENABLED (__ARM_FEATURE_DSP=1)");
#else
    Serial.println("DISABLED (no __ARM_FEATURE_DSP — SIMD not active!)");
#endif
    Serial.print("Compiler: GCC ");
    Serial.print(__GNUC__);
    Serial.print(".");
    Serial.print(__GNUC_MINOR__);
    Serial.print(".");
    Serial.println(__GNUC_PATCHLEVEL__);

    // Initialize feature extraction (CMSIS-DSP FFT)
    feature_extraction_init();
    Serial.println("Feature extraction: initialized");

    // Load TFLite model
    model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("ERROR: Model schema version mismatch: ");
        Serial.print(model->version());
        Serial.print(" vs ");
        Serial.println(TFLITE_SCHEMA_VERSION);
        while (true) { delay(1000); }
    }

    // Op resolver — register only the ops used by M6 DS-CNN
    // (see model_data.h header comments for the full ops list)
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddMean();        // GlobalAveragePooling2D maps to Mean
    resolver.AddSoftmax();

    // Create interpreter
    static tflite::MicroInterpreter staticInterpreter(
        model, resolver, tensorArena, TENSOR_ARENA_SIZE);
    interpreter = &staticInterpreter;

    // Allocate tensors
    TfLiteStatus allocateStatus = interpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed — arena too small?");
        Serial.print("Arena size: ");
        Serial.print(TENSOR_ARENA_SIZE);
        Serial.println(" bytes");
        while (true) { delay(1000); }
    }

    // Get input/output tensors
    inputTensor = interpreter->input(0);
    outputTensor = interpreter->output(0);

    // Read quantization parameters
    inputScale = inputTensor->params.scale;
    inputZeroPoint = inputTensor->params.zero_point;
    outputScale = outputTensor->params.scale;
    outputZeroPoint = outputTensor->params.zero_point;

    // Print configuration
    Serial.print("Input: [");
    for (int i = 0; i < inputTensor->dims->size; i++) {
        Serial.print(inputTensor->dims->data[i]);
        if (i < inputTensor->dims->size - 1) Serial.print(", ");
    }
    Serial.print("], scale=");
    Serial.print(inputScale, 6);
    Serial.print(", zp=");
    Serial.println(inputZeroPoint);

    Serial.print("Output: ");
    Serial.print(NUM_CLASSES);
    Serial.println(" classes");

    Serial.print("Tensor arena: ");
    Serial.print(TENSOR_ARENA_SIZE);
    Serial.print(" bytes allocated");
    size_t usedBytes = interpreter->arena_used_bytes();
    if (usedBytes > 0) {
        Serial.print(" (");
        Serial.print(usedBytes);
        Serial.print(" used)");
    }
    Serial.println();

    Serial.println();
    Serial.println("Commands:");
    Serial.println("  READY    — Trigger one capture-classify cycle");
    Serial.println("  DUMP     — Trigger one cycle and stream the mel spectrogram");
    Serial.println("  AUTO     — Start continuous classification");
    Serial.println("  STOP     — Stop continuous classification");
    Serial.println();
    Serial.println("Mode: TRIGGERED (send 'READY' or 'AUTO' to begin)");
    Serial.println();
}

// ─── Loop ───────────────────────────────────────────────────────────────

bool continuousMode = false;

void loop() {
    // Check for serial commands
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            serialBuffer[serialPos] = '\0';
            if (serialPos > 0) {
                if (strncmp(serialBuffer, "READY", 5) == 0) {
                    // Triggered mode: run one inference
                    runInference(false);
                } else if (strncmp(serialBuffer, "DUMP", 4) == 0) {
                    // Triggered mode: run one inference AND dump spectrogram
                    runInference(true);
                } else if (strncmp(serialBuffer, "AUTO", 4) == 0) {
                    continuousMode = true;
                    Serial.println("Mode: CONTINUOUS");
                } else if (strncmp(serialBuffer, "STOP", 4) == 0) {
                    continuousMode = false;
                    Serial.println("Mode: TRIGGERED");
                }
            }
            serialPos = 0;
        } else if (serialPos < (int)sizeof(serialBuffer) - 1) {
            serialBuffer[serialPos++] = c;
        }
    }

    // Only run continuously if AUTO mode is enabled
    if (continuousMode) {
        runInference(false);
    }
}
