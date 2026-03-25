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
 *   2. Install "Arduino_TensorFlowLite" via Library Manager
 *   3. Select Board: Arduino Nano 33 BLE
 *   4. Upload this sketch
 *   5. Open Serial Monitor at 115200 baud
 *
 * Serial Commands:
 *   "READY" — Trigger one capture-classify cycle (used by playback_test.py)
 *   Clips are also classified continuously in a loop.
 */

#include <PDM.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/schema/schema_generated.h>

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

// ─── Inference ──────────────────────────────────────────────────────────

void runInference() {
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

    Serial.println("============================================================");
    Serial.println("  Car Sound Classifier — Arduino Nano 33 BLE Sense Rev2");
    Serial.println("============================================================");
    Serial.print("Model: ");
    Serial.println(MODEL_NAME);

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
    Serial.println("Ready. Send 'READY' to trigger classification,");
    Serial.println("or the classifier will run continuously.");
    Serial.println();
}

// ─── Loop ───────────────────────────────────────────────────────────────

void loop() {
    // Check for serial commands
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            serialBuffer[serialPos] = '\0';
            if (serialPos > 0) {
                // Process command
                if (strncmp(serialBuffer, "READY", 5) == 0) {
                    // Triggered mode: run one inference immediately
                    runInference();
                }
            }
            serialPos = 0;
        } else if (serialPos < (int)sizeof(serialBuffer) - 1) {
            serialBuffer[serialPos++] = c;
        }
    }

    // Continuous mode: run inference in a loop
    runInference();
}
