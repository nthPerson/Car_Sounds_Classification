#ifndef CONFIG_H
#define CONFIG_H

// ─── Audio Capture ──────────────────────────────────────────────────────

#define SAMPLE_RATE       16000
#define AUDIO_DURATION_MS 1500
#define AUDIO_SAMPLES     24000   // SAMPLE_RATE * AUDIO_DURATION_MS / 1000
#define NUM_CHANNELS      1
#define PDM_GAIN          20      // PDM microphone gain (10, 20, 40, or 60)

// ─── Feature Extraction ─────────────────────────────────────────────────

#define N_FFT             512
#define HOP_LENGTH        256
#define N_MELS            40
#define N_FRAMES          92      // floor((AUDIO_SAMPLES - N_FFT) / HOP_LENGTH) + 1
#define N_FFT_BINS        257     // N_FFT / 2 + 1
#define SPECTROGRAM_SIZE  3680    // N_MELS * N_FRAMES
#define TOP_DB            80.0f   // librosa power_to_db clamp

// ─── Model ──────────────────────────────────────────────────────────────

#define MODEL_NAME        "DS-CNN (M6) Tier 2 PTQ"
#define NUM_CLASSES       6
#define TENSOR_ARENA_SIZE 65536   // 64 KB (M6 DS-CNN needs ~60 KB per AllocateTensors)

// ─── Serial ─────────────────────────────────────────────────────────────

#define SERIAL_BAUD       115200

// ─── Class Names ────────────────────────────────────────────────────────

const char* const CLASS_NAMES[NUM_CLASSES] = {
    "Normal Braking",
    "Braking Fault",
    "Normal Idle",
    "Idle Fault",
    "Normal Start-Up",
    "Start-Up Fault",
};

#endif // CONFIG_H
