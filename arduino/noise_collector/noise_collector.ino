/*
 * Background Noise Data Collector — Arduino Nano 33 BLE Sense Rev2
 *
 * Purpose: Collect labeled background noise samples for data augmentation.
 *          Records 1.5s audio clips via the on-board PDM microphone and
 *          transmits them as binary data over Serial to a PC-side Python
 *          script (src/collect_noise.py) which saves them as WAV files.
 *
 * Hardware: Arduino Nano 33 BLE Sense Rev2 (nRF52840)
 * Library:  PDM (built-in with Arduino Mbed OS Nano Boards package)
 *
 * Setup:
 *   1. Install "Arduino Mbed OS Nano Boards" via Boards Manager
 *   2. Select Board: Arduino Nano 33 BLE
 *   3. Upload this sketch
 *   4. Run: python src/collect_noise.py --port /dev/ttyACM0
 *      (or use Arduino IDE Serial Monitor at 115200 baud for testing)
 *
 * Commands (send via Serial Monitor with "Newline" line ending):
 *
 *   label <name>   Set the label for subsequent recordings.
 *                   Use underscores for spaces (e.g., "label road_60mph").
 *
 *   record         Capture one 1.5s audio clip and send to PC.
 *
 *   record <N>     Capture N consecutive clips (e.g., "record 10").
 *                   ~2 second gap between clips.
 *
 *   gain <value>   Set PDM microphone gain (10, 20, 40, or 60).
 *
 *   status         Show current label and gain.
 *
 *   help           Print the command menu.
 *
 * Binary Transfer Protocol:
 *   Each clip is sent as:
 *     "=== BEGIN_AUDIO ===" (text line)
 *     "LABEL:<name>"        (text line)
 *     "SAMPLE_RATE:16000"   (text line)
 *     "NUM_SAMPLES:24000"   (text line)
 *     "FORMAT:INT16_BINARY" (text line)
 *     "=== DATA ==="        (text line)
 *     <48000 bytes of raw int16_t samples, little-endian>
 *     "=== END_AUDIO ==="   (text line)
 */

#include <PDM.h>

// ─── Configuration ───────────────────────────────────────────────────────

#define SAMPLE_RATE       16000
#define AUDIO_DURATION_MS 1500
#define AUDIO_SAMPLES     (SAMPLE_RATE * AUDIO_DURATION_MS / 1000)  // 24000
#define NUM_CHANNELS      1
#define SERIAL_BAUD       115200
#define MAX_LABEL_LEN     64
#define INPUT_BUF_LEN     128

// ─── Buffers ─────────────────────────────────────────────────────────────

int16_t audioBuffer[AUDIO_SAMPLES];       // 48 KB — main audio capture buffer
volatile int samplesRead = 0;
volatile bool captureComplete = false;

char currentLabel[MAX_LABEL_LEN] = "";    // Current recording label
char inputBuffer[INPUT_BUF_LEN];          // Serial input buffer
int inputPos = 0;

int currentGain = 20;                     // PDM gain (10, 20, 40, 60)

// ─── PDM Callback ────────────────────────────────────────────────────────

void onPDMdata() {
  int bytesAvailable = PDM.available();
  int samples = bytesAvailable / sizeof(int16_t);

  // Cap to prevent buffer overflow
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

// ─── Audio Capture ───────────────────────────────────────────────────────

bool captureClip() {
  // Reset state
  captureComplete = false;
  samplesRead = 0;

  // Start PDM
  PDM.onReceive(onPDMdata);
  PDM.setGain(currentGain);
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

// ─── Binary Transmission ─────────────────────────────────────────────────

void sendClip() {
  // Text header
  Serial.println("=== BEGIN_AUDIO ===");
  Serial.print("LABEL:");
  Serial.println(currentLabel);
  Serial.print("SAMPLE_RATE:");
  Serial.println(SAMPLE_RATE);
  Serial.print("NUM_SAMPLES:");
  Serial.println(AUDIO_SAMPLES);
  Serial.println("FORMAT:INT16_BINARY");
  Serial.println("=== DATA ===");

  // Flush text before sending binary
  Serial.flush();

  // Send raw binary audio data
  // Write in chunks to avoid overwhelming the Serial buffer
  const int CHUNK_BYTES = 512;
  uint8_t* bufPtr = (uint8_t*)audioBuffer;
  int totalBytes = AUDIO_SAMPLES * sizeof(int16_t);  // 48000
  int bytesSent = 0;

  while (bytesSent < totalBytes) {
    int chunkSize = min(CHUNK_BYTES, totalBytes - bytesSent);
    Serial.write(&bufPtr[bytesSent], chunkSize);
    bytesSent += chunkSize;
  }

  // Flush binary data before sending footer
  Serial.flush();

  // Text footer (on new line — the binary data doesn't end with newline)
  Serial.println();
  Serial.println("=== END_AUDIO ===");
  Serial.flush();
}

// ─── Command Handlers ────────────────────────────────────────────────────

bool isValidLabel(const char* label) {
  if (label[0] == '\0') return false;
  for (int i = 0; label[i] != '\0'; i++) {
    char c = label[i];
    if (!isalnum(c) && c != '_' && c != '-') return false;
  }
  return true;
}

void cmdLabel(const char* args) {
  // Skip leading whitespace
  while (*args == ' ') args++;

  if (*args == '\0') {
    if (currentLabel[0] == '\0') {
      Serial.println("Current label: (none)");
    } else {
      Serial.print("Current label: ");
      Serial.println(currentLabel);
    }
    return;
  }

  if (!isValidLabel(args)) {
    Serial.println("ERROR: Invalid label. Use only letters, numbers, underscores, and hyphens.");
    return;
  }

  strncpy(currentLabel, args, MAX_LABEL_LEN - 1);
  currentLabel[MAX_LABEL_LEN - 1] = '\0';
  Serial.print("Label set: ");
  Serial.println(currentLabel);
}

void cmdRecord(const char* args) {
  if (currentLabel[0] == '\0') {
    Serial.println("ERROR: Set a label first with 'label <name>'");
    return;
  }

  // Parse optional count
  int count = 1;
  while (*args == ' ') args++;
  if (*args != '\0') {
    count = atoi(args);
    if (count < 1) count = 1;
    if (count > 100) count = 100;  // Safety limit
  }

  for (int i = 0; i < count; i++) {
    if (count > 1) {
      Serial.print("Recording ");
      Serial.print(i + 1);
      Serial.print("/");
      Serial.print(count);
      Serial.print(" [");
      Serial.print(currentLabel);
      Serial.println("]...");
    } else {
      Serial.print("Recording [");
      Serial.print(currentLabel);
      Serial.println("]...");
    }

    if (!captureClip()) {
      Serial.println("ERROR: Capture failed. Aborting.");
      return;
    }

    sendClip();

    // Brief pause between consecutive recordings
    if (i < count - 1) {
      delay(200);
    }
  }

  if (count > 1) {
    Serial.print("Done. Recorded ");
    Serial.print(count);
    Serial.print(" clips labeled \"");
    Serial.print(currentLabel);
    Serial.println("\".");
  }
}

void cmdGain(const char* args) {
  while (*args == ' ') args++;

  if (*args == '\0') {
    Serial.print("Current gain: ");
    Serial.println(currentGain);
    return;
  }

  int gain = atoi(args);
  if (gain != 10 && gain != 20 && gain != 40 && gain != 60) {
    Serial.println("ERROR: Gain must be 10, 20, 40, or 60.");
    return;
  }

  currentGain = gain;
  Serial.print("Gain set: ");
  Serial.println(currentGain);
}

void cmdStatus() {
  Serial.println("─── Status ───");
  Serial.print("  Label: ");
  if (currentLabel[0] == '\0') {
    Serial.println("(none)");
  } else {
    Serial.println(currentLabel);
  }
  Serial.print("  Gain:  ");
  Serial.println(currentGain);
  Serial.print("  Sample rate: ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.print("  Clip duration: ");
  Serial.print(AUDIO_DURATION_MS);
  Serial.println(" ms");
  Serial.print("  Samples per clip: ");
  Serial.println(AUDIO_SAMPLES);
}

void printHelp() {
  Serial.println();
  Serial.println("=== Background Noise Collector ===");
  Serial.println("Commands:");
  Serial.println("  label <name>   Set the label (e.g., \"label road_60mph\")");
  Serial.println("  record         Capture one 1.5s clip and send to PC");
  Serial.println("  record <N>     Capture N consecutive clips (e.g., \"record 10\")");
  Serial.println("  gain <value>   Set mic gain (10, 20, 40, 60)");
  Serial.println("  status         Show current settings");
  Serial.println("  help           Show this menu");
  Serial.println();
}

// ─── Command Dispatch ────────────────────────────────────────────────────

void processCommand(const char* line) {
  // Skip leading whitespace
  while (*line == ' ') line++;
  if (*line == '\0') return;

  if (strncmp(line, "label", 5) == 0) {
    cmdLabel(line + 5);
  } else if (strncmp(line, "record", 6) == 0) {
    cmdRecord(line + 6);
  } else if (strncmp(line, "gain", 4) == 0) {
    cmdGain(line + 4);
  } else if (strcmp(line, "status") == 0) {
    cmdStatus();
  } else if (strcmp(line, "help") == 0 || strcmp(line, "h") == 0) {
    printHelp();
  } else {
    Serial.print("Unknown command: ");
    Serial.println(line);
    Serial.println("Type 'help' for available commands.");
  }
}

// ─── Setup & Loop ────────────────────────────────────────────────────────

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
    delay(10);  // Wait for Serial Monitor connection
  }

  printHelp();
  cmdStatus();
  Serial.println();
  Serial.print("> ");
}

void loop() {
  // Read Serial input character by character until newline
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (inputPos > 0) {
        inputBuffer[inputPos] = '\0';
        processCommand(inputBuffer);
        inputPos = 0;
        Serial.print("> ");
      }
    } else {
      if (inputPos < INPUT_BUF_LEN - 1) {
        inputBuffer[inputPos++] = c;
      }
    }
  }
}
