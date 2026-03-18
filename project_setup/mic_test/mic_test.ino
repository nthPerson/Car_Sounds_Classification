/*
 * PDM Microphone Test Sketch — Arduino Nano 33 BLE Sense Rev2
 *
 * Purpose: Interactively verify that the on-board MP34DT06JTR PDM MEMS
 *          microphone is working correctly. Provides multiple visualization
 *          modes for the Serial Monitor and Serial Plotter.
 *
 * Hardware: Arduino Nano 33 BLE Sense Rev2 (nRF52840)
 * Library:  PDM (built-in with Arduino Mbed OS Nano Boards package)
 *
 * Setup:
 *   1. Install "Arduino Mbed OS Nano Boards" via Boards Manager
 *   2. Select Board: Arduino Nano 33 BLE
 *   3. Upload this sketch
 *   4. Open Serial Monitor at 115200 baud
 *
 * Interactive modes (send a command via Serial Monitor):
 *
 *   w — WAVEFORM   Live raw audio samples for Serial Plotter (real-time
 *                   waveform). Open Tools > Serial Plotter to see the wave.
 *
 *   l — LEVEL      Streams RMS level over time for Serial Plotter. Shows a
 *                   smoothed loudness curve — good for verifying the mic
 *                   responds to sound at different distances/volumes.
 *
 *   v — VU METER   ASCII bar-graph volume meter in Serial Monitor. Visual
 *                   real-time feedback without needing Serial Plotter.
 *
 *   s — STATS      Batch capture of 1.5 s (matches dataset clip length),
 *                   then prints min/max/mean/RMS and first 100 raw samples.
 *                   This is the original diagnostic mode.
 *
 *   g — GAIN       Cycle the PDM gain through 10 / 20 / 40 / 60 so you can
 *                   find the best setting for your environment.
 *
 *   h — HELP       Re-print the command menu.
 *
 * Expected output for a working microphone:
 *   - Silence: RMS ~50–200 (background noise floor)
 *   - Talking/clapping near mic: RMS ~500–10000+
 *   - If all samples are 0 or -1: microphone is not working
 */

#include <PDM.h>

// ---- Configuration ----
#define SAMPLE_RATE       16000    // Hz — matches our training pipeline
#define AUDIO_DURATION_MS 1500     // 1.5 seconds — matches dataset clip length
#define AUDIO_SAMPLES     (SAMPLE_RATE * AUDIO_DURATION_MS / 1000)  // 24000
#define NUM_CHANNELS      1        // Mono

// Small ring buffer for streaming modes (waveform / level / VU).
// PDM delivers samples in chunks; we process them in the main loop.
#define STREAM_BUF_SIZE   512
int16_t streamBuf[STREAM_BUF_SIZE];
volatile int streamSamplesReady = 0;

// Large buffer used only in STATS mode (batch capture).
int16_t audioBuffer[AUDIO_SAMPLES];
volatile int batchSamplesRead = 0;
volatile bool captureComplete = false;

// ---- Mode enum ----
enum Mode { MODE_WAVEFORM, MODE_LEVEL, MODE_VU, MODE_STATS, MODE_IDLE };
volatile Mode currentMode = MODE_IDLE;

// ---- Gain settings ----
const int gainSteps[] = {10, 20, 40, 60};
const int numGainSteps = sizeof(gainSteps) / sizeof(gainSteps[0]);
int gainIndex = 1;  // default 20

// ---- PDM callback ----
// Called by the PDM library when new samples are available.
// Runs in interrupt context — keep it fast, no Serial prints here.
void onPDMdata() {
  int bytesAvailable = PDM.available();
  int samples = bytesAvailable / sizeof(int16_t);

  if (currentMode == MODE_STATS) {
    // Batch capture into the large buffer
    if (batchSamplesRead + samples > AUDIO_SAMPLES) {
      samples = AUDIO_SAMPLES - batchSamplesRead;
    }
    if (samples > 0) {
      PDM.read(&audioBuffer[batchSamplesRead], samples * sizeof(int16_t));
      batchSamplesRead += samples;
    }
    if (batchSamplesRead >= AUDIO_SAMPLES) {
      captureComplete = true;
    }
  } else {
    // Streaming modes: read into the small ring buffer
    if (samples > STREAM_BUF_SIZE) samples = STREAM_BUF_SIZE;
    PDM.read(streamBuf, samples * sizeof(int16_t));
    streamSamplesReady = samples;
  }
}

// ---- Helpers ----

void printMenu() {
  Serial.println();
  Serial.println("==========================================");
  Serial.println("  PDM Microphone Test — Interactive");
  Serial.println("  Arduino Nano 33 BLE Sense Rev2");
  Serial.println("==========================================");
  Serial.println();
  Serial.println("Commands (send via Serial Monitor):");
  Serial.println("  w  Waveform — live audio for Serial Plotter");
  Serial.println("  l  Level    — RMS loudness for Serial Plotter");
  Serial.println("  v  VU meter — ASCII bar in Serial Monitor");
  Serial.println("  s  Stats    — batch 1.5 s capture + statistics");
  Serial.println("  g  Gain     — cycle mic gain (10/20/40/60)");
  Serial.println("  h  Help     — show this menu");
  Serial.println();
  Serial.print("Current gain: ");
  Serial.println(gainSteps[gainIndex]);
  Serial.print("Sample rate:  ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Waiting for command...");
}

void printBar(int level, int maxLevel, int barWidth) {
  // Scale level to bar width
  int filled = (long)level * barWidth / maxLevel;
  if (filled > barWidth) filled = barWidth;
  if (filled < 0) filled = 0;

  Serial.print("[");
  for (int i = 0; i < barWidth; i++) {
    if (i < filled) Serial.print("#");
    else Serial.print(" ");
  }
  Serial.print("] ");
}

// ---- Mode: Waveform (Serial Plotter) ----
// Streams raw int16 samples, one per line. The Serial Plotter renders
// these as a real-time waveform — the most direct view of what the mic hears.

void runWaveform() {
  Serial.println("WAVEFORM mode — open Serial Plotter (Ctrl+Shift+L)");
  Serial.println("Send any character to stop.");
  delay(500);

  while (true) {
    if (Serial.available()) { Serial.read(); break; }

    if (streamSamplesReady > 0) {
      int n = streamSamplesReady;
      streamSamplesReady = 0;
      // Print every 4th sample to avoid overwhelming Serial at 115200 baud.
      // Effective plot rate: ~4000 points/s which Serial Plotter handles well.
      for (int i = 0; i < n; i += 4) {
        Serial.println(streamBuf[i]);
      }
    }
  }
}

// ---- Mode: Level meter (Serial Plotter) ----
// Computes RMS over each incoming PDM chunk and prints it as a single value
// per line. Serial Plotter shows a smooth loudness curve over time.

void runLevel() {
  Serial.println("LEVEL mode — open Serial Plotter (Ctrl+Shift+L)");
  Serial.println("Send any character to stop.");
  delay(500);

  // Print header labels for Serial Plotter legend
  Serial.println("RMS\tPeak");

  while (true) {
    if (Serial.available()) { Serial.read(); break; }

    if (streamSamplesReady > 0) {
      int n = streamSamplesReady;
      streamSamplesReady = 0;

      int64_t sumSq = 0;
      int16_t peak = 0;
      for (int i = 0; i < n; i++) {
        int16_t s = streamBuf[i];
        sumSq += (int64_t)s * s;
        int16_t absS = abs(s);
        if (absS > peak) peak = absS;
      }
      float rms = sqrt((float)sumSq / n);

      // Two tab-separated columns for Serial Plotter
      Serial.print(rms, 0);
      Serial.print("\t");
      Serial.println(peak);
    }
  }
}

// ---- Mode: VU meter (Serial Monitor) ----
// Prints an ASCII bar graph that updates in-place, giving real-time visual
// feedback without needing Serial Plotter.

void runVU() {
  Serial.println("VU METER mode — watch the bar graph below.");
  Serial.println("Send any character to stop.");
  Serial.println();
  delay(500);

  const int barWidth = 50;
  const int maxRMS = 6000;  // clips above this

  while (true) {
    if (Serial.available()) { Serial.read(); break; }

    if (streamSamplesReady > 0) {
      int n = streamSamplesReady;
      streamSamplesReady = 0;

      int64_t sumSq = 0;
      for (int i = 0; i < n; i++) {
        int16_t s = streamBuf[i];
        sumSq += (int64_t)s * s;
      }
      float rms = sqrt((float)sumSq / n);

      // Determine signal label
      const char* label;
      if (rms < 5)        label = "NO SIGNAL";
      else if (rms < 100) label = "quiet";
      else if (rms < 500) label = "low";
      else if (rms < 2000) label = "moderate";
      else if (rms < 4000) label = "loud";
      else                  label = "LOUD!";

      printBar((int)rms, maxRMS, barWidth);
      Serial.print("RMS: ");
      Serial.print(rms, 0);
      Serial.print("  ");
      Serial.println(label);
    }

    delay(50);  // ~20 updates/sec — readable in Serial Monitor
  }
}

// ---- Mode: Stats (batch capture — original behavior) ----
// Captures a full 1.5-second clip (matching our dataset clip length) and
// prints comprehensive statistics. Useful for one-shot verification.

void runStats() {
  captureComplete = false;
  batchSamplesRead = 0;
  memset(audioBuffer, 0, sizeof(audioBuffer));

  Serial.println("--- Capturing 1.5 seconds of audio... ---");
  unsigned long captureStart = millis();

  currentMode = MODE_STATS;

  while (!captureComplete) {
    if (Serial.available()) {
      Serial.read();
      Serial.println("(Capture interrupted)");
      currentMode = MODE_IDLE;
      return;
    }
    delay(10);
  }

  currentMode = MODE_IDLE;

  unsigned long captureEnd = millis();
  Serial.print("Capture complete in ");
  Serial.print(captureEnd - captureStart);
  Serial.println(" ms");

  // Compute statistics
  int16_t minVal = 32767;
  int16_t maxVal = -32768;
  int64_t sum = 0;
  int64_t sumSquared = 0;

  for (int i = 0; i < AUDIO_SAMPLES; i++) {
    int16_t sample = audioBuffer[i];
    if (sample < minVal) minVal = sample;
    if (sample > maxVal) maxVal = sample;
    sum += sample;
    sumSquared += (int64_t)sample * sample;
  }

  float mean = (float)sum / AUDIO_SAMPLES;
  float rms = sqrt((float)sumSquared / AUDIO_SAMPLES);
  float peakToPeak = maxVal - minVal;

  Serial.println();
  Serial.println("Audio Statistics:");
  Serial.print("  Min sample:    ");
  Serial.println(minVal);
  Serial.print("  Max sample:    ");
  Serial.println(maxVal);
  Serial.print("  Peak-to-peak:  ");
  Serial.println((int)peakToPeak);
  Serial.print("  Mean:          ");
  Serial.println(mean, 1);
  Serial.print("  RMS amplitude: ");
  Serial.println(rms, 1);

  // Assess signal quality
  Serial.println();
  if (rms < 5) {
    Serial.println("WARNING: RMS is extremely low. Microphone may not be working.");
    Serial.println("         Expected RMS > 50 even in a quiet room.");
  } else if (rms < 100) {
    Serial.println("Signal: QUIET — microphone is working (quiet environment or low gain)");
  } else if (rms < 2000) {
    Serial.println("Signal: MODERATE — microphone is capturing audio normally");
  } else {
    Serial.println("Signal: LOUD — strong audio detected");
  }

  // Print raw samples for inspection
  const int printCount = 100;
  Serial.println();
  Serial.print("First ");
  Serial.print(printCount);
  Serial.println(" raw samples:");
  for (int i = 0; i < printCount && i < AUDIO_SAMPLES; i++) {
    Serial.print(audioBuffer[i]);
    if (i < printCount - 1) Serial.print(", ");
    if ((i + 1) % 20 == 0) Serial.println();
  }
  Serial.println();
}

// ---- Gain cycling ----

void cycleGain() {
  gainIndex = (gainIndex + 1) % numGainSteps;
  PDM.setGain(gainSteps[gainIndex]);
  Serial.print("Gain set to: ");
  Serial.println(gainSteps[gainIndex]);
}

// ---- Arduino setup ----

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for Serial Monitor to connect (USB CDC)
  }

  // Register the PDM data callback
  PDM.onReceive(onPDMdata);

  // Initialize PDM microphone
  if (!PDM.begin(NUM_CHANNELS, SAMPLE_RATE)) {
    Serial.println("ERROR: Failed to initialize PDM microphone!");
    Serial.println("Check that you are using an Arduino Nano 33 BLE Sense (Rev2).");
    while (1) {
      ; // Halt
    }
  }

  PDM.setGain(gainSteps[gainIndex]);

  printMenu();
}

// ---- Arduino main loop ----

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    // Consume any trailing newline/carriage return
    delay(10);
    while (Serial.available()) Serial.read();

    switch (cmd) {
      case 'w': case 'W':
        currentMode = MODE_WAVEFORM;
        runWaveform();
        currentMode = MODE_IDLE;
        printMenu();
        break;

      case 'l': case 'L':
        currentMode = MODE_LEVEL;
        runLevel();
        currentMode = MODE_IDLE;
        printMenu();
        break;

      case 'v': case 'V':
        currentMode = MODE_VU;
        runVU();
        currentMode = MODE_IDLE;
        printMenu();
        break;

      case 's': case 'S':
        runStats();
        Serial.println();
        Serial.println("Send another command (h for help).");
        break;

      case 'g': case 'G':
        cycleGain();
        break;

      case 'h': case 'H':
        printMenu();
        break;

      default:
        // Ignore stray characters (newlines, etc.)
        break;
    }
  }
}
