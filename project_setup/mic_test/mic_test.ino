/*
 * PDM Microphone Test Sketch — Arduino Nano 33 BLE Sense Rev2
 *
 * Purpose: Verify that the on-board MP34DT06JTR PDM MEMS microphone is
 *          working correctly by capturing audio at 16 kHz and reporting
 *          basic statistics over Serial.
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
 * What it does:
 *   - Captures 1.5 seconds of audio (24,000 samples at 16 kHz)
 *   - Computes and prints: min, max, mean, RMS amplitude
 *   - Prints the first 100 raw samples for visual inspection
 *   - Repeats every cycle (with a pause between captures)
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
#define PRINT_RAW_SAMPLES 100      // How many raw samples to print per capture

// ---- Audio buffer ----
// 24,000 samples × 2 bytes = 48,000 bytes (48 KB)
// This is the same buffer size the final classifier will use.
int16_t audioBuffer[AUDIO_SAMPLES];
volatile int samplesRead = 0;
volatile bool captureComplete = false;

// ---- PDM callback ----
// Called by the PDM library when new samples are available.
// Runs in interrupt context — keep it fast, no Serial prints here.
void onPDMdata() {
  int bytesAvailable = PDM.available();
  int samplesToRead = bytesAvailable / sizeof(int16_t);

  // Don't overflow the buffer
  if (samplesRead + samplesToRead > AUDIO_SAMPLES) {
    samplesToRead = AUDIO_SAMPLES - samplesRead;
  }

  if (samplesToRead > 0) {
    PDM.read(&audioBuffer[samplesRead], samplesToRead * sizeof(int16_t));
    samplesRead += samplesToRead;
  }

  if (samplesRead >= AUDIO_SAMPLES) {
    captureComplete = true;
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for Serial Monitor to connect (USB CDC)
  }

  Serial.println("==========================================");
  Serial.println("  PDM Microphone Test");
  Serial.println("  Arduino Nano 33 BLE Sense Rev2");
  Serial.println("==========================================");
  Serial.print("Sample rate:     ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.print("Capture duration: ");
  Serial.print(AUDIO_DURATION_MS);
  Serial.println(" ms");
  Serial.print("Buffer size:     ");
  Serial.print(AUDIO_SAMPLES);
  Serial.print(" samples (");
  Serial.print(AUDIO_SAMPLES * 2);
  Serial.println(" bytes)");
  Serial.println();

  // Register the PDM data callback
  PDM.onReceive(onPDMdata);

  // Initialize PDM microphone
  // Arguments: number of channels, sample rate
  if (!PDM.begin(NUM_CHANNELS, SAMPLE_RATE)) {
    Serial.println("ERROR: Failed to initialize PDM microphone!");
    Serial.println("Check that you are using an Arduino Nano 33 BLE Sense (Rev2).");
    while (1) {
      ; // Halt — nothing to do without a microphone
    }
  }

  // Optional: set the PDM gain (default is 20, range ~0-80)
  // Higher gain = more sensitive, but also more noise
  PDM.setGain(20);

  Serial.println("PDM microphone initialized successfully.");
  Serial.println("Starting audio capture...");
  Serial.println();
}

void loop() {
  // ---- 1. Start a new capture ----
  captureComplete = false;
  samplesRead = 0;
  memset(audioBuffer, 0, sizeof(audioBuffer));

  Serial.println("--- Capturing 1.5 seconds of audio... ---");
  unsigned long captureStart = millis();

  // Wait for the buffer to fill (blocking)
  while (!captureComplete) {
    // The PDM callback fills the buffer in the background.
    // At 16 kHz, 24,000 samples takes exactly 1.5 seconds.
    delay(10);
  }

  unsigned long captureEnd = millis();
  Serial.print("Capture complete in ");
  Serial.print(captureEnd - captureStart);
  Serial.println(" ms");

  // ---- 2. Compute statistics ----
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

  // ---- 3. Assess signal quality ----
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

  // ---- 4. Print raw samples for inspection ----
  Serial.println();
  Serial.print("First ");
  Serial.print(PRINT_RAW_SAMPLES);
  Serial.println(" raw samples:");
  for (int i = 0; i < PRINT_RAW_SAMPLES && i < AUDIO_SAMPLES; i++) {
    Serial.print(audioBuffer[i]);
    if (i < PRINT_RAW_SAMPLES - 1) Serial.print(", ");
    if ((i + 1) % 20 == 0) Serial.println();  // Line break every 20 samples
  }
  Serial.println();

  // ---- 5. Pause before next capture ----
  Serial.println();
  Serial.println("Waiting 3 seconds before next capture...");
  Serial.println("==========================================");
  Serial.println();
  delay(3000);
}
