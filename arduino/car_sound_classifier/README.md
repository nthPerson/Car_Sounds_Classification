# Car Sound Classifier — Arduino Deployment Guide

This guide walks you through compiling and uploading the car sound classifier to the Arduino Nano 33 BLE Sense Rev2.

## How Arduino Sketches Work (Multi-File)

The Arduino IDE automatically compiles **all files in the sketch folder** together. You don't need to manually add or link files — if a `.ino`, `.cpp`, `.h`, or `.c` file is in the same folder as the main `.ino` file, the IDE includes it in the build automatically.

This folder contains:

| File | Type | Description |
|------|------|-------------|
| `car_sound_classifier.ino` | Main sketch | Entry point (setup + loop) |
| `config.h` | Header | Compile-time constants (audio, model, classes) |
| `feature_extraction.h` | Header | Feature extraction API declarations |
| `feature_extraction.cpp` | Source | CMSIS-DSP mel-spectrogram implementation |
| `model_data.h` | Generated header | TFLite model as C byte array (21.3 KB) |
| `mel_filterbank.h` | Generated header | Sparse mel filter weights (490 values) |
| `normalization.h` | Generated header | Per-band z-score mean/std arrays |
| `hann_window.h` | Generated header | 512-point Hann window coefficients |

The 4 "generated" headers are created by running `python src/export_for_arduino.py` from the project root. They are already included in this folder and ready to use.

## Prerequisites

### 1. Install Arduino IDE

Download and install [Arduino IDE 2.x](https://www.arduino.cc/en/software) if you don't have it.

### 2. Install Board Support

1. Open Arduino IDE
2. Go to **Tools > Board > Boards Manager** (or click the board icon in the left sidebar)
3. Search for **"Arduino Mbed OS Nano Boards"**
4. Click **Install** (this may take a few minutes — it downloads the Mbed OS toolchain including CMSIS-DSP)

### 3. Install TFLite Micro Library

1. Go to **Tools > Manage Libraries** (or click the library icon in the left sidebar)
2. Search for **"Arduino_TensorFlowLite"**
3. Install the latest version
4. If the library isn't found in the Library Manager, you can install it manually:
   - Download from [Harvard_TinyMLx TFLite library](https://github.com/Harvard-CS249R-Fa22/tinyml-workshop/raw/main/Arduino_TensorFlowLite-2.4.0-ALPHA.zip) (or search for the latest Arduino-compatible TFLite Micro release)
   - In Arduino IDE: **Sketch > Include Library > Add .ZIP Library** and select the downloaded file

## Compiling and Uploading

### 1. Open the Sketch

In Arduino IDE: **File > Open** and navigate to this file:

```
Car_Sounds/arduino/car_sound_classifier/car_sound_classifier.ino
```

The IDE will open the main `.ino` file and show tabs for all the other files in the folder (`.cpp`, `.h`). You can click the tabs to view them, but you don't need to edit anything.

### 2. Select the Board

1. Connect the Arduino Nano 33 BLE Sense Rev2 via USB
2. Go to **Tools > Board** and select **Arduino Nano 33 BLE**
3. Go to **Tools > Port** and select the port where your Arduino appears (e.g., COM3 on Windows, /dev/ttyACM0 on Linux)

If the board doesn't appear in the port list:
- Try a different USB cable (some cables are charge-only)
- Double-tap the reset button on the Arduino to enter bootloader mode (the LED will pulse)

### 3. Compile (Verify)

Click the **checkmark button** (Verify) or press **Ctrl+R** to compile without uploading. This checks for errors and reports memory usage:

```
Sketch uses XXXXXX bytes (XX%) of program storage space. Maximum is 1,048,576 bytes.
Global variables use XXXXXX bytes (XX%) of dynamic memory. Maximum is 262,144 bytes.
```

**Expected:** Flash usage ~40-50%, SRAM usage ~70-80%. If SRAM exceeds ~85%, the sketch may crash at runtime.

### 4. Upload

Click the **right-arrow button** (Upload) or press **Ctrl+U**. The IDE will compile and then flash the sketch to the Arduino. The onboard LED will blink during upload.

If upload fails:
- Double-tap the reset button to enter bootloader mode, then try uploading again
- Make sure the correct port is selected in **Tools > Port**

### 5. Open Serial Monitor

1. Go to **Tools > Serial Monitor** or press **Ctrl+Shift+M**
2. Set the baud rate to **115200** (dropdown in the bottom-right of the Serial Monitor)
3. You should see the startup banner:

```
============================================================
  Car Sound Classifier — Arduino Nano 33 BLE Sense Rev2
============================================================
Model: DS-CNN (M6) Tier 2 PTQ
Feature extraction: initialized
Input: [1, 40, 92, 1], scale=0.034728, zp=40
Output: 6 classes
Tensor arena: 51200 bytes allocated (XXXXX used)

Ready. Send 'READY' to trigger classification,
or the classifier will run continuously.
```

4. The classifier will immediately begin capturing audio and classifying in a loop. You'll see results like:

```
Listening...
--- Inference #1 ---
Predicted: Normal Idle (confidence: 0.782)
Timing: capture=1501.2ms  feat=134.5ms  infer=89.2ms  total=1725.0ms
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| **"AllocateTensors() failed"** | Tensor arena too small | Increase `TENSOR_ARENA_SIZE` in `config.h` (try 65536 or 81920) |
| **Sketch too large (>1 MB flash)** | Too many TFLite ops | This sketch uses AllOpsResolver; switch to MicroMutableOpResolver with only the needed ops (listed in `model_data.h` comments) |
| **SRAM overflow (>256 KB)** | Buffers + arena too large | Reduce `TENSOR_ARENA_SIZE`, or switch to M5 (smaller model) |
| **No serial output** | Baud rate mismatch | Set Serial Monitor to 115200 baud |
| **"PDM.begin() failed"** | Board revision mismatch | Ensure you selected "Arduino Nano 33 BLE" (not the non-Sense variant) |
| **Garbled output** | Wrong baud rate | Ensure both sketch (`SERIAL_BAUD` in config.h) and Serial Monitor are at 115200 |
| **Very slow inference (>1s)** | CMSIS-NN not active | Ensure you're using the int8 model (not float32) and the Mbed OS board package is installed |

## Switching Models

To deploy a different model (e.g., M2 or M5), regenerate the headers from the project root:

```bash
# M2 (2-D CNN) — larger tensor arena needed
python src/export_for_arduino.py --model m2 --tier 2 --method ptq

# M5 (1-D CNN) — smallest model, uses MFCCs instead of mel-spectrograms
python src/export_for_arduino.py --model m5 --tier 2 --method ptq
```

Then update `TENSOR_ARENA_SIZE` in `config.h`:
- M6 (DS-CNN): 51200 (50 KB)
- M2 (2-D CNN): 81920 (80 KB)
- M5 (1-D CNN): 25600 (25 KB)

Recompile and upload.

## Running the Automated Playback Test

After the sketch is running on the Arduino, you can run the automated evaluation from your PC:

```bash
python src/playback_test.py --port COM3 --num-clips 208 --delay 3.0
```

This plays test set audio through your speakers while the Arduino classifies each clip. Results are saved to `results/playback_test_results.json`. See the main project's `GETTING_STARTED.md` Step 11d for details.
