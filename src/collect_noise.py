"""PC-side receiver for Arduino background noise data collection.

Connects to the Arduino running noise_collector.ino, receives labeled
audio clips over Serial, and saves them as WAV files organized by label.

Usage:
    python src/collect_noise.py --port /dev/ttyACM0
    python src/collect_noise.py --port COM3 --output data/noise_samples/
    From Windows:
    python src\collect_noise.py --port COM3 --output \\wsl$\Ubuntu\home\robert\BDA602\car-sounds\data\noise_samples\
    

The script provides an interactive terminal:
  - Type commands (label, record, gain, etc.) which are forwarded to the Arduino
  - Audio clips are automatically received and saved as WAV files
  - A collection log CSV tracks all recordings

Press Ctrl+C to exit.
"""

import argparse
import csv
import os
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import soundfile as sf


# ─── Constants ────────────────────────────────────────────────────────────

BAUD_RATE = 115200
BEGIN_MARKER = "=== BEGIN_AUDIO ==="
DATA_MARKER = "=== DATA ==="
END_MARKER = "=== END_AUDIO ==="
BYTES_PER_SAMPLE = 2  # int16


# ─── Receiver ─────────────────────────────────────────────────────────────


class NoiseCollector:
    """Manages serial communication with the Arduino and saves audio clips."""

    def __init__(self, port: str, output_dir: str):
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ser = None
        self.running = False
        self.clips_saved = 0

        # Collection log
        self.log_path = self.output_dir / "collection_log.csv"
        self._init_log()

    def _init_log(self):
        """Create collection log CSV if it doesn't exist."""
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "label", "filename", "sample_rate",
                    "num_samples", "duration_s", "rms",
                ])

    def _log_clip(self, label: str, filename: str, sample_rate: int,
                  num_samples: int, rms: float):
        """Append a row to the collection log."""
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                label,
                filename,
                sample_rate,
                num_samples,
                f"{num_samples / sample_rate:.3f}",
                f"{rms:.1f}",
            ])

    def connect(self):
        """Open the serial connection."""
        print(f"Connecting to {self.port} at {BAUD_RATE} baud...")
        self.ser = serial.Serial(self.port, BAUD_RATE, timeout=0.1)
        time.sleep(2)  # Wait for Arduino reset after serial connection
        # Flush any startup data
        self.ser.reset_input_buffer()
        print(f"Connected. Output directory: {self.output_dir}")
        print("Type commands below (label, record, gain, status, help).")
        print("Audio clips will be saved automatically. Ctrl+C to exit.\n")

    def _read_text_line(self, timeout: float = 30.0) -> str | None:
        """Read a text line from Serial, handling mixed binary/text."""
        deadline = time.time() + timeout
        line_bytes = b""
        while time.time() < deadline:
            if self.ser.in_waiting > 0:
                byte = self.ser.read(1)
                if byte == b"\n":
                    return line_bytes.decode("utf-8", errors="replace").strip()
                if byte != b"\r":
                    line_bytes += byte
            else:
                time.sleep(0.01)
        return None

    def _receive_clip(self):
        """Receive a single audio clip after BEGIN_AUDIO marker detected."""
        metadata = {}

        # Read metadata lines until DATA marker
        while True:
            line = self._read_text_line(timeout=5.0)
            if line is None:
                print("  WARNING: Timeout reading metadata")
                return
            if line == DATA_MARKER:
                break
            if ":" in line:
                key, _, value = line.partition(":")
                metadata[key.strip()] = value.strip()

        label = metadata.get("LABEL", "unknown")
        sample_rate = int(metadata.get("SAMPLE_RATE", 16000))
        num_samples = int(metadata.get("NUM_SAMPLES", 24000))
        expected_bytes = num_samples * BYTES_PER_SAMPLE

        # Read binary audio data
        audio_bytes = b""
        deadline = time.time() + 30.0  # 30s timeout for binary data
        while len(audio_bytes) < expected_bytes and time.time() < deadline:
            remaining = expected_bytes - len(audio_bytes)
            chunk = self.ser.read(min(remaining, 4096))
            if chunk:
                audio_bytes += chunk

        if len(audio_bytes) < expected_bytes:
            print(f"  WARNING: Received {len(audio_bytes)}/{expected_bytes} bytes "
                  f"(incomplete clip)")
            return

        # Read past any trailing newline + END marker
        # The Arduino sends \n + "=== END_AUDIO ===" after binary data
        end_line = self._read_text_line(timeout=5.0)
        if end_line and end_line != END_MARKER:
            # Might have read the newline; try one more
            end_line2 = self._read_text_line(timeout=2.0)

        # Convert binary to numpy array
        samples = np.frombuffer(audio_bytes, dtype=np.int16)

        # Compute RMS for logging
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

        # Save as WAV
        label_dir = self.output_dir / label
        label_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}_{self.clips_saved:04d}.wav"
        filepath = label_dir / filename

        sf.write(str(filepath), samples, sample_rate, subtype="PCM_16")

        # Log
        self._log_clip(label, str(filepath.relative_to(self.output_dir)),
                        sample_rate, num_samples, rms)
        self.clips_saved += 1

        print(f"  Saved: {filepath.relative_to(self.output_dir)} "
              f"(RMS: {rms:.0f}, {num_samples / sample_rate:.1f}s)")

    def _reader_thread(self):
        """Background thread that reads Serial output and handles audio."""
        while self.running:
            try:
                if self.ser.in_waiting == 0:
                    time.sleep(0.01)
                    continue

                # Read a line
                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()

                if line == BEGIN_MARKER:
                    self._receive_clip()
                elif line:
                    # Regular text output from Arduino — display it
                    print(line)

            except serial.SerialException:
                if self.running:
                    print("\nSerial connection lost.")
                break
            except Exception as e:
                if self.running:
                    print(f"\nReader error: {e}")

    def run(self):
        """Main loop: read user input, forward to Arduino, receive clips."""
        self.running = True

        # Start background reader thread
        reader = threading.Thread(target=self._reader_thread, daemon=True)
        reader.start()

        # Main thread handles user input
        try:
            while self.running:
                try:
                    user_input = input()
                except EOFError:
                    break

                if not self.running:
                    break

                # Forward command to Arduino
                self.ser.write((user_input + "\n").encode("utf-8"))

        except KeyboardInterrupt:
            print(f"\n\nExiting. Saved {self.clips_saved} clips to {self.output_dir}")

        finally:
            self.running = False
            if self.ser and self.ser.is_open:
                self.ser.close()


# ─── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Receive background noise audio from Arduino and save as WAV files.",
    )
    parser.add_argument(
        "--port", required=True,
        help="Serial port (e.g., /dev/ttyACM0 on Linux, COM3 on Windows)",
    )
    parser.add_argument(
        "--output", default="data/noise_samples/",
        help="Output directory for WAV files (default: data/noise_samples/)",
    )
    args = parser.parse_args()

    collector = NoiseCollector(port=args.port, output_dir=args.output)

    try:
        collector.connect()
        collector.run()
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        print(f"Make sure the Arduino is connected and the port '{args.port}' is correct.")
        sys.exit(1)


if __name__ == "__main__":
    main()
