"""Automated playback evaluation for on-device car sound classification.

Plays test set audio through the laptop speaker while the Arduino listens,
captures, classifies, and reports results over Serial. Compares on-device
predictions against ground truth labels to produce a full evaluation report.

This bridges evaluation dimensions D1 (PC accuracy) and D4 (end-to-end
system) from the evaluation plan.

Usage:
    python src/playback_test.py --port /dev/ttyS3 --num-clips 208
    python src/playback_test.py --port COM3 --num-clips 50 --delay 3.0

Prerequisites:
    - Arduino running car_sound_classifier.ino, connected via USB
    - Speakers/audio output positioned near the Arduino's PDM microphone
    - pip install sounddevice soundfile pyserial
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import serial
import sounddevice as sd
import soundfile as sf

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import compute_metrics, get_tier_config, plot_confusion_matrix

# ─── Paths ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
DATASET_DIR = PROJECT_ROOT / "car_diagnostics_dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

# ─── Constants ────────────────────────────────────────────────────────────

SERIAL_BAUD = 115200
RESULT_MARKER = "RESULT|"
TIER = 2  # Primary deployment target


# ─── Serial Communication ────────────────────────────────────────────────


def connect_serial(port: str) -> serial.Serial:
    """Connect to the Arduino over serial."""
    ser = serial.Serial(port, SERIAL_BAUD, timeout=10.0)
    time.sleep(2.0)  # Wait for Arduino to reset after connection
    # Flush any startup messages
    while ser.in_waiting:
        ser.readline()
    return ser


def wait_for_result(ser: serial.Serial, timeout: float = 15.0) -> dict | None:
    """Wait for a RESULT| line from the Arduino.

    Returns parsed result dict or None on timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line.startswith(RESULT_MARKER):
                return parse_result_line(line)
            elif line:
                # Print non-result lines for debugging
                print(f"    Arduino: {line}")
    return None


def parse_result_line(line: str) -> dict:
    """Parse a RESULT|...|... line from the Arduino.

    Format: RESULT|inference_num|predicted_class|confidence|feat_ms|infer_ms|total_ms|p0,p1,...,p5
    """
    parts = line[len(RESULT_MARKER):].split("|")
    if len(parts) < 7:
        return {"error": f"Malformed result: {line}"}

    probs_str = parts[6].split(",")
    probs = [float(p) for p in probs_str]

    return {
        "inference_num": int(parts[0]),
        "predicted_class": int(parts[1]),
        "confidence": float(parts[2]),
        "feat_ms": float(parts[3]),
        "infer_ms": float(parts[4]),
        "total_ms": float(parts[5]),
        "probabilities": probs,
    }


# ─── Audio Playback ──────────────────────────────────────────────────────


def play_audio(file_path: str, volume: float = 0.8) -> None:
    """Play a WAV file through the default audio output."""
    data, sr = sf.read(file_path, dtype="float32")
    # Normalize and apply volume
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Convert stereo to mono
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * volume
    sd.play(data, samplerate=sr)
    sd.wait()


# ─── Main Evaluation ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Automated playback evaluation")
    parser.add_argument("--port", required=True, help="Serial port (e.g., /dev/ttyS3, COM3)")
    parser.add_argument("--num-clips", type=int, default=0,
                        help="Number of clips to test (0 = all)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds between clips")
    parser.add_argument("--volume", type=float, default=0.8,
                        help="Playback volume (0.0-1.0)")
    parser.add_argument("--output", default=str(RESULTS_DIR),
                        help="Output directory for results")
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2, 3],
                        help="Taxonomy tier (default: 2)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    tier = args.tier

    print("=" * 60)
    print("  Playback Evaluation Test")
    print("=" * 60)

    # Load test manifest
    manifest = pd.read_csv(SPLITS_DIR / "test_manifest.csv")
    if args.num_clips > 0:
        manifest = manifest.head(args.num_clips)

    tier_cfg = get_tier_config(tier)
    class_names = tier_cfg["class_names"]
    num_classes = tier_cfg["num_classes"]

    print(f"  Test clips: {len(manifest)}")
    print(f"  Tier: {tier} ({num_classes} classes)")
    print(f"  Port: {args.port}")
    print(f"  Volume: {args.volume}")
    print(f"  Delay: {args.delay}s")
    print()

    # Connect to Arduino
    print(f"Connecting to {args.port}...")
    try:
        ser = connect_serial(args.port)
        print(f"  Connected.")
    except serial.SerialException as e:
        print(f"  ERROR: {e}")
        print(f"  Make sure the Arduino is connected and the port is correct.")
        sys.exit(1)

    # Drain any continuous-mode results
    time.sleep(1.0)
    while ser.in_waiting:
        ser.readline()

    # Run evaluation
    results = []
    y_true = []
    y_pred = []
    feat_times = []
    infer_times = []
    total_times = []

    print(f"\nRunning playback test ({len(manifest)} clips)...")
    print(f"{'#':<5} {'File':<40} {'True':<20} {'Predicted':<20} {'Conf':>6} {'Match':>6}")
    print("-" * 100)

    for idx, row in manifest.iterrows():
        clip_num = idx + 1
        file_path = PROJECT_ROOT / row["file_path"]
        true_label = row[f"tier{tier}_label"]
        true_name = row[f"tier{tier}_name"]

        if not file_path.exists():
            print(f"  SKIP: {file_path} not found")
            continue

        # Flush serial buffer
        while ser.in_waiting:
            ser.readline()

        # Send READY command to trigger capture
        ser.write(b"READY\n")
        time.sleep(0.5)  # Brief pause for Arduino to start listening

        # Play audio
        play_audio(str(file_path), volume=args.volume)

        # Wait for result
        result = wait_for_result(ser, timeout=15.0)

        if result is None or "error" in result:
            print(f"{clip_num:<5} {row['filename']:<40} {true_name:<20} {'TIMEOUT':<20}")
            continue

        pred_class = result["predicted_class"]
        pred_name = class_names[pred_class] if pred_class < len(class_names) else f"Unknown({pred_class})"
        confidence = result["confidence"]
        match = "Y" if pred_class == true_label else "N"

        y_true.append(true_label)
        y_pred.append(pred_class)
        feat_times.append(result["feat_ms"])
        infer_times.append(result["infer_ms"])
        total_times.append(result["total_ms"])

        results.append({
            "clip_num": clip_num,
            "filename": row["filename"],
            "true_label": int(true_label),
            "true_name": true_name,
            "predicted_label": pred_class,
            "predicted_name": pred_name,
            "confidence": confidence,
            "feat_ms": result["feat_ms"],
            "infer_ms": result["infer_ms"],
            "total_ms": result["total_ms"],
            "probabilities": result["probabilities"],
            "correct": pred_class == true_label,
        })

        print(f"{clip_num:<5} {row['filename']:<40} {true_name:<20} {pred_name:<20} "
              f"{confidence:>6.3f} {match:>6}")

        # Delay between clips
        time.sleep(args.delay)

    ser.close()

    # ── Compute Evaluation Metrics ────────────────────────────────────

    if len(y_true) == 0:
        print("\nERROR: No results collected. Check Arduino connection.")
        sys.exit(1)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"\n{'=' * 60}")
    print(f"  Results ({len(y_true)} clips evaluated)")
    print(f"{'=' * 60}")

    metrics = compute_metrics(y_true, y_pred, tier=tier)
    print(f"\n  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")

    print(f"\n  Latency Statistics:")
    print(f"    Feature extraction: {np.mean(feat_times):.1f} +/- {np.std(feat_times):.1f} ms")
    print(f"    Inference:          {np.mean(infer_times):.1f} +/- {np.std(infer_times):.1f} ms")
    print(f"    Total cycle:        {np.mean(total_times):.1f} +/- {np.std(total_times):.1f} ms")

    print(f"\n  Classification Report:")
    print(metrics["report_str"])

    # Generate confusion matrix
    from sklearn.metrics import confusion_matrix as sk_cm
    labels = list(range(num_classes))
    cm = sk_cm(y_true, y_pred, labels=labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    np.nan_to_num(cm_norm, copy=False, nan=0.0)

    plot_confusion_matrix(
        cm, class_names,
        title=f"Playback Test — Confusion Matrix (Tier {tier})",
        save_path=str(output_dir / f"cm_playback_tier_{tier}.png"),
        normalized=False,
    )
    plot_confusion_matrix(
        cm_norm, class_names,
        title=f"Playback Test — Normalized CM (Tier {tier})",
        save_path=str(output_dir / f"cm_playback_tier_{tier}_norm.png"),
        normalized=True,
    )

    # ── Save Results ──────────────────────────────────────────────────

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "port": args.port,
            "tier": tier,
            "num_clips": len(results),
            "volume": args.volume,
            "delay": args.delay,
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
        },
        "latency": {
            "feat_ms_mean": float(np.mean(feat_times)),
            "feat_ms_std": float(np.std(feat_times)),
            "infer_ms_mean": float(np.mean(infer_times)),
            "infer_ms_std": float(np.std(infer_times)),
            "total_ms_mean": float(np.mean(total_times)),
            "total_ms_std": float(np.std(total_times)),
        },
        "clips": results,
    }

    json_path = output_dir / "playback_test_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # CSV summary
    csv_path = output_dir / "playback_test_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "true_name", "predicted_label",
                         "predicted_name", "confidence", "correct",
                         "feat_ms", "infer_ms", "total_ms"])
        for r in results:
            writer.writerow([
                r["filename"], r["true_label"], r["true_name"],
                r["predicted_label"], r["predicted_name"],
                f"{r['confidence']:.4f}", r["correct"],
                f"{r['feat_ms']:.1f}", f"{r['infer_ms']:.1f}", f"{r['total_ms']:.1f}",
            ])
    print(f"  CSV saved: {csv_path}")

    print(f"\n{'=' * 60}")
    print(f"  Playback test complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
