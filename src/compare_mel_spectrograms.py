r"""Side-by-side mel-spectrogram comparison: Python (librosa) vs Arduino (on-device).

For each clip, this script:
  1. Sends a "DUMP" command to the Arduino (which has been flashed with the
     updated car_sound_classifier.ino sketch).
  2. Plays the clip through the system audio output (Bluetooth speaker, etc.).
  3. Reads the 40x92 log-mel spectrogram the Arduino computed on-device,
     streamed back over Serial as text (MELDUMP_START ... MELDUMP_END).
  4. Computes the librosa-equivalent log-mel spectrogram from the original WAV.
  5. Saves three PNGs:
       <clip>_python.png        — Python-only spectrogram
       <clip>_arduino.png       — Arduino-only spectrogram
       <clip>_side_by_side.png  — Both side-by-side, Python on the left

Both spectrograms use librosa's power_to_db(ref=np.max, top_db=80), so the
two views are directly comparable on a shared dB color scale.

Usage:
    python src/compare_mel_spectrograms.py --port COM3
    python src/compare_mel_spectrograms.py --port /dev/ttyACM0 --loops 2 \
        --output-dir results/mel_comparison

Prerequisites:
    - Arduino flashed with the latest car_sound_classifier.ino (DUMP command)
    - Speaker paired and selected as system default audio output
    - pip install sounddevice soundfile pyserial matplotlib librosa numpy
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial
import sounddevice as sd

from presentation_demo import (
    CLASS_NAMES,
    DATASET_DIR,
    DEFAULT_CLIPS,
    NUM_CLASSES,
    PROJECT_ROOT,
    build_playback_buffer,
    compute_mel_for_display,
    connect_serial,
    parse_result_line,
    start_playback_async,
    wait_for_listening,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "mel_comparison"

# Must match arduino/car_sound_classifier/config.h
N_MELS_EXPECTED = 40
N_FRAMES_EXPECTED = 92

# librosa.power_to_db(ref=np.max, top_db=80) always lives in [-80, 0] dB.
# Hard-coding shared limits keeps the Python and Arduino color scales
# directly comparable.
DB_VMIN = -80.0
DB_VMAX = 0.0

# The first ~25 frames (~400 ms) of every Arduino capture are corrupted by
# a stack of pipeline artifacts — a PDM startup transient followed by
# Bluetooth-speaker output latency — both deterministic and outside the
# scope of this comparison tool. Trimming them off both panels gives a
# meaningful side-by-side. The full untrimmed arrays are still preserved
# in the .npz files when --save-npz is used.
DEFAULT_TRIM_LEADING_FRAMES = 30

MELDUMP_START = "MELDUMP_START|"
MELDUMP_END = "MELDUMP_END"
RESULT_MARKER = "RESULT|"


# ─── Helpers ──────────────────────────────────────────────────────────────


def safe_clip_name(rel_path: str) -> str:
    """Turn 'idle state/low_oil/low_oil_9.wav' into 'idle_state_low_oil_low_oil_9'."""
    p = Path(rel_path)
    stem = "_".join(list(p.parent.parts) + [p.stem])
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


def read_arduino_mel(ser: serial.Serial, timeout: float = 25.0) -> np.ndarray | None:
    """Block until a complete MELDUMP_START..MELDUMP_END payload is received.

    Returns the spectrogram as a (n_mels, n_frames) float32 ndarray, or None
    on timeout / parse failure. Lines that are not part of the dump are
    silently skipped (the sketch also prints "Listening...", inference logs,
    etc.).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not ser.in_waiting:
            time.sleep(0.01)
            continue
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line.startswith(MELDUMP_START):
            continue

        header = line.split("|")
        try:
            n_mels = int(header[1])
            n_frames = int(header[2])
        except (IndexError, ValueError):
            print(f"  ERROR: malformed MELDUMP header: {line!r}")
            return None

        if (n_mels, n_frames) != (N_MELS_EXPECTED, N_FRAMES_EXPECTED):
            print(f"  WARNING: unexpected dims {n_mels}x{n_frames} "
                  f"(expected {N_MELS_EXPECTED}x{N_FRAMES_EXPECTED})")

        mel = np.zeros((n_mels, n_frames), dtype=np.float32)
        for b in range(n_mels):
            row_line = ser.readline().decode("utf-8", errors="replace").strip()
            values = row_line.split(",")
            if len(values) != n_frames:
                print(f"  ERROR: row {b} has {len(values)} values, "
                      f"expected {n_frames}; raw={row_line[:80]!r}")
                return None
            try:
                mel[b, :] = np.fromiter((float(v) for v in values), dtype=np.float32,
                                        count=n_frames)
            except ValueError as exc:
                print(f"  ERROR: parse failure on row {b}: {exc}")
                return None

        # Drain a few trailing lines looking for MELDUMP_END (non-fatal).
        for _ in range(5):
            tail = ser.readline().decode("utf-8", errors="replace").strip()
            if tail == MELDUMP_END or tail.startswith("MELDUMP"):
                break
        return mel

    print("  ERROR: timed out waiting for MELDUMP_START")
    return None


def read_result_line(ser: serial.Serial, timeout: float = 8.0) -> dict | None:
    """Read serial until the RESULT|... line that follows a DUMP cycle."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not ser.in_waiting:
            time.sleep(0.01)
            continue
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line.startswith(RESULT_MARKER):
            return parse_result_line(line)
    return None


# ─── Plotting ─────────────────────────────────────────────────────────────


def _draw_mel(ax, mel_db: np.ndarray, title: str):
    im = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma",
                   vmin=DB_VMIN, vmax=DB_VMAX)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Mel band")
    return im


def save_individual(mel_db: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    im = _draw_mel(ax, mel_db, title)
    fig.colorbar(im, ax=ax, label="dB (relative to max)", shrink=0.85)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_side_by_side(mel_py: np.ndarray, mel_ar: np.ndarray,
                       output_path: Path, *, clip_label: str,
                       prediction_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), constrained_layout=True)
    _draw_mel(axes[0], mel_py, "Python (librosa, from WAV)")
    im = _draw_mel(axes[1], mel_ar, "Arduino (on-device, via PDM mic)")
    fig.colorbar(im, ax=axes, label="dB (relative to max)", shrink=0.9, pad=0.02)
    suptitle = clip_label
    if prediction_label:
        suptitle = f"{clip_label}\n{prediction_label}"
    fig.suptitle(suptitle, fontsize=12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─── Live preview window ──────────────────────────────────────────────────


class ComparisonViewer:
    """Minimal live matplotlib window that mirrors presentation_demo.py's UX."""

    def __init__(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(
            1, 2, figsize=(13, 5), constrained_layout=True,
        )
        self.fig.canvas.manager.set_window_title(
            "Mel-spectrogram Comparison — Python vs Arduino",
        )
        self.pending_key: str | None = None
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._show_idle("Press [SPACE] to capture the first clip   |   "
                        "[s] skip   |   [q] quit")
        plt.show(block=False)
        plt.pause(0.1)

    def _on_key(self, event):
        if event.key in (" ", "s", "q", "r"):
            self.pending_key = event.key

    def _show_idle(self, message: str):
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        self.fig.suptitle(message, fontsize=13)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def show_status(self, message: str):
        self.fig.suptitle(message, fontsize=13)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def show_comparison(self, mel_py: np.ndarray, mel_ar: np.ndarray,
                         clip_label: str, prediction_label: str):
        for ax in self.axes:
            ax.clear()
        _draw_mel(self.axes[0], mel_py, "Python (librosa, from WAV)")
        _draw_mel(self.axes[1], mel_ar, "Arduino (on-device, via PDM mic)")
        suptitle = f"{clip_label}\n{prediction_label}" if prediction_label else clip_label
        self.fig.suptitle(suptitle, fontsize=12)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def wait_for_key(self, prompt: str = "") -> str:
        if prompt:
            self.show_status(prompt)
        self.pending_key = None
        while self.pending_key is None:
            plt.pause(0.1)
        key = self.pending_key
        self.pending_key = None
        return key


# ─── Per-clip capture ─────────────────────────────────────────────────────


def capture_one(ser: serial.Serial, clip_rel_path: str, true_label: int,
                note: str, volume: float, num_loops: int,
                output_dir: Path, save_npz: bool,
                viewer: ComparisonViewer | None,
                trim_leading_frames: int = DEFAULT_TRIM_LEADING_FRAMES) -> bool:
    full_path = DATASET_DIR / clip_rel_path
    if not full_path.exists():
        print(f"  ERROR: {full_path} not found")
        return False

    safe = safe_clip_name(clip_rel_path)
    filename = Path(clip_rel_path).name

    if viewer is not None:
        viewer.show_status(f"Capturing {filename} — Arduino is listening...")

    # Drain pending serial, request a dump.
    while ser.in_waiting:
        ser.readline()
    ser.write(b"DUMP\n")
    if not wait_for_listening(ser, timeout=5.0):
        print("  WARNING: Arduino did not confirm 'Listening' — proceeding anyway")

    # Start playback the moment the Arduino starts capturing.
    try:
        play_data, play_sr = build_playback_buffer(full_path, volume, num_loops)
    except Exception as exc:
        print(f"  ERROR loading audio: {exc}")
        return False
    start_playback_async(play_data, play_sr)

    mel_arduino = read_arduino_mel(ser, timeout=20.0)
    if mel_arduino is None:
        print("  ERROR: failed to read Arduino mel spectrogram")
        sd.stop()
        return False

    raw = read_result_line(ser, timeout=8.0)

    mel_python = compute_mel_for_display(full_path)

    true_name = CLASS_NAMES[true_label]
    if raw is not None:
        pred_idx = raw["predicted_class"]
        pred_name = (CLASS_NAMES[pred_idx]
                     if 0 <= pred_idx < NUM_CLASSES else f"Unknown({pred_idx})")
        verdict = "CORRECT" if pred_idx == true_label else "INCORRECT"
        prediction_label = (
            f"True: {true_name}   |   Predicted: {pred_name} "
            f"(conf {raw['confidence']:.3f})   |   {verdict}"
        )
    else:
        prediction_label = f"True: {true_name}   |   (no prediction received)"

    clip_label = filename + (f"  —  {note}" if note else "")

    # Trim leading frames in the *displayed* arrays only — the .npz keeps
    # the full untrimmed spectrograms.
    n_frames_total = mel_arduino.shape[1]
    trim = max(0, min(trim_leading_frames, n_frames_total - 1))
    mel_python_view = mel_python[:, trim:]
    mel_arduino_view = mel_arduino[:, trim:]

    output_dir.mkdir(parents=True, exist_ok=True)
    py_path = output_dir / f"{safe}_python.png"
    ar_path = output_dir / f"{safe}_arduino.png"
    sbs_path = output_dir / f"{safe}_side_by_side.png"

    save_individual(mel_python_view, f"Python (librosa) — {clip_label}", py_path)
    save_individual(mel_arduino_view, f"Arduino (on-device) — {clip_label}", ar_path)
    save_side_by_side(mel_python_view, mel_arduino_view, sbs_path,
                       clip_label=clip_label, prediction_label=prediction_label)

    if save_npz:
        np.savez(output_dir / f"{safe}_spectrograms.npz",
                 python=mel_python, arduino=mel_arduino,
                 trim_leading_frames=trim)

    print(f"  saved: {py_path.name}, {ar_path.name}, {sbs_path.name}")

    if viewer is not None:
        viewer.show_comparison(mel_python_view, mel_arduino_view,
                                clip_label=clip_label,
                                prediction_label=prediction_label)
    return True


# ─── Clip list loading ────────────────────────────────────────────────────


DEFAULT_CLIPS_JSON = Path(__file__).resolve().parent / "presentation_demo_clips.json"


def load_clip_list(clips_file: Path | None) -> list:
    if clips_file is None and DEFAULT_CLIPS_JSON.exists():
        clips_file = DEFAULT_CLIPS_JSON
    if clips_file is None:
        return DEFAULT_CLIPS
    with open(clips_file) as f:
        data = json.load(f)
    return [(c["path"], c["true_label"], c.get("note", "")) for c in data]


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side mel-spectrogram comparison: Python vs Arduino.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", required=True,
                        help="Serial port (e.g., COM3, /dev/ttyACM0)")
    parser.add_argument("--volume", type=float, default=0.8,
                        help="Playback volume (0.0-1.0)")
    parser.add_argument("--loops", type=int, default=1,
                        help="Number of times to play each clip. Only the first "
                             "1.5s is captured by the Arduino; extra loops are for "
                             "the audience.")
    parser.add_argument("--clips", type=Path, default=None,
                        help="Optional JSON file with [{path, true_label, note}, ...]. "
                             "Defaults to presentation_demo_clips.json next to this "
                             "script if it exists, otherwise to DEFAULT_CLIPS.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory where the PNGs (and optional .npz files) "
                             "will be written.")
    parser.add_argument("--save-npz", action="store_true",
                        help="Also save the raw (untrimmed) spectrograms as a .npz "
                             "alongside the PNGs (useful for downstream analysis).")
    parser.add_argument("--trim-leading-frames", type=int,
                        default=DEFAULT_TRIM_LEADING_FRAMES,
                        help="Number of leading frames to drop from the displayed "
                             "spectrograms (PDM startup + Bluetooth speaker latency "
                             "artifact). Set to 0 to display the full 92 frames.")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Capture every clip in sequence without waiting for "
                             "keypresses. The live preview window is suppressed.")
    args = parser.parse_args()

    print("=" * 70)
    print("  Mel-spectrogram Comparison — Python vs Arduino")
    print("=" * 70)

    clips = load_clip_list(args.clips)
    print(f"  Clips: {len(clips)}")
    print(f"  Output directory: {args.output_dir}")

    missing = [p for p, _, _ in clips if not (DATASET_DIR / p).exists()]
    if missing:
        print("\n  ERROR: missing clip files:")
        for p in missing:
            print(f"    {p}")
        sys.exit(1)

    try:
        ser = connect_serial(args.port)
    except serial.SerialException as exc:
        print(f"  ERROR: {exc}")
        print(f"  Is the Arduino connected? Is the port '{args.port}' correct?")
        sys.exit(1)

    viewer = None if args.non_interactive else ComparisonViewer()

    quit_requested = False
    try:
        for idx, (clip_path, true_label, note) in enumerate(clips):
            filename = Path(clip_path).name
            print(f"\n[{idx + 1}/{len(clips)}] {filename}")
            if note:
                print(f"  note: {note}")

            if viewer is not None:
                key = viewer.wait_for_key(
                    f"Clip {idx + 1}/{len(clips)} ready: {filename}\n"
                    f"[SPACE] capture   [s] skip   [q] quit",
                )
                if key == "q":
                    quit_requested = True
                    break
                if key == "s":
                    print("  skipped")
                    continue

            capture_one(
                ser, clip_path, true_label, note,
                volume=args.volume, num_loops=args.loops,
                output_dir=args.output_dir, save_npz=args.save_npz,
                viewer=viewer,
                trim_leading_frames=args.trim_leading_frames,
            )

        if not quit_requested and viewer is not None:
            viewer.wait_for_key(
                "All clips captured. Press [q] (or any key) to exit.",
            )
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        sd.stop()
        plt.close("all")
        print("\nDone.")


if __name__ == "__main__":
    main()
