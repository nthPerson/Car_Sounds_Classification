r"""Live presentation demo for the deployed car sound classifier.

Plays curated audio clips through a Bluetooth speaker to the Arduino,
captures its classification over Serial, and displays a real-time
visualization dashboard with:

Top row:
  - Mel-spectrogram of what the Arduino is "hearing"
  - Live probability bar chart (green=correct, red=incorrect)
  - Running confusion matrix that accumulates across the demo

Middle: centered status banner (clip info, verdict)

Bottom row:
  - Latency timeline (capture / feature extraction / inference)
  - Per-class accuracy bars (updates as clips arrive)

The script is driven by keyboard input so the presenter controls pacing:

    [SPACE]  Play the next clip
    [r]      Replay the current clip
    [s]      Skip to the next clip without playing
    [q]      Quit

Audio clips can be looped so the audience has time to recognize them
(configurable via --loops). Only the first 1.5s is captured by the Arduino;
additional loops play AFTER the Arduino result is returned, acting as a
"preview" phase for the audience.

Usage:
    python src/presentation_demo.py --port COM3
    python src/presentation_demo.py --port /dev/ttyS3 --view-time 5.0 --loops 3
    Windows -> WSL:
        python \\wsl.localhost\Ubuntu-24.04\home\robert\BDA602\Car_Sounds\src\presentation_demo.py --port COM3 --view-time 6.0 --loops 1

Prerequisites:
    - Arduino flashed with car_sound_classifier.ino, in TRIGGERED mode
    - Bluetooth speaker paired and set as system default audio output
    - pip install sounddevice soundfile pyserial matplotlib librosa numpy
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import serial
import sounddevice as sd
import soundfile as sf
from matplotlib.patches import Rectangle
from matplotlib.widgets import TextBox

# ─── Paths and constants ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "car_diagnostics_dataset"
RESULTS_DIR = PROJECT_ROOT / "results"

SERIAL_BAUD = 115200
RESULT_MARKER = "RESULT|"

CLASS_NAMES = [
    "Normal Braking",
    "Braking Fault",
    "Normal Idle",
    "Idle Fault",
    "Normal Start-Up",
    "Start-Up Fault",
]
NUM_CLASSES = len(CLASS_NAMES)

# Short labels for confusion matrix axes
SHORT_LABELS = ["NrmBrk", "BrkFlt", "NrmIdl", "IdlFlt", "NrmSU", "SUFlt"]

# ─── Default curated demo clip list ───────────────────────────────────────
# Each tuple: (relative_path_from_dataset_dir, true_label_index, display_note)
# Start with 10 samples: one strong correct per class + four known misclassifications.

DEFAULT_CLIPS = [
    # Strong correct predictions (one per class, mostly high-confidence)
    ("braking state/normal_brakes/normal_brakes_43.wav", 0, "Normal braking — strong case"),
    ("braking state/worn_out_brakes/worn_out_brakes_52.wav", 1, "Worn brakes — very high confidence"),
    ("idle state/normal_engine_idle/normal_engine_idle_92.wav", 2, "Normal idle — clean engine sound"),
    ("idle state/low_oil/low_oil_9.wav", 3, "Low oil — idle fault"),
    ("startup state/normal_engine_startup/normal_engine_startup_6.wav", 4, "Normal start-up"),
    ("startup state/bad_ignition/bad_ignition_61.wav", 5, "Bad ignition — startup fault"),

    # Honest misclassifications that illustrate the speaker-ablation story
    ("startup state/bad_ignition/bad_ignition_54.wav", 5, "Bad ignition (confidently wrong — pivot to speaker ablation)"),
    ("idle state/normal_engine_idle/normal_engine_idle_2.wav", 2, "Normal idle (confidently wrong as Braking Fault)"),
    ("braking state/normal_brakes/normal_brakes_41.wav", 0, "Normal braking (low-frequency cues lost through speaker)"),
    ("idle state/serpentine_belt/serpentine_belt_41.wav", 3, "Serpentine belt (hard case — low confidence)"),
]


# ─── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ClipResult:
    clip_idx: int
    filename: str
    true_label: int
    true_name: str
    predicted_label: int
    predicted_name: str
    confidence: float
    probabilities: list
    feat_ms: float
    infer_ms: float
    total_ms: float
    correct: bool


@dataclass
class DemoState:
    clips: list  # list of (filepath, true_label, note)
    current_idx: int = 0
    results: list = field(default_factory=list)
    confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    )
    per_class_history: list = field(
        default_factory=lambda: [[] for _ in range(NUM_CLASSES)]
    )


# ─── Serial communication ─────────────────────────────────────────────────


def connect_serial(port: str) -> serial.Serial:
    print(f"Connecting to Arduino on {port} at {SERIAL_BAUD} baud...")
    ser = serial.Serial(port, SERIAL_BAUD, timeout=10.0)
    time.sleep(2.0)  # Wait for Arduino to reset
    while ser.in_waiting:
        ser.readline()
    print("  Connected.")
    return ser


def parse_result_line(line: str) -> dict | None:
    """Parse a RESULT|... line from the Arduino sketch."""
    if not line.startswith(RESULT_MARKER):
        return None
    parts = line[len(RESULT_MARKER):].split("|")
    if len(parts) < 7:
        return None
    try:
        probs = [float(p) for p in parts[6].split(",")]
        return {
            "inference_num": int(parts[0]),
            "predicted_class": int(parts[1]),
            "confidence": float(parts[2]),
            "feat_ms": float(parts[3]),
            "infer_ms": float(parts[4]),
            "total_ms": float(parts[5]),
            "probabilities": probs,
        }
    except (ValueError, IndexError):
        return None


def wait_for_result(ser: serial.Serial, timeout: float = 15.0) -> dict | None:
    """Block until the Arduino sends a RESULT line or we time out."""
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            if raw.startswith(RESULT_MARKER):
                return parse_result_line(raw)
    return None


def wait_for_listening(ser: serial.Serial, timeout: float = 5.0) -> bool:
    """Block until the Arduino prints 'Listening...' (capture started)."""
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
            if "Listening" in raw:
                return True
    return False


# ─── Audio playback and feature visualization ─────────────────────────────


def _load_and_normalize(path: Path, volume: float) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * volume
    return data, sr


def build_playback_buffer(path: Path, volume: float, num_loops: int) -> tuple[np.ndarray, int]:
    """Build the full audio buffer (with loop repetitions) without playing it."""
    data, sr = _load_and_normalize(path, volume)
    if num_loops > 1:
        gap = np.zeros(int(sr * 0.0), dtype=data.dtype)
        pieces = []
        for i in range(num_loops):
            pieces.append(data)
            if i < num_loops - 1:
                pieces.append(gap)
        data = np.concatenate(pieces)
    return data, sr


def play_audio(path: Path, volume: float = 0.8, num_loops: int = 1) -> None:
    """Blocking playback (kept as a convenience)."""
    data, sr = build_playback_buffer(path, volume, num_loops)
    sd.play(data, samplerate=sr)
    sd.wait()


def start_playback_async(data: np.ndarray, sr: int) -> None:
    """Start audio playback without blocking. Caller stops it via sd.stop() / sd.wait()."""
    sd.play(data, samplerate=sr)


def compute_mel_for_display(path: Path) -> np.ndarray:
    """Compute a log-mel spectrogram matching the training pipeline (for display only)."""
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    # Pad/crop to 1.5s = 24000 samples
    target = 24000
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        start = (len(y) - target) // 2
        y = y[start:start + target]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    mel = librosa.feature.melspectrogram(
        y=y, sr=16000, n_fft=512, hop_length=256, n_mels=40,
        fmin=20, fmax=8000, power=2.0, window="hann", center=False,
    )
    return librosa.power_to_db(mel, ref=np.max)  # shape (40, 92)


# ─── Dashboard ────────────────────────────────────────────────────────────


class DemoDashboard:
    """matplotlib-based live dashboard for the demo."""

    def __init__(self, view_time: float, save_snapshots: bool):
        self.view_time = view_time
        self.save_snapshots = save_snapshots
        self.pending_key: str | None = None

        plt.ion()
        self.fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        self.fig.canvas.manager.set_window_title("Car Sound Classifier — Live Demo")

        gs = self.fig.add_gridspec(
            nrows=3, ncols=2,
            height_ratios=[1.4, 0.35, 0.9],
        )

        # Top row: 3 visualizations side-by-side (nested in its own gridspec)
        top_gs = gs[0, :].subgridspec(1, 3, width_ratios=[1.2, 1.0, 1.1])
        self.ax_mel = self.fig.add_subplot(top_gs[0, 0])
        self.ax_probs = self.fig.add_subplot(top_gs[0, 1])
        self.ax_cm = self.fig.add_subplot(top_gs[0, 2])

        # Middle: centered status banner spanning full width
        self.ax_status = self.fig.add_subplot(gs[1, :])
        self.ax_status.axis("off")

        # Bottom row: latency timeline (left) and per-class accuracy (right)
        self.ax_latency = self.fig.add_subplot(gs[2, 0])
        self.ax_accuracy = self.fig.add_subplot(gs[2, 1])

        # Listen for key presses
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw_initial()
        plt.show(block=False)
        plt.pause(0.1)

    def _on_key(self, event):
        if event.key in (" ", "r", "s", "q", "f"):
            self.pending_key = event.key

    def wait_for_key(self, prompt: str = "") -> str:
        """Block until a recognized key is pressed, return the key char."""
        self.pending_key = None
        if prompt:
            self._set_status_banner(prompt)
        while self.pending_key is None:
            plt.pause(0.1)
        key = self.pending_key
        self.pending_key = None
        return key

    def _set_status_banner(self, text: str, color: str = "#333"):
        self.ax_status.clear()
        self.ax_status.axis("off")
        self.ax_status.text(
            0.5, 0.5, text, ha="center", va="center",
            fontsize=14, color=color,
            transform=self.ax_status.transAxes,
            wrap=True,
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def _draw_initial(self):
        # Mel
        self.ax_mel.set_title("Mel-Spectrogram (what the Arduino \"hears\")", fontsize=11)
        self.ax_mel.set_xlabel("Time (frames)")
        self.ax_mel.set_ylabel("Mel band")
        self.ax_mel.imshow(np.zeros((40, 92)), aspect="auto", origin="lower", cmap="magma")

        # Probability bars
        self.ax_probs.set_title("Predicted probabilities", fontsize=11)
        self.ax_probs.set_xlim(0, 1)
        self.ax_probs.set_xlabel("Probability")
        self.ax_probs.set_yticks(range(NUM_CLASSES))
        self.ax_probs.set_yticklabels(CLASS_NAMES)
        self.ax_probs.invert_yaxis()
        self.ax_probs.barh(range(NUM_CLASSES), [0] * NUM_CLASSES, color="#bbb")

        # Confusion matrix
        self.ax_cm.set_title("Running confusion matrix", fontsize=11)
        self._draw_confusion(np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int))

        # Latency timeline
        self.ax_latency.set_title("Latency timeline (ms)", fontsize=11)
        self.ax_latency.set_xlim(0, 2600)
        self.ax_latency.set_ylim(0, 1)
        self.ax_latency.set_yticks([])
        self.ax_latency.set_xlabel("Time (ms)")

        # Per-class accuracy (stacked blocks)
        self._draw_per_class_accuracy([[] for _ in range(NUM_CLASSES)])

        # Status
        self._set_status_banner(
            "Press [SPACE] to play the next clip   |   [r] replay   |   [s] skip   |   "
            "[f] finish & show summary   |   [q] quit",
        )

    def _draw_confusion(self, cm: np.ndarray):
        self.ax_cm.clear()
        self.ax_cm.set_title("Running confusion matrix", fontsize=11)
        total = cm.sum()
        if total > 0:
            cm_display = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        else:
            cm_display = np.zeros_like(cm, dtype=float)
        im = self.ax_cm.imshow(cm_display, cmap="Blues", vmin=0, vmax=1)
        self.ax_cm.set_xticks(range(NUM_CLASSES))
        self.ax_cm.set_yticks(range(NUM_CLASSES))
        self.ax_cm.set_xticklabels(SHORT_LABELS, rotation=45, ha="right", fontsize=8)
        self.ax_cm.set_yticklabels(SHORT_LABELS, fontsize=8)
        self.ax_cm.set_xlabel("Predicted")
        self.ax_cm.set_ylabel("True")
        # Annotate with counts
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                n = cm[i, j]
                if n > 0:
                    color = "white" if cm_display[i, j] > 0.5 else "black"
                    self.ax_cm.text(j, i, str(n), ha="center", va="center",
                                    color=color, fontsize=9)

    def _draw_per_class_accuracy(self, per_class_history: list):
        """Stacked blocks (green=correct, red=incorrect) per class, with summary text.

        per_class_history is a list of lists: per_class_history[class_idx] is
        a chronological list of booleans, one per prediction of that class
        (True = correct, False = incorrect).
        """
        self.ax_accuracy.clear()
        self.ax_accuracy.set_title("Per-class results (running)", fontsize=11)
        self.ax_accuracy.set_yticks(range(NUM_CLASSES))
        self.ax_accuracy.set_yticklabels(CLASS_NAMES, fontsize=9)
        self.ax_accuracy.invert_yaxis()

        # Find the maximum number of blocks across classes to size x-axis
        max_blocks = max((len(h) for h in per_class_history), default=0)
        # Reserve space on the right for the "k/n (XX%)" summary label.
        # Assume a summary takes ~8 blocks worth of width.
        label_reserve = 8
        x_upper = max(max_blocks + label_reserve, 15)
        self.ax_accuracy.set_xlim(-0.5, x_upper)
        self.ax_accuracy.set_xticks([])
        self.ax_accuracy.set_xlabel("")

        # Draw each block as a small square
        block_size = 0.8
        for class_idx in range(NUM_CLASSES):
            history = per_class_history[class_idx]
            for block_idx, was_correct in enumerate(history):
                color = "#2ca02c" if was_correct else "#d62728"
                rect = Rectangle(
                    (block_idx, class_idx - block_size / 2),
                    width=block_size, height=block_size,
                    facecolor=color, edgecolor="white", linewidth=1.0,
                )
                self.ax_accuracy.add_patch(rect)

            # Summary label at the end of the row
            n_total = len(history)
            if n_total > 0:
                n_correct = sum(history)
                pct = n_correct / n_total
                label = f"  {n_correct}/{n_total} ({pct:.0%})"
            else:
                label = "  (not yet seen)"
            self.ax_accuracy.text(
                max(n_total, 0) + 0.2, class_idx,
                label, va="center", fontsize=9,
                fontweight="bold" if n_total > 0 else "normal",
                color="#333" if n_total > 0 else "#999",
            )

        self.ax_accuracy.set_ylim(NUM_CLASSES - 0.5, -0.5)
        # Hide the frame/spines for a cleaner look
        for spine in ("top", "right", "bottom"):
            self.ax_accuracy.spines[spine].set_visible(False)

    def show_per_class_accuracy(self, per_class_history: list):
        self._draw_per_class_accuracy(per_class_history)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def show_clip_start(self, clip_idx: int, total: int, filename: str,
                         true_name: str, note: str):
        """Called when a clip is about to play — clears old prediction state."""
        # Reset probabilities to grey
        self.ax_probs.clear()
        self.ax_probs.set_title("Predicted probabilities", fontsize=11)
        self.ax_probs.set_xlim(0, 1)
        self.ax_probs.set_xlabel("Probability")
        self.ax_probs.set_yticks(range(NUM_CLASSES))
        self.ax_probs.set_yticklabels(CLASS_NAMES)
        self.ax_probs.invert_yaxis()
        self.ax_probs.barh(range(NUM_CLASSES), [0] * NUM_CLASSES, color="#ddd")

        # Reset latency bar
        self.ax_latency.clear()
        self.ax_latency.set_title("Latency timeline (ms)", fontsize=11)
        self.ax_latency.set_xlim(0, 2600)
        self.ax_latency.set_ylim(0, 1)
        self.ax_latency.set_yticks([])
        self.ax_latency.set_xlabel("Time (ms)")

        # Status banner shows the clip info but hides the ground truth until reveal
        status = (
            f"Clip {clip_idx + 1} of {total}: {filename}\n"
            f"{note}\n\n"
            f"[Playing audio... Arduino is listening]"
        )
        self._set_status_banner(status, color="#0a5")

    def show_mel(self, mel_db: np.ndarray):
        self.ax_mel.clear()
        self.ax_mel.set_title("Mel-Spectrogram (what the Arduino \"hears\")", fontsize=11)
        self.ax_mel.set_xlabel("Time (frames)")
        self.ax_mel.set_ylabel("Mel band")
        self.ax_mel.imshow(mel_db, aspect="auto", origin="lower", cmap="magma",
                           vmin=mel_db.min(), vmax=mel_db.max())
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def _draw_latency_frame(self, capture_ms: float, feat_ms: float,
                             infer_ms: float, elapsed_ms: float | None = None,
                             phase: str | None = None) -> None:
        """Render one frame of the latency timeline.

        If elapsed_ms is given, only the leading segments are drawn filled
        up to that wall-clock elapsed time (animation in progress). When
        elapsed_ms is None, the full three-segment bar is drawn (final state).
        """
        self.ax_latency.clear()
        self.ax_latency.set_title("Latency timeline (ms)", fontsize=11)
        self.ax_latency.set_xlim(0, 2600)
        self.ax_latency.set_ylim(0, 1)
        self.ax_latency.set_yticks([])
        self.ax_latency.set_xlabel("Time (ms)")

        segments = [
            ("Audio capture", capture_ms, "#6baed6"),
            ("Feature extraction", feat_ms, "#fd8d3c"),
            ("Inference", infer_ms, "#74c476"),
        ]
        total = capture_ms + feat_ms + infer_ms

        x = 0.0
        for seg_label, seg_width, seg_color in segments:
            if elapsed_ms is None:
                # Final state: draw full segment
                draw_width = seg_width
            else:
                # Animating: fill up to the elapsed time
                draw_width = max(0.0, min(seg_width, elapsed_ms - x))
            if draw_width > 0:
                self.ax_latency.add_patch(Rectangle(
                    (x, 0.15), draw_width, 0.7,
                    facecolor=seg_color, edgecolor="black", linewidth=0.5,
                ))
                # Label only when segment is substantially filled and wide enough
                if draw_width > 100 and (elapsed_ms is None or draw_width >= seg_width * 0.6):
                    self.ax_latency.text(
                        x + draw_width / 2, 0.5, f"{seg_label}\n{seg_width:.0f} ms",
                        ha="center", va="center", fontsize=9,
                    )
            # Draw a light outline for unfilled portion
            if elapsed_ms is not None and draw_width < seg_width:
                self.ax_latency.add_patch(Rectangle(
                    (x + draw_width, 0.15), seg_width - draw_width, 0.7,
                    facecolor="none", edgecolor="#ccc", linewidth=0.5,
                    linestyle="--",
                ))
            x += seg_width

        # Total label on the right
        if elapsed_ms is None:
            total_label = f"Total: {total:.0f} ms"
        else:
            total_label = f"{min(elapsed_ms, total):.0f} / {total:.0f} ms"
        self.ax_latency.text(
            total + 50, 0.5, total_label,
            ha="left", va="center", fontsize=10, fontweight="bold",
        )

        # Phase indicator (optional)
        if phase:
            self.ax_latency.text(
                10, 0.95, f"▶ {phase}", ha="left", va="top",
                fontsize=9, color="#0a5", fontweight="bold",
            )

    def animate_latency_until(self, capture_ms: float, feat_ms: float,
                               infer_ms: float, stop_event_check) -> None:
        """Progressively fill the latency bar in real wall-clock time.

        Uses the supplied (expected) timings. The animation runs until either
        the total elapsed time is reached OR stop_event_check() returns True
        (which is how we stop early when the real RESULT arrives from the
        Arduino). stop_event_check is a zero-arg callable.
        """
        total_ms = capture_ms + feat_ms + infer_ms
        start = time.time()
        frame_interval = 0.08  # ~12 fps — smooth enough, low CPU
        last_draw = 0.0
        while True:
            elapsed_ms = (time.time() - start) * 1000.0
            if elapsed_ms >= total_ms:
                break
            if stop_event_check():
                break
            # Determine which phase we're in for the indicator
            if elapsed_ms < capture_ms:
                phase = "Audio capture"
            elif elapsed_ms < capture_ms + feat_ms:
                phase = "Feature extraction"
            else:
                phase = "Inference"
            # Throttle redraws to frame_interval
            if elapsed_ms - last_draw >= frame_interval * 1000:
                self._draw_latency_frame(
                    capture_ms, feat_ms, infer_ms,
                    elapsed_ms=elapsed_ms, phase=phase,
                )
                self.fig.canvas.draw_idle()
                plt.pause(0.01)
                last_draw = elapsed_ms
            else:
                plt.pause(0.01)

    def show_latency(self, capture_ms: float, feat_ms: float, infer_ms: float):
        """Draw the final static latency timeline using the actual measured values."""
        self._draw_latency_frame(capture_ms, feat_ms, infer_ms)

    def show_prediction(self, result: ClipResult):
        # Update probability bars with correct/incorrect coloring
        self.ax_probs.clear()
        self.ax_probs.set_title("Predicted probabilities", fontsize=11)
        self.ax_probs.set_xlim(0, 1.05)
        self.ax_probs.set_xlabel("Probability")
        self.ax_probs.set_yticks(range(NUM_CLASSES))
        self.ax_probs.set_yticklabels(CLASS_NAMES)
        self.ax_probs.invert_yaxis()

        colors = ["#ddd"] * NUM_CLASSES
        pred_color = "#2ca02c" if result.correct else "#d62728"  # green or red
        colors[result.predicted_label] = pred_color
        # Mark the true label with a yellow edge if different from prediction
        edges = ["none"] * NUM_CLASSES
        if not result.correct:
            edges[result.true_label] = "gold"

        bars = self.ax_probs.barh(
            range(NUM_CLASSES), result.probabilities, color=colors,
            edgecolor=edges, linewidth=[2.5 if e != "none" else 0 for e in edges],
        )
        for bar, p in zip(bars, result.probabilities):
            if p > 0.02:
                self.ax_probs.text(
                    p + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{p:.2f}", va="center", fontsize=9,
                )

        # Update status banner with prediction reveal
        status_color = "#2ca02c" if result.correct else "#d62728"
        verdict = "CORRECT" if result.correct else "INCORRECT"
        status = (
            f"Predicted: {result.predicted_name}   (confidence {result.confidence:.3f})\n"
            f"True label: {result.true_name}\n"
            f"Verdict: {verdict}"
        )
        self._set_status_banner(status, color=status_color)

        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def update_confusion_matrix(self, cm: np.ndarray):
        self._draw_confusion(cm)
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

    def show_final_summary(self, results: list):
        if not results:
            self._set_status_banner("No clips played. Goodbye.", color="#333")
            return

        n_correct = sum(1 for r in results if r.correct)
        n_total = len(results)
        acc = n_correct / n_total
        avg_feat = np.mean([r.feat_ms for r in results])
        avg_infer = np.mean([r.infer_ms for r in results])
        avg_conf_correct = np.mean([r.confidence for r in results if r.correct]) if n_correct > 0 else 0
        n_incorrect = n_total - n_correct
        avg_conf_incorrect = (
            np.mean([r.confidence for r in results if not r.correct])
            if n_incorrect > 0 else 0
        )

        # Color-code by accuracy tier
        if acc >= 0.7:
            banner_color = "#0a5"  # green
        elif acc >= 0.5:
            banner_color = "#c60"  # orange
        else:
            banner_color = "#b22"  # red

        # Build a two-line summary: headline accuracy + timing/confidence stats
        lines = [
            f"DEMO COMPLETE — {n_correct}/{n_total} correct  ({acc:.1%} accuracy)",
            f"Avg feature extraction: {avg_feat:.0f} ms   |   "
            f"Avg inference: {avg_infer:.0f} ms   |   "
            f"Confidence: {avg_conf_correct:.2f} correct vs {avg_conf_incorrect:.2f} incorrect",
            "",
            "Thank you! Questions?",
        ]
        summary = "\n".join(lines)
        self._set_status_banner(summary, color=banner_color)

        # Enlarge the headline so it stands out on the projector
        # (status banner renders at fontsize=14 by default; the first line
        # is already visually dominant because it's on its own line with
        # caps, but we can reinforce it by redrawing with bigger text)
        self.ax_status.clear()
        self.ax_status.axis("off")
        self.ax_status.text(
            0.5, 0.80, lines[0], ha="center", va="center",
            fontsize=20, fontweight="bold", color=banner_color,
            transform=self.ax_status.transAxes,
        )
        self.ax_status.text(
            0.5, 0.40, lines[1], ha="center", va="center",
            fontsize=11, color="#333",
            transform=self.ax_status.transAxes,
        )
        self.ax_status.text(
            0.5, 0.10, "Thank you! Questions?", ha="center", va="center",
            fontsize=13, fontweight="bold", color=banner_color,
            transform=self.ax_status.transAxes,
        )
        self.fig.canvas.draw_idle()
        plt.pause(0.05)

        if self.save_snapshots:
            out = RESULTS_DIR / "demo_final_dashboard.png"
            self.fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved final dashboard: {out}")


# ─── Main demo loop ───────────────────────────────────────────────────────


def run_demo(state: DemoState, ser: serial.Serial, dashboard: DemoDashboard,
             volume: float, view_time: float, num_loops: int,
             save_snapshots: bool) -> None:

    while state.current_idx < len(state.clips):
        clip_path, true_label, note = state.clips[state.current_idx]
        full_path = DATASET_DIR / clip_path
        filename = Path(clip_path).name

        if not full_path.exists():
            print(f"  WARNING: {full_path} not found — skipping")
            state.current_idx += 1
            continue

        # Prompt for action
        key = dashboard.wait_for_key(
            f"Clip {state.current_idx + 1} of {len(state.clips)} ready: {filename}   |   "
            f"[SPACE] play   [s] skip   [f] finish   [q] quit"
        )

        if key == "q":
            print("  Quit requested.")
            return
        if key == "f":
            print(f"  Finish requested after {len(state.results)} clip(s) played.")
            return
        if key == "s":
            print(f"  Skipped clip {state.current_idx + 1}")
            state.current_idx += 1
            continue
        # SPACE or 'r' (first time) → play

        # Show "playing" status and update mel-spectrogram display
        dashboard.show_clip_start(
            state.current_idx, len(state.clips), filename,
            CLASS_NAMES[true_label], note,
        )
        try:
            mel_db = compute_mel_for_display(full_path)
            dashboard.show_mel(mel_db)
        except Exception as e:
            print(f"  WARNING: could not compute mel for display: {e}")

        # Drain any pending serial, then trigger Arduino and play audio
        while ser.in_waiting:
            ser.readline()
        ser.write(b"READY\n")
        if not wait_for_listening(ser, timeout=5.0):
            print("  WARNING: Arduino did not confirm listening. Proceeding anyway.")

        # Estimate expected per-phase timings for the animation. If we've
        # already seen some results, use the running mean; otherwise fall
        # back to sensible defaults from prior Phase 6 measurements.
        if state.results:
            exp_capture = float(np.mean([
                max(0.0, r.total_ms - r.feat_ms - r.infer_ms) for r in state.results
            ]))
            exp_feat = float(np.mean([r.feat_ms for r in state.results]))
            exp_infer = float(np.mean([r.infer_ms for r in state.results]))
        else:
            exp_capture, exp_feat, exp_infer = 1490.0, 197.0, 785.0

        # Start audio playback asynchronously so the animation can proceed
        try:
            play_data, play_sr = build_playback_buffer(full_path, volume, num_loops)
        except Exception as e:
            print(f"  ERROR loading audio: {e}")
            state.current_idx += 1
            continue
        start_playback_async(play_data, play_sr)

        # Animate the latency timeline while polling for the Arduino RESULT.
        # The animation stops as soon as the RESULT arrives or it completes.
        raw_container: dict = {"raw": None}

        def check_result_ready() -> bool:
            if raw_container["raw"] is not None:
                return True
            # Peek at the serial buffer without blocking
            if ser.in_waiting:
                try:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                except Exception:
                    return False
                if line.startswith(RESULT_MARKER):
                    parsed = parse_result_line(line)
                    if parsed is not None:
                        raw_container["raw"] = parsed
                        return True
            return False

        dashboard.animate_latency_until(
            exp_capture, exp_feat, exp_infer, check_result_ready,
        )

        # If the animation finished before the RESULT arrived, fall back to
        # the original blocking wait with a short timeout.
        if raw_container["raw"] is None:
            raw_container["raw"] = wait_for_result(ser, timeout=8.0)

        # Ensure audio playback doesn't block the next iteration if audio
        # is still playing (the loops may extend beyond the inference cycle).
        # We don't sd.stop() — the looped audio continues for the audience,
        # but we don't wait for it here. The next iteration's drain-and-READY
        # sequence will interrupt naturally.

        raw = raw_container["raw"]
        if raw is None:
            print("  ERROR: no result received from Arduino.")
            sd.stop()
            state.current_idx += 1
            continue

        pred_label = raw["predicted_class"]
        result = ClipResult(
            clip_idx=state.current_idx,
            filename=filename,
            true_label=true_label,
            true_name=CLASS_NAMES[true_label],
            predicted_label=pred_label,
            predicted_name=CLASS_NAMES[pred_label] if 0 <= pred_label < NUM_CLASSES else f"Unknown({pred_label})",
            confidence=raw["confidence"],
            probabilities=raw["probabilities"],
            feat_ms=raw["feat_ms"],
            infer_ms=raw["infer_ms"],
            total_ms=raw["total_ms"],
            correct=(pred_label == true_label),
        )
        state.results.append(result)
        state.confusion_matrix[true_label, pred_label] += 1
        state.per_class_history[true_label].append(result.correct)

        # Update visualizations
        capture_ms = max(0.0, result.total_ms - result.feat_ms - result.infer_ms)
        dashboard.show_latency(capture_ms, result.feat_ms, result.infer_ms)
        dashboard.show_prediction(result)
        dashboard.update_confusion_matrix(state.confusion_matrix)
        dashboard.show_per_class_accuracy(state.per_class_history)

        if save_snapshots:
            out = RESULTS_DIR / f"demo_snapshot_{state.current_idx + 1:02d}_{filename}.png"
            dashboard.fig.savefig(out, dpi=120, bbox_inches="tight")

        print(f"  Clip {state.current_idx + 1}: true={result.true_name}, "
              f"pred={result.predicted_name}, conf={result.confidence:.3f}, "
              f"{'CORRECT' if result.correct else 'INCORRECT'}")

        # Hold on the results for the configured view time, but allow the
        # presenter to advance immediately by pressing SPACE.
        hold_start = time.time()
        dashboard.pending_key = None
        while time.time() - hold_start < view_time:
            if dashboard.pending_key is not None:
                break
            plt.pause(0.1)

        # If the presenter pressed 'r' during the hold, replay the same clip
        if dashboard.pending_key == "r":
            dashboard.pending_key = None
            continue  # don't increment — re-enter loop on same clip
        if dashboard.pending_key == "q":
            return
        if dashboard.pending_key == "f":
            print(f"  Finish requested after {len(state.results)} clip(s) played.")
            return

        state.current_idx += 1


# ─── Clip list loading ────────────────────────────────────────────────────


DEFAULT_CLIPS_JSON = Path(__file__).resolve().parent / "presentation_demo_clips.json"


def load_clip_list(clips_file: Path | None) -> list:
    """Load a list of clips from a JSON file.

    Preference order:
      1. Explicit --clips path (if provided)
      2. presentation_demo_clips.json next to this script (if it exists)
      3. Hardcoded DEFAULT_CLIPS fallback

    JSON format:
      [{"path": "idle state/low_oil/low_oil_9.wav", "true_label": 3, "note": "..."}]
    """
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
        description="Live presentation demo for car sound classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", required=True,
                        help="Serial port (e.g., COM3, /dev/ttyS3, /dev/ttyACM0)")
    parser.add_argument("--volume", type=float, default=0.8,
                        help="Playback volume (0.0-1.0)")
    parser.add_argument("--view-time", type=float, default=6.0,
                        help="Seconds to hold the result display after each clip. "
                             "Presenter can press SPACE to advance sooner.")
    parser.add_argument("--loops", type=int, default=1,
                        help="Number of times to play each clip. The Arduino only "
                             "captures the first 1.5s; additional loops let the "
                             "audience hear the sound repeated. (e.g., --loops 3 "
                             "means 3 plays = ~4.8s total audible with gaps.)")
    parser.add_argument("--clips", type=Path, default=None,
                        help="Optional JSON file specifying custom clips to play. "
                             "Format: [{path, true_label, note}]")
    parser.add_argument("--save-snapshots", action="store_true",
                        help="Save a PNG of the dashboard after each clip "
                             "(useful for post-presentation review)")
    args = parser.parse_args()

    print("=" * 70)
    print("  Car Sound Classifier — Live Presentation Demo")
    print("=" * 70)

    clips = load_clip_list(args.clips)
    print(f"  Loaded {len(clips)} clips")
    print(f"  View time per clip: {args.view_time}s (adjustable, press SPACE to advance)")
    if args.loops > 1:
        print(f"  Audio loops: {args.loops} (~{args.loops * 1.6:.1f}s per clip)")

    # Verify all clip files exist before we start
    missing = [p for p, _, _ in clips if not (DATASET_DIR / p).exists()]
    if missing:
        print("\n  ERROR: Missing clip files:")
        for p in missing:
            print(f"    {p}")
        sys.exit(1)

    try:
        ser = connect_serial(args.port)
    except serial.SerialException as e:
        print(f"  ERROR: {e}")
        print(f"  Is the Arduino connected? Is the port '{args.port}' correct?")
        sys.exit(1)

    state = DemoState(clips=clips)
    dashboard = DemoDashboard(view_time=args.view_time, save_snapshots=args.save_snapshots)

    try:
        run_demo(state, ser, dashboard, volume=args.volume,
                 view_time=args.view_time, num_loops=args.loops,
                 save_snapshots=args.save_snapshots)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        ser.close()
        dashboard.show_final_summary(state.results)
        print("\n  Demo finished. Press any key on the window to close.")
        try:
            plt.show(block=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
