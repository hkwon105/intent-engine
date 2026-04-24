"""
intent_engine.py - Self-Contained CNN-LSTM Intent Prediction Engine

Single file. No local imports. Ship this file + your weights (.pth)
+ your annotation JSON to the target device and run.

Dependencies:
    pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pillow opencv-python-headless

Usage:
    python3 intent_engine.py \
        --resume_path     /path/to/weights.pth \
        --annotation_path /path/to/labels.json \
        --n_classes        3 \
        --sample_size      150 \
        --sample_duration  8 \
        --smoothing_window 3 \
        --confidence_threshold 0.75

FIXES vs previous versions
---------------------------
FIX 1 - smoothed_conf was the wrong scale.
    The confidence threshold was trained/tuned against raw softmax
    probabilities (0.0-1.0 per class). A previous "fix" changed
    smoothed_conf to a vote-fraction (weighted sum / total weight),
    which is a different scale entirely and broke the threshold
    comparison. Reverted: confidence is now float(probs[smoothed_idx]),
    matching exactly what the original live_inference_gui.py used.

FIX 2 - ResNet runs with gradient tracking inside the LSTM loop.
    model.forward() calls self.resnet() once per timestep inside a
    Python for-loop. torch.no_grad() on the outer clip inference only
    covers the LSTM+FC portion if the resnet call escapes the context.
    Fixed: model.resnet is explicitly set to eval() AND torch.no_grad()
    is applied as a persistent context via torch.inference_mode() for
    the entire inference call, which is zero-overhead on CPU (Jetson).

FIX 3 - _prev_flag race condition (TOCTOU).
    prev = self._prev_flag was read inside lock acquisition #1, then
    `if flag != prev` and the _prev_flag write happened in a separate
    lock acquisition #2. Another thread could update _prev_flag between
    the two acquisitions, causing on_flag_change to fire twice or not
    at all. Fixed: the entire read-compare-write is now one atomic
    block inside a single lock acquisition.

FIX 4 - Hardcoded labels["labels"] key crashes on Jetson labels.json.
    The Jetson labels.json uses numeric string keys {"0": "INTERACTION",
    "1": "PASSTHRU", "2": "WAIT"} not the {"labels": [...]} format.
    The hardcoded json.load(f)["labels"] raises KeyError silently,
    leaving class_labels empty and producing random index lookups.
    Fixed: _load_labels() tries all known formats in order.

FIX 5 - Inference loop spins at 100% CPU between forward passes.
    No rate limiting means the Jetson inference thread runs as fast as
    possible, starving the camera capture thread and causing thermal
    throttle. Fixed: max_infer_hz cap (default 10 Hz) with sleep for
    remainder of frame budget after each forward pass.

FIX 6 - Silent thread death on any exception.
    Any exception inside _inference_loop killed the thread permanently
    with no log, no restart, and stale state printed forever. Fixed:
    the loop catches all exceptions, logs them with traceback, and
    restarts after a short back-off delay.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("IntentEngine")


class CNNLSTM(nn.Module):
    """
    ResNet-101 spatial feature extractor + 3-layer LSTM temporal classifier.
    Architecture matches the original cnnlstm.py exactly so checkpoint keys
    load without remapping.

    Input  : (batch, time, C, H, W)
    Output : (batch, num_classes) logits
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet101_Weights
            backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        except ImportError:
            backbone = resnet101(pretrained=True)

        # nn.Sequential wrapper is required so checkpoint keys are
        # resnet.fc.0.weight / resnet.fc.0.bias, matching the saved file.
        backbone.fc = nn.Sequential(nn.Linear(backbone.fc.in_features, 300))
        self.resnet = backbone

        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3,
                            batch_first=False)
        self.fc1  = nn.Linear(256, 128)
        self.fc2  = nn.Linear(128, num_classes)

    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        hidden = None
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)
        x = F.relu(self.fc1(out[-1]))
        return self.fc2(x)


class _RSSdkCamera:
    """RealSense SDK colour stream."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        if not _RS_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed.")
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self._pipeline.start(cfg)
        except Exception as e:
            raise RuntimeError(f"RealSense pipeline failed: {e}") from e
        log.info("Camera: RealSense SDK")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=200)
            colour = frames.get_color_frame()
            if not colour:
                return False, None
            return True, np.asanyarray(colour.get_data())
        except Exception:
            return False, None

    def release(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:
            pass

    @property
    def backend_name(self) -> str:
        return "realsense_sdk"


class _V4L2Camera:
    """OpenCV V4L2 camera."""

    def __init__(self, index: int = 4,
                 width: int = 640, height: int = 480, fps: int = 30) -> None:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"V4L2 index {index} failed. "
                f"Try: sudo chmod 777 /dev/video{index}"
            )
        for _ in range(5):
            cap.grab()
        self._cap   = cap
        self._index = index
        log.info("Camera: V4L2 index %d", index)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._cap.release()

    @property
    def backend_name(self) -> str:
        return f"v4l2_{self._index}"


class CameraFactory:
    """
    Opens the first available camera backend in priority order:
      1. RealSense SDK
      2. V4L2 index 4  (confirmed Jetson device — update if port changes)
      3. V4L2 index 0
    """

    @staticmethod
    def open(width: int = 640, height: int = 480, fps: int = 30):
        errors = []

        try:
            return _RSSdkCamera(width, height, fps)
        except Exception as e:
            errors.append(f"RealSense: {e}")
            log.warning("RealSense unavailable, trying V4L2 index 4 ...")

        try:
            return _V4L2Camera(4, width, height, fps)
        except Exception as e:
            errors.append(f"V4L2@4: {e}")
            log.warning("V4L2 index 4 failed, trying V4L2 index 0 ...")

        try:
            return _V4L2Camera(0, width, height, fps)
        except Exception as e:
            errors.append(f"V4L2@0: {e}")

        raise RuntimeError("All camera backends failed:\n  " + "\n  ".join(errors))


def _frame_is_valid(frame: Optional[np.ndarray],
                    min_mean: float = 5.0) -> bool:
    """Reject None, wrong shape, and near-black (warming-up) frames."""
    if frame is None:
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    if frame.mean() < min_mean:
        return False
    return True


def _load_labels(annotation_path: str) -> List[str]:
    """
    Parse class labels from the annotation JSON.
    Supports all formats used across the project:
      Format A: {"labels": ["INTERACTION", "PASSTHRU", "WAIT"]}
      Format B: {"0": "INTERACTION", "1": "PASSTHRU", "2": "WAIT"}
      Format C: {"INTERACTION": 0, "PASSTHRU": 1, "WAIT": 2}
    """
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Format A
    for key in ("labels", "classes", "label"):
        if key in data and isinstance(data[key], list):
            return list(data[key])

    # Format B - numeric string keys
    if data and all(k.isdigit() for k in data.keys()):
        ordered = sorted(data.items(), key=lambda kv: int(kv[0]))
        labels  = [v for _, v in ordered]
        log.info("Parsed labels from numeric-keyed JSON: %s", labels)
        return labels

    # Format C - inverted map
    if data and all(isinstance(v, int) for v in data.values()):
        ordered = sorted(data.items(), key=lambda kv: kv[1])
        labels  = [k for k, _ in ordered]
        log.info("Parsed labels from inverted class-map JSON: %s", labels)
        return labels

    raise KeyError(
        f"Unrecognised annotation JSON format. Keys: {list(data.keys())}. "
        "Expected one of: {\"labels\": [...]}, "
        "{\"0\": \"CLASS\", ...}, or {\"CLASS\": 0, ...}"
    )


@dataclass
class EngineConfig:
    resume_path: str = ""

    n_classes: int = 3

    sample_size: int             = 150
    sample_duration: int         = 8
    confidence_threshold: float  = 0.75
    smoothing_window: int        = 3

    # Rate cap to prevent Jetson thermal throttle
    max_infer_hz: float          = 10.0

    # Discard this many frames on startup before inference starts
    camera_warmup_frames: int    = 30

    # Restart inference thread this many seconds after a crash
    thread_restart_delay_s: float = 2.0

    gpu: int = 0

    annotation_path: str     = ""
    class_labels: List[str]  = field(default_factory=list)

    stop_class:    str = "INTERACTION"
    caution_class: str = "PASSTHRU"


class IntentEngine:
    """
    Thread-safe, self-healing CNN-LSTM intent prediction engine.

    Lifecycle:
        engine = IntentEngine(cfg)
        engine.start()
        state  = engine.get_state()
        engine.stop()

    Callback (fires exactly once per flag transition):
        def on_flag_change(flag: str, confidence: float): ...
        engine = IntentEngine(cfg, on_flag_change=on_flag_change)

    Flags:
        IntentEngine.FLAG_STOP         - INTERACTION above confidence_threshold
        IntentEngine.FLAG_CAUTION      - PASSTHRU detected
        IntentEngine.FLAG_CLEAR        - WAIT / nothing
        IntentEngine.FLAG_CAMERA_ERROR - camera returning bad frames
    """

    FLAG_STOP         = "STOP"
    FLAG_CAUTION      = "CAUTION"
    FLAG_CLEAR        = "CLEAR"
    FLAG_CAMERA_ERROR = "CAMERA_ERROR"

    def __init__(
        self,
        cfg: EngineConfig,
        on_flag_change: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.cfg             = cfg
        self._on_flag_change = on_flag_change

        self.device = torch.device(
            f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
        )
        log.info("Device: %s", self.device)

        if cfg.class_labels:
            self.class_labels: List[str] = list(cfg.class_labels)
        elif cfg.annotation_path:
            self.class_labels = _load_labels(cfg.annotation_path)
        else:
            raise ValueError("Set cfg.class_labels or cfg.annotation_path.")
        log.info("Classes: %s", self.class_labels)

        self._model     = self._load_model()
        self._transform = self._build_transform()

        self._camera            = None
        self.camera_backend: str = "not_opened"

        self._raw_frames   = collections.deque(maxlen=cfg.sample_duration)
        self._pred_history = collections.deque(maxlen=cfg.smoothing_window)

        self._warmup_remaining   = cfg.camera_warmup_frames
        self._bad_frame_streak   = 0

        self._lock          = threading.Lock()
        self._probs         = np.zeros(len(self.class_labels))
        self._label:   str  = self.class_labels[-1]
        self._confidence    = 0.0
        self._flag:    str  = self.FLAG_CLEAR
        self._prev_flag: Optional[str] = None
        self._infer_fps     = 0.0
        self._last_infer_t  = time.monotonic()

        self._running       = False
        self._cam_thread:   Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        log.info("Opening camera ...")
        self._camera        = CameraFactory.open()
        self.camera_backend = self._camera.backend_name

        self._running      = True
        self._cam_thread   = threading.Thread(
            target=self._camera_loop, daemon=True, name="IE-capture"
        )
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="IE-infer"
        )
        self._cam_thread.start()
        self._infer_thread.start()
        log.info("Engine started (camera=%s, device=%s).",
                 self.camera_backend, self.device)

    def stop(self) -> None:
        log.info("Stopping engine ...")
        self._running = False
        if self._cam_thread:
            self._cam_thread.join(timeout=5.0)
        if self._infer_thread:
            self._infer_thread.join(timeout=5.0)
        if self._camera:
            self._camera.release()
        log.info("Engine stopped.")

    def push_frame(self, bgr_frame: np.ndarray) -> None:
        """Manually push a BGR uint8 frame when managing your own camera loop."""
        if _frame_is_valid(bgr_frame):
            self._raw_frames.append(bgr_frame)

    def get_state(self) -> dict:
        """Non-blocking snapshot. Keys: flag, label, confidence, probs, infer_fps, camera."""
        with self._lock:
            return {
                "flag":       self._flag,
                "label":      self._label,
                "confidence": self._confidence,
                "probs":      self._probs.copy(),
                "infer_fps":  self._infer_fps,
                "camera":     self.camera_backend,
            }

    @property
    def flag(self) -> str:
        with self._lock:
            return self._flag

    @property
    def should_stop(self) -> bool:
        return self.flag == self.FLAG_STOP

    def _build_transform(self) -> transforms.Compose:
        mean = [114.7748, 107.7354, 99.4750]
        std  = [1.0, 1.0, 1.0]
        return transforms.Compose([
            transforms.Resize((self.cfg.sample_size, self.cfg.sample_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            transforms.Normalize(mean, std),
        ])

    def _load_model(self) -> nn.Module:
        if not self.cfg.resume_path:
            raise ValueError("cfg.resume_path must be set.")
        if not os.path.isfile(self.cfg.resume_path):
            raise FileNotFoundError(f"Weights not found: {self.cfg.resume_path}")

        model = CNNLSTM(num_classes=self.cfg.n_classes).to(self.device)

        load_kwargs: dict = {"map_location": self.device}
        try:
            checkpoint = torch.load(
                self.cfg.resume_path, weights_only=True, **load_kwargs
            )
        except Exception:
            checkpoint = torch.load(self.cfg.resume_path, **load_kwargs)

        state_dict = checkpoint.get("state_dict", checkpoint)

        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"Checkpoint missing keys: {missing}")
        if unexpected:
            log.warning("Checkpoint has extra keys (ignored): %s", unexpected)

        model.eval()
        model.resnet.eval()
        log.info("Loaded weights: %s", self.cfg.resume_path)
        return model

    def _camera_loop(self) -> None:
        MAX_BAD_STREAK = 30

        while self._running:
            try:
                ret, frame = self._camera.read()

                if not ret or not _frame_is_valid(frame):
                    self._bad_frame_streak += 1
                    if self._bad_frame_streak >= MAX_BAD_STREAK:
                        log.error("Camera returning bad frames — raising CAMERA_ERROR.")
                        self._set_flag(self.FLAG_CAMERA_ERROR, 1.0, "CAMERA_ERROR")
                    time.sleep(0.01)
                    continue

                self._bad_frame_streak = 0

                if self._warmup_remaining > 0:
                    self._warmup_remaining -= 1
                    continue

                self._raw_frames.append(frame)

            except Exception:
                log.error("Camera loop exception:\n%s", traceback.format_exc())
                time.sleep(0.5)

    def _inference_loop(self) -> None:
        min_interval = 1.0 / max(self.cfg.max_infer_hz, 0.1)

        while self._running:
            t0 = time.monotonic()
            try:
                self._inference_step()
            except Exception:
                log.error(
                    "Inference loop exception (restarting in %.1fs):\n%s",
                    self.cfg.thread_restart_delay_s,
                    traceback.format_exc(),
                )
                time.sleep(self.cfg.thread_restart_delay_s)
                continue

            sleep_t = min_interval - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _inference_step(self) -> None:
        if len(self._raw_frames) < self.cfg.sample_duration:
            time.sleep(0.02)
            return

        frames = list(self._raw_frames)

        t_frames = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t_frames.append(self._transform(Image.fromarray(rgb)))

        clip = torch.stack(t_frames).unsqueeze(0).to(self.device)

        # FIX 2: torch.inference_mode() covers the entire forward pass
        # including the per-timestep resnet calls inside model.forward(),
        # preventing gradient tracking on the CNN backbone on Jetson CPU.
        with torch.inference_mode():
            logits = self._model(clip)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        self._pred_history.append(pred_idx)

        weights = np.linspace(0.5, 1.0, len(self._pred_history))
        votes   = np.zeros(len(self.class_labels))
        for w, idx in zip(weights, self._pred_history):
            votes[idx] += w

        smoothed_idx = int(np.argmax(votes))
        label        = self.class_labels[smoothed_idx]

        # FIX 1: confidence is raw softmax prob of the smoothed class,
        # exactly matching live_inference_gui.py and the scale that
        # confidence_threshold was tuned against.
        confidence = float(probs[smoothed_idx])

        flag = self._resolve_flag(label, confidence)

        now = time.monotonic()
        fps = 1.0 / max(now - self._last_infer_t, 1e-6)
        self._last_infer_t = now

        self._set_flag(flag, confidence, label, probs=probs, fps=fps)

    def _set_flag(
        self,
        flag:       str,
        confidence: float,
        label:      str,
        probs:      Optional[np.ndarray] = None,
        fps:        float = 0.0,
    ) -> None:
        # FIX 3: entire read-compare-write on _prev_flag is one atomic
        # lock acquisition — no TOCTOU window.
        with self._lock:
            self._flag       = flag
            self._label      = label
            self._confidence = confidence
            self._infer_fps  = fps
            if probs is not None:
                self._probs = probs
            transitioned    = (flag != self._prev_flag)
            self._prev_flag = flag

        if transitioned and self._on_flag_change is not None:
            try:
                self._on_flag_change(flag, confidence)
            except Exception:
                log.error("on_flag_change raised:\n%s", traceback.format_exc())

    def _resolve_flag(self, label: str, confidence: float) -> str:
        if label == self.cfg.stop_class and confidence >= self.cfg.confidence_threshold:
            return self.FLAG_STOP
        if label == self.cfg.caution_class:
            return self.FLAG_CAUTION
        return self.FLAG_CLEAR


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNN-LSTM Intent Prediction Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resume_path",           required=True)
    p.add_argument("--annotation_path",       default="")
    p.add_argument("--class_labels",          nargs="+", default=[])
    p.add_argument("--n_classes",             type=int,   default=3)
    p.add_argument("--sample_size",           type=int,   default=150)
    p.add_argument("--sample_duration",       type=int,   default=8)
    p.add_argument("--confidence_threshold",  type=float, default=0.75)
    p.add_argument("--smoothing_window",      type=int,   default=3)
    p.add_argument("--max_infer_hz",          type=float, default=10.0)
    p.add_argument("--camera_warmup_frames",  type=int,   default=30)
    p.add_argument("--gpu",                   type=int,   default=0)
    p.add_argument("--log_level",             default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(args.log_level)

    cfg = EngineConfig(
        resume_path          = args.resume_path,
        annotation_path      = args.annotation_path,
        class_labels         = args.class_labels,
        n_classes            = args.n_classes,
        sample_size          = args.sample_size,
        sample_duration      = args.sample_duration,
        confidence_threshold = args.confidence_threshold,
        smoothing_window     = args.smoothing_window,
        max_infer_hz         = args.max_infer_hz,
        camera_warmup_frames = args.camera_warmup_frames,
        gpu                  = args.gpu,
    )

    def on_flag_change(flag: str, confidence: float) -> None:
        label = {
            IntentEngine.FLAG_STOP:         "STOP",
            IntentEngine.FLAG_CAUTION:      "CAUTION",
            IntentEngine.FLAG_CLEAR:        "CLEAR",
            IntentEngine.FLAG_CAMERA_ERROR: "CAMERA ERROR",
        }.get(flag, flag)
        print(f"\n[FLAG] {label}  conf={confidence:.2f}\n")

    log.info("Initialising engine ...")
    engine = IntentEngine(cfg, on_flag_change=on_flag_change)

    log.info("Starting engine ...")
    engine.start()
    log.info("Running - Ctrl-C to quit\n")

    try:
        while True:
            s = engine.get_state()
            bars = "  ".join(
                f"{lbl}:{p*100:4.0f}%"
                for lbl, p in zip(engine.class_labels, s["probs"])
            )
            print(
                f"\r{s['flag']:<14}{s['label']:<16}"
                f"conf={s['confidence']:.2f}  "
                f"{s['infer_fps']:.1f}hz  "
                f"{bars}   ",
                end="", flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down ...")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
