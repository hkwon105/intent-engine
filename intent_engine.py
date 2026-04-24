"""
intent_engine.py  Self-Contained CNN-LSTM Intent Prediction Engine
====================================================================
Single file. No local imports. Ship this file + your weights file
(.pth) + your annotation JSON to the target device and run.

Dependencies (install on target device):
    pip install torch torchvision opencv-python pillow numpy
    sudo chmod 777 /dev/video10 

Usage
-----
    python3 /home/robotics/Documents/human_robot_collab/test/gearbox/intent_engine.py \
    --resume_path /home/robotics/Documents/human_robot_collab/test/gearbox/cnnlstm-Epoch-196-Loss-0.01737015192823795.pth \
    --annotation_path /home/robotics/Documents/human_robot_collab/test/gearbox/intentpredictionattempt4-1/datasets/labels.json \
    --n_classes 3 \
    --sample_size 150 \
    --sample_duration 8 \
    --smoothing_window 3 \
    --confidence_threshold 0.75

The engine will start the RealSense camera, run inference in a
background thread, and print state to stdout. Integrate by importing
IntentEngine and EngineConfig directly, or subclass / extend as needed.
"""

from __future__ import annotations

# std lib
import argparse
import collections
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

# third party
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
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# Model definition  (inlined from models/cnnlstm.py  no local imports needed)
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

class CNNLSTM(nn.Module):
    """ResNet-101 feature extractor + 3-layer LSTM classifier."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Suppress the deprecated pretrained= warning on newer torchvision
        try:
            from torchvision.models import ResNet101_Weights
            self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        except ImportError:
            self.resnet = resnet101(pretrained=True)  # torchvision < 0.13

        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 300)
        )
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1  = nn.Linear(256, 128)
        self.fc2  = nn.Linear(128, num_classes)

    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        # x_3d: (batch, time, C, H, W)
        hidden = None
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# Camera
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

class RealSenseCamera:
    """Jetson-Optimized camera class using OpenCV Index 10."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        # We use index 10 and V4L2 backend as confirmed for your setup
        self.cap = cv2.VideoCapture(10, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera at index 10. Check: sudo chmod 777 /dev/video10")
        print("[INFO] Camera initialized successfully at index 10")

    def read(self):
        """Return (True, bgr_ndarray) or (False, None)."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self):
        self.cap.release()


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# Configuration
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

@dataclass
class EngineConfig:
    # required
    resume_path: str = ""           # path to .pth weights file

    # model 
    n_classes: int   = 3

    # inference 
    sample_size: int            = 150   # spatial resize (px)
    sample_duration: int        = 16    # sliding window length (frames)
    confidence_threshold: float = 0.6
    smoothing_window: int       = 5     # temporal vote window

    # hardware 
    gpu: int = 0

    # labels 
    # Provide one of:  annotation_path  OR  class_labels
    annotation_path: str       = ""
    class_labels: List[str]    = field(default_factory=list)

    # flag mapping 
    stop_class:    str = "INTERACTION"
    caution_class: str = "PASSTHRU"


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# Inference engine
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

class IntentEngine:
    """
    Thread-safe inference engine.

    Lifecycle
    ---------
        engine = IntentEngine(cfg)
        engine.start()
        # push frames from any source
        engine.push_frame(bgr_numpy_array)
        state = engine.get_state()
        engine.stop()

    Callback
    --------
        def on_flag_change(flag: str, confidence: float): ...
        engine = IntentEngine(cfg, on_flag_change=on_flag_change)

    Output flags
    ------------
        IntentEngine.FLAG_STOP     INTERACTION above threshold
        IntentEngine.FLAG_CAUTION  PASSTHRU detected
        IntentEngine.FLAG_CLEAR    normal / waiting
    """

    FLAG_STOP    = "STOP"
    FLAG_CAUTION = "CAUTION"
    FLAG_CLEAR   = "CLEAR"

    def __init__(
        self,
        cfg: EngineConfig,
        on_flag_change: Optional[Callable[[str, float], None]] = None,
    ):
        self.cfg = cfg
        self._on_flag_change = on_flag_change

        # Device
        self.device = torch.device(
            f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
        )

        # Labels
        if cfg.class_labels:
            self.class_labels: List[str] = cfg.class_labels
        elif cfg.annotation_path:
            with open(cfg.annotation_path, "r") as f:
                self.class_labels = json.load(f)["labels"]
        else:
            raise ValueError(
                "Set cfg.class_labels or cfg.annotation_path."
            )

        self._transform = self._build_transform()
        self._model     = self._load_model()

        # Sliding buffers
        self._raw_frames   = collections.deque(maxlen=cfg.sample_duration)
        self._pred_history = collections.deque(maxlen=cfg.smoothing_window)

        # Shared state
        self._lock         = threading.Lock()
        self._probs        = np.zeros(len(self.class_labels))
        self._label        = self.class_labels[-1]
        self._confidence   = 0.0
        self._flag         = self.FLAG_CLEAR
        self._prev_flag: Optional[str] = None
        self._infer_fps    = 0.0
        self._last_infer_t = time.time()

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # lifecycle 

    def start(self):
        """Start the background inference thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
            name="IntentEngine-infer",
        )
        self._thread.start()

    def stop(self):
        """Stop the background inference thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # frame ingestion 

    def push_frame(self, bgr_frame: np.ndarray):
        """
        Push one BGR uint8 frame (H�W�3 ndarray).
        Non-blocking safe to call from a camera thread at full frame rate.
        """
        self._raw_frames.append(bgr_frame)

    # state readout 

    def get_state(self) -> dict:
        """
        Snapshot of the latest inference result.

        Keys
        ----
            flag        : str           STOP | CAUTION | CLEAR
            label       : str           raw class name
            confidence  : float         softmax probability of predicted class
            probs       : np.ndarray    full distribution over all classes
            infer_fps   : float         inference loop rate
        """
        with self._lock:
            return {
                "flag":       self._flag,
                "label":      self._label,
                "confidence": self._confidence,
                "probs":      self._probs.copy(),
                "infer_fps":  self._infer_fps,
            }

    @property
    def flag(self) -> str:
        with self._lock:
            return self._flag

    @property
    def should_stop(self) -> bool:
        """Convenience boolean for control loops."""
        return self.flag == self.FLAG_STOP

    # internals
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
        model = CNNLSTM(num_classes=self.cfg.n_classes).to(self.device)
        if not self.cfg.resume_path:
            raise ValueError("cfg.resume_path must point to a .pth weights file.")
        checkpoint = torch.load(self.cfg.resume_path, map_location=self.device)
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state)
        model.eval()
        model.resnet.eval()
        print(f"[IntentEngine] Loaded weights from: {self.cfg.resume_path}")
        print(f"[IntentEngine] Running on: {self.device}")
        return model

    def _inference_loop(self):
        while self._running:
            if len(self._raw_frames) < self.cfg.sample_duration:
                time.sleep(0.01)
                continue
            frames   = list(self._raw_frames)          # snapshot
            t_frames = self._preprocess(frames)
            self._run_inference(t_frames)

    def _preprocess(self, frames: list) -> list:
        out = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            out.append(self._transform(Image.fromarray(rgb)))
        return out

    def _run_inference(self, t_frames: list):
        clip = torch.stack(t_frames).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(clip)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        self._pred_history.append(pred_idx)

        # Recency-weighted temporal smoothing
        weights = np.linspace(0.5, 1.0, len(self._pred_history))
        votes   = np.zeros(len(self.class_labels))
        for w, idx in zip(weights, self._pred_history):
            votes[idx] += w
        smoothed_idx = int(np.argmax(votes))

        label = self.class_labels[smoothed_idx]
        conf  = float(probs[smoothed_idx])
        flag  = self._resolve_flag(label, conf)

        now = time.time()
        fps = 1.0 / max(now - self._last_infer_t, 1e-6)
        self._last_infer_t = now

        with self._lock:
            self._probs      = probs
            self._label      = label
            self._confidence = conf
            self._flag       = flag
            self._infer_fps  = fps
            prev             = self._prev_flag

        if flag != prev:
            with self._lock:
                self._prev_flag = flag
            if self._on_flag_change:
                try:
                    self._on_flag_change(flag, conf)
                except Exception as exc:
                    print(f"[IntentEngine] on_flag_change raised: {exc}")

    def _resolve_flag(self, label: str, confidence: float) -> str:
        if label == self.cfg.stop_class and confidence >= self.cfg.confidence_threshold:
            return self.FLAG_STOP
        if label == self.cfg.caution_class:
            return self.FLAG_CAUTION
        return self.FLAG_CLEAR


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# Camera capture loop  (runs in its own thread)
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

class CaptureThread:
    """
    Reads frames from a RealSenseCamera and feeds them into an IntentEngine.
    Runs as a daemon thread dies automatically when the main process exits.
    """

    def __init__(self, camera: RealSenseCamera, engine: IntentEngine):
        self._camera  = camera
        self._engine  = engine
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop,
            daemon=True,
            name="IntentEngine-capture",
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._camera.release()

    def _loop(self):
        while self._running:
            ret, frame = self._camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            self._engine.push_frame(frame)


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# CLI entry-point  (for standalone testing on the target device)
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intent Prediction Engine")
    p.add_argument("--resume_path",      required=True,
                   help="Path to model weights (.pth)")
    p.add_argument("--annotation_path",  default="",
                   help="Path to annotation JSON (for class labels)")
    p.add_argument("--class_labels",     nargs="+", default=[],
                   help="Class labels in order (alternative to annotation_path)")
    p.add_argument("--n_classes",        type=int,   default=3)
    p.add_argument("--sample_size",      type=int,   default=150)
    p.add_argument("--sample_duration",  type=int,   default=16)
    p.add_argument("--confidence_threshold", type=float, default=0.6)
    p.add_argument("--smoothing_window", type=int,   default=5)
    p.add_argument("--gpu",              type=int,   default=0)
    return p.parse_args()


def main():
    args = _parse_args()

    cfg = EngineConfig(
        resume_path          = args.resume_path,
        annotation_path      = args.annotation_path,
        class_labels         = args.class_labels,
        n_classes            = args.n_classes,
        sample_size          = args.sample_size,
        sample_duration      = args.sample_duration,
        confidence_threshold = args.confidence_threshold,
        smoothing_window     = args.smoothing_window,
        gpu                  = args.gpu,
    )

    # Example callback replace with your robot arm integration
    def on_flag_change(flag: str, confidence: float):
        print(f"[FLAG CHANGE] {flag}  (conf={confidence:.2f})")

    print("[INFO] Initialising engine ...")
    engine = IntentEngine(cfg, on_flag_change=on_flag_change)

    print("[INFO] Starting RealSense camera ...")
    camera  = RealSenseCamera()
    capture = CaptureThread(camera, engine)

    engine.start()
    capture.start()
    print("[INFO] Running Ctrl-C to quit\n")

    try:
        while True:
            state = engine.get_state()
            print(
                f"\r  flag={state['flag']:<8}  "
                f"label={state['label']:<14}  "
                f"conf={state['confidence']:.2f}  "
                f"fps={state['infer_fps']:.1f}   ",
                end="", flush=True,
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down ...")
    finally:
        capture.stop()
        engine.stop()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
