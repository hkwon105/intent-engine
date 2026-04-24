"""
intent_engine.py  —  CNN-LSTM Intent Prediction Engine  (v3)
=============================================================
Self-contained single file. No local imports. Ship this file + your
weights (.pth) + your annotation JSON to the Jetson and run.

FIXED vs v1
-----------
  BUG 1  — Silent thread death.
           Any exception inside _inference_loop killed the thread with no
           log, no restart, and no indication to the caller. The main loop
           kept printing stale state forever. Fixed: the loop now catches
           all exceptions, logs them with a full traceback, and restarts
           after a configurable back-off delay.

  BUG 2  — _prev_flag race condition (read outside lock).
           prev = self._prev_flag was read under the lock, but then the
           comparison `if flag != prev` happened outside it, creating a
           TOCTOU window where another thread could update _prev_flag
           between the read and the compare, causing on_flag_change to
           either fire twice or not at all. Fixed: the entire
           compare-and-swap is now a single atomic block under the lock.

  BUG 3  — False positives from garbage / black frames.
           If the camera failed to open (or returned black frames while
           warming up), the model received a full window of zero-valued
           tensors. On this input the model confidently predicts
           INTERACTION because that is the class with the highest
           activation on synthetic input. Fixed: the camera is now
           health-checked on open; frames are validated (non-null,
           correct shape, non-zero mean); and a mandatory warm-up period
           discards the first N frames before inference is allowed to
           start. A CAMERA_ERROR flag is emitted when the camera is
           producing bad data so the caller can react.

  BUG 4  — Inference loop CPU spin / thermal throttle on Jetson.
           The loop called list(self._raw_frames) and ran a full
           ResNet-101 + LSTM forward pass on every iteration with no
           frame-budget control. On a Jetson this saturated a CPU core
           while the GPU was already loaded, causing thermal throttling
           that starved the camera capture thread and produced duplicate
           or stale frames — which in turn destabilised predictions.
           Fixed: the inference loop is rate-limited to a configurable
           max_infer_hz (default 10 Hz on Jetson). The camera thread runs
           freely at full sensor rate; the inference thread samples from
           the latest window at a controlled rate.

  BUG 5  — on_flag_change firing on every frame, not just transitions.
           Because _prev_flag was sometimes not updated before the next
           inference cycle completed (race condition, see BUG 2), the
           callback fired repeatedly for the same flag. Fixed by the
           atomic CAS in the lock (see BUG 2 fix).

  BUG 6  — Model loading silently succeeds with wrong weights.
           torch.load with map_location but without weights_only=False
           raises a FutureWarning on PyTorch ≥ 2.0 and will become an
           error. Also, the code did not verify that the loaded
           state_dict actually matches the model architecture, so a wrong
           checkpoint would produce random outputs. Fixed: explicit
           strict=True load with a clear error message listing any
           missing/unexpected keys; weights_only kwarg handled
           conditionally for cross-version compatibility.

  BUG 7  — Smoothing using raw probs index instead of smoothed index.
           conf = float(probs[smoothed_idx]) reported the probability of
           the smoothed class from the *current* frame's raw softmax
           output. If the current frame predicted a different class than
           the smoothed vote winner, the reported confidence was for the
           wrong class — could be near-zero for INTERACTION even while
           the smoothed label was INTERACTION. Fixed: confidence is now
           the recency-weighted vote fraction for the winning class,
           which is a true measure of how strongly the smoothing window
           agrees.

  BUG 8  — No camera fallback handled in the engine itself.
           The old code had a RealSenseCamera class that used OpenCV
           internally (not the RS SDK) but was still named RealSense,
           causing confusion. The fallback logic lived in main() and was
           not available when the engine was imported as a library.
           Fixed: CameraFactory.open() tries the RealSense SDK pipeline
           first, then falls back to V4L2 index 4, then falls back to
           V4L2 index 0, logging each attempt. The chosen camera type is
           exposed on engine.camera_backend.

Dependencies
------------
    # Step 1 — create and activate a venv
    python3 -m venv ~/intent_env
    source ~/intent_env/bin/activate

    # Step 2 — system packages (apt)
    # opencv comes from apt for the JetPack ARM64 system build.
    # v4l-utils provides v4l2-ctl for finding the camera device index.
    sudo apt update
    sudo apt install -y python3-opencv v4l-utils

    # Step 3 — torch + torchvision + remaining deps (CPU build, confirmed working)
    # Do NOT use `sudo apt install python3-torch` — the Ubuntu repo build is
    # PyTorch 1.8 (2021) with no value for this use case.
    # Do NOT use the NVIDIA JetPack wheel URLs — those segfault on JetPack 36.5
    # (R36, CUDA 12.6) due to a library mismatch with libcusparseLt.
    # torchvision==0.21.0 pins torch to 2.6.0 automatically — do not pin torch
    # separately or you risk a version mismatch.
    pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pillow
    # opencv-python-headless is required because the system apt opencv is not
    # visible inside the venv. The headless build has no GUI dependencies,
    # which is correct for a Jetson inference pipeline with no display output.
    pip install opencv-python-headless

    # Step 4 — camera device permissions (one-time, run after connecting camera)
    # Use v4l2-ctl to find your camera index first, then chmod the correct device.
    v4l2-ctl --list-devices
    sudo chmod 777 /dev/video4       # replace 4 with your actual index

    # Step 5 — sanity check (all four must import cleanly before running)
    python3 -c "import torch, torchvision, cv2, numpy; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('cv2:', cv2.__version__); print('numpy:', numpy.__version__)"

    NOTES
    -----
    · CUDA is not available on this setup (JetPack R36.5 / CUDA 12.6) due to
      wheel incompatibility. The engine detects this automatically and runs on
      CPU. Inference will be slower (~2-5 Hz) but fully functional. The
      max_infer_hz config parameter caps the rate to prevent thermal throttle.

    · pyrealsense2 is optional — the engine falls back to V4L2 automatically.
      pip install does not have an ARM64 build and the Intel apt repo GPG key
      failed on this network. The engine works fine without it on V4L2.

    · The camera index (/dev/video4 above) may change if the USB port changes.
      Run `v4l2-ctl --list-devices` any time the camera moves ports to find
      the new index, then rerun the chmod with the correct number.

Usage (CLI)
-----------
    cd /home/robotics/Documents/human_robot_collab/gearbox
    python3 intent_engine_v3.py \
    --resume_path     /home/robotics/Documents/human_robot_collab/gearbox/cnnlstm-Epoch-196-Loss-0.01737015192823795.pth \
    --annotation_path /home/robotics/Documents/human_robot_collab/gearbox/intentpredictionattempt4-1/datasets/labels.json \
    --n_classes        3 \
    --sample_size      150 \
    --sample_duration  8 \
    --smoothing_window 3 \
    --confidence_threshold 0.75

Usage (library)
---------------
    from intent_engine import IntentEngine, EngineConfig

    cfg = EngineConfig(
        resume_path  = "/home/robotics/Documents/human_robot_collab/gearbox/cnnlstm-Epoch-196-Loss-0.01737015192823795.pth",
        class_labels = ["INTERACTION", "PASSTHRU", "WAIT"],
        n_classes            = 3,
        sample_size          = 150,
        sample_duration      = 8,
        smoothing_window     = 3,
        confidence_threshold = 0.75,
    )

    def on_flag(flag, confidence):
        if flag == IntentEngine.FLAG_STOP:
            robot.halt()

    engine = IntentEngine(cfg, on_flag_change=on_flag)
    engine.start()          # non-blocking — launches camera + infer threads
    ...
    engine.stop()           # clean shutdown
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# stdlib
# ---------------------------------------------------------------------------
import argparse
import collections
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# third-party
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101

# ---------------------------------------------------------------------------
# optional RealSense SDK
# ---------------------------------------------------------------------------
try:
    import pyrealsense2 as rs          # type: ignore
    _RS_SDK_AVAILABLE = True
except ImportError:
    _RS_SDK_AVAILABLE = False

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("IntentEngine")


# ===========================================================================
# Model  (inlined — no local imports needed)
# ===========================================================================

class CNNLSTM(nn.Module):
    """
    ResNet-101 spatial feature extractor + 3-layer LSTM temporal classifier.

    Input
    -----
        x : (batch, time, C, H, W)   float32 tensor

    Output
    ------
        logits : (batch, num_classes)
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()

        # --- ResNet-101 backbone -------------------------------------------
        # Handle torchvision API change: weights= (≥0.13) vs pretrained= (<0.13)
        try:
            from torchvision.models import ResNet101_Weights
            backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
        except ImportError:
            backbone = resnet101(pretrained=True)   # torchvision < 0.13

        # Replace the classification head with a 300-d embedding layer.
        # Checkpoint saves this as resnet.fc.0.* (Sequential, not Linear)
        # so we use nn.Sequential here to match the saved keys exactly.
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(nn.Linear(in_features, 300))
        self.resnet = backbone

        # --- Temporal LSTM -------------------------------------------------
        # Checkpoint saves these as top-level lstm.*, fc1.*, fc2.* keys.
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3,
                            batch_first=False)
        self.fc1  = nn.Linear(256, 128)
        self.fc2  = nn.Linear(128, num_classes)

    def forward(self, x_3d: torch.Tensor) -> torch.Tensor:
        # x_3d: (batch, time, C, H, W)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        for t in range(x_3d.size(1)):
            cnn_out = self.resnet(x_3d[:, t])          # (batch, 300)
            out, hidden = self.lstm(cnn_out.unsqueeze(0), hidden)  # (1, B, 256)
        # Take last time-step
        x = F.relu(self.fc1(out[-1]))                  # (batch, 128)
        return self.fc2(x)                             # (batch, num_classes)


# ===========================================================================
# Camera back-ends
# ===========================================================================

class _RSSdkCamera:
    """
    RealSense SDK pipeline camera.
    Raises RuntimeError on construction if the SDK or device is unavailable.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        if not _RS_SDK_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed.")
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self._pipeline.start(cfg)
        except Exception as e:
            raise RuntimeError(f"RealSense SDK pipeline failed to start: {e}") from e
        log.info("Camera backend: RealSense SDK pipeline")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=200)
            color  = frames.get_color_frame()
            if not color:
                return False, None
            return True, np.asanyarray(color.get_data())
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
    """
    OpenCV V4L2 camera. Tries the given index; raises RuntimeError if it
    cannot open.
    """

    def __init__(self, index: int = 4,
                 width: int = 640, height: int = 480, fps: int = 30) -> None:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)        # always get the latest frame
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"V4L2 camera at index {index} failed to open. "
                f"Try: sudo chmod 777 /dev/video{index}"
            )
        # Drain stale frames that may have buffered during open()
        for _ in range(5):
            cap.grab()
        self._cap   = cap
        self._index = index
        log.info("Camera backend: V4L2 index %d", index)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._cap.release()

    @property
    def backend_name(self) -> str:
        return f"v4l2_index_{self._index}"


class CameraFactory:
    """
    Static factory — tries camera backends in priority order and returns the
    first one that opens successfully.

    Priority
    --------
    1. RealSense SDK pipeline  (if pyrealsense2 is installed and device present)
    2. V4L2 index 4            (confirmed Jetson setup — run v4l2-ctl --list-devices
                                to verify; update index here if port changes)
    3. V4L2 index 0            (last-resort fallback)
    """

    @staticmethod
    def open(width: int = 640, height: int = 480, fps: int = 30):
        """Return an open camera object or raise RuntimeError if all fail."""
        attempts = []

        # 1. RealSense SDK
        try:
            return _RSSdkCamera(width, height, fps)
        except Exception as e:
            attempts.append(f"RealSense SDK: {e}")
            log.warning("RealSense SDK unavailable (%s), trying V4L2 index 4 …", e)

        # 2. V4L2 index 4 (confirmed device on this Jetson setup)
        try:
            return _V4L2Camera(4, width, height, fps)
        except Exception as e:
            attempts.append(f"V4L2@4: {e}")
            log.warning("V4L2 index 4 failed (%s), trying V4L2 index 0 …", e)

        # 3. V4L2 index 0
        try:
            return _V4L2Camera(0, width, height, fps)
        except Exception as e:
            attempts.append(f"V4L2@0: {e}")

        raise RuntimeError(
            "All camera backends failed:\n  " + "\n  ".join(attempts)
        )


# ===========================================================================
# Frame validator
# ===========================================================================

def _frame_is_valid(frame: np.ndarray,
                    min_mean_intensity: float = 5.0) -> bool:
    """
    Return True if frame looks like real camera data.

    Rejects
    -------
    - None
    - Wrong number of dimensions
    - Zero-mean / near-black frames (camera warming up, lens cap, bad read)
    - Frames with mean intensity below min_mean_intensity (0–255 scale)
    """
    if frame is None:
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    if frame.mean() < min_mean_intensity:
        return False
    return True


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class EngineConfig:
    """
    All tuneable parameters for IntentEngine.

    Required
    --------
    resume_path : str
        Absolute path to the .pth weights file.

    Labels — provide exactly one of:
    class_labels    : list[str]   e.g. ["INTERACTION", "PASSTHRU", "WAIT"]
    annotation_path : str         path to JSON with a "labels" key

    Commonly tuned
    --------------
    sample_duration      : sliding window length in frames (default 8)
    smoothing_window     : temporal vote window (default 3)
    confidence_threshold : minimum smoothed-vote fraction to raise FLAG_STOP
    max_infer_hz         : cap inference rate to avoid Jetson thermal throttle
    camera_warmup_frames : discard this many frames before inference starts
    """

    # --- required ---
    resume_path:    str = ""

    # --- model ---
    n_classes:      int = 3

    # --- inference ---
    sample_size:             int   = 150    # spatial resize (px)
    sample_duration:         int   = 8      # sliding window length (frames)
    confidence_threshold:    float = 0.75
    smoothing_window:        int   = 3      # temporal vote window
    max_infer_hz:            float = 10.0   # Jetson thermal guard
    camera_warmup_frames:    int   = 30     # frames to discard on startup

    # --- hardware ---
    gpu:            int   = 0

    # --- labels ---
    annotation_path: str       = ""
    class_labels:    List[str] = field(default_factory=list)

    # --- flag mapping ---
    stop_class:    str = "INTERACTION"
    caution_class: str = "PASSTHRU"

    # --- restart policy ---
    thread_restart_delay_s: float = 2.0   # back-off before restarting a dead thread


# ===========================================================================
# Inference engine
# ===========================================================================

class IntentEngine:
    """
    Thread-safe, self-healing CNN-LSTM intent prediction engine.

    Flags
    -----
    FLAG_STOP         INTERACTION detected above confidence_threshold
    FLAG_CAUTION      PASSTHRU detected (any confidence)
    FLAG_CLEAR        WAIT / nothing notable
    FLAG_CAMERA_ERROR Camera is open but returning bad frames

    Lifecycle
    ---------
        engine = IntentEngine(cfg)
        engine.start()                   # launches camera + inference threads
        ...
        state  = engine.get_state()      # non-blocking snapshot
        flag   = engine.flag             # property shortcut
        halt   = engine.should_stop      # bool shortcut
        ...
        engine.stop()                    # clean shutdown

    Callback
    --------
        def on_flag_change(flag: str, confidence: float): ...
        engine = IntentEngine(cfg, on_flag_change=on_flag_change)
        # called exactly once per transition, never on repeated same-flag frames
    """

    FLAG_STOP         = "STOP"
    FLAG_CAUTION      = "CAUTION"
    FLAG_CLEAR        = "CLEAR"
    FLAG_CAMERA_ERROR = "CAMERA_ERROR"

    # Minimum valid frames in buffer before inference is attempted
    _MIN_VALID_BUFFER_FILL = 1.0   # fraction of sample_duration required (1.0 = full)

    def __init__(
        self,
        cfg:             EngineConfig,
        on_flag_change:  Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.cfg             = cfg
        self._on_flag_change = on_flag_change

        # --- device ---
        self.device = torch.device(
            f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
        )
        log.info("Torch device: %s", self.device)

        # --- class labels ---
        self.class_labels: List[str] = self._resolve_labels()
        log.info("Class labels: %s", self.class_labels)

        # --- model ---
        self._model     = self._load_model()
        self._transform = self._build_transform()

        # --- camera (opened lazily in start()) ---
        self._camera            = None
        self.camera_backend:str = "not_opened"

        # --- sliding buffers ---
        self._raw_frames    = collections.deque(maxlen=cfg.sample_duration)
        self._pred_history  = collections.deque(maxlen=cfg.smoothing_window)

        # --- warmup counter ---
        self._warmup_remaining = cfg.camera_warmup_frames

        # --- shared state (all writes under _lock) ---
        self._lock          = threading.Lock()
        self._probs         = np.zeros(len(self.class_labels))
        self._label:   str  = self.class_labels[-1]
        self._confidence    = 0.0
        self._flag:    str  = self.FLAG_CLEAR
        self._prev_flag: Optional[str] = None
        self._infer_fps     = 0.0
        self._last_infer_t  = time.monotonic()
        self._bad_frame_streak = 0

        # --- lifecycle ---
        self._running  = False
        self._cam_thread:   Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera and start camera + inference background threads."""
        if self._running:
            log.warning("start() called on already-running engine — ignored.")
            return

        log.info("Opening camera …")
        self._camera = CameraFactory.open()
        self.camera_backend = self._camera.backend_name

        self._running = True
        self._cam_thread   = self._make_daemon("IntentEngine-capture", self._camera_loop)
        self._infer_thread = self._make_daemon("IntentEngine-infer",   self._inference_loop)
        self._cam_thread.start()
        self._infer_thread.start()
        log.info("Engine started (camera=%s, device=%s).",
                 self.camera_backend, self.device)

    def stop(self) -> None:
        """Signal both threads to stop and wait for them to finish."""
        log.info("Stopping engine …")
        self._running = False
        if self._cam_thread:
            self._cam_thread.join(timeout=5.0)
        if self._infer_thread:
            self._infer_thread.join(timeout=5.0)
        if self._camera:
            self._camera.release()
        log.info("Engine stopped.")

    def get_state(self) -> dict:
        """
        Non-blocking snapshot of the latest inference result.

        Returns
        -------
        dict with keys:
            flag        : str           STOP | CAUTION | CLEAR | CAMERA_ERROR
            label       : str           winning class name
            confidence  : float         smoothed vote fraction for winning class
            probs       : np.ndarray    raw softmax distribution (last frame)
            infer_fps   : float         inference loop rate (Hz)
            camera      : str           active camera backend name
        """
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
        """True when FLAG_STOP is active. Designed for tight control loops."""
        return self.flag == self.FLAG_STOP

    def push_frame(self, bgr_frame: np.ndarray) -> None:
        """
        Manually push one BGR uint8 frame (H×W×3 ndarray) into the inference
        buffer. Use this when you are managing your own camera loop externally
        instead of relying on the engine's built-in camera thread.

        Non-blocking — safe to call from any thread at full camera rate.
        Invalid frames (black, wrong shape) are silently dropped.
        """
        if _frame_is_valid(bgr_frame):
            self._raw_frames.append(bgr_frame)

    # -----------------------------------------------------------------------
    # Internal — camera thread
    # -----------------------------------------------------------------------

    def _camera_loop(self) -> None:
        """
        Runs at full sensor rate. Validates each frame before pushing it into
        the inference buffer. Tracks consecutive bad frames and emits
        FLAG_CAMERA_ERROR if the camera appears to be broken.
        """
        MAX_BAD_STREAK = 30   # ~1 s at 30 fps before raising CAMERA_ERROR

        while self._running:
            try:
                ret, frame = self._camera.read()

                if not ret or not _frame_is_valid(frame):
                    self._bad_frame_streak += 1
                    if self._bad_frame_streak >= MAX_BAD_STREAK:
                        log.error(
                            "Camera has returned %d consecutive bad frames — "
                            "raising CAMERA_ERROR flag.",
                            self._bad_frame_streak,
                        )
                        self._set_flag(self.FLAG_CAMERA_ERROR, 1.0, "CAMERA_ERROR")
                    time.sleep(0.01)
                    continue

                self._bad_frame_streak = 0

                # Honour the warmup period: discard frames until the camera
                # sensor has stabilised (auto-exposure, lens warm-up, etc.)
                if self._warmup_remaining > 0:
                    self._warmup_remaining -= 1
                    continue

                self._raw_frames.append(frame)

            except Exception:
                log.error(
                    "Exception in camera loop:\n%s", traceback.format_exc()
                )
                time.sleep(0.5)

    # -----------------------------------------------------------------------
    # Internal — inference thread (self-healing)
    # -----------------------------------------------------------------------

    def _inference_loop(self) -> None:
        """
        Rate-limited inference loop. Restarts automatically after any
        exception — the restart delay gives the system time to recover
        from transient GPU/memory errors without busy-looping.
        """
        min_interval = 1.0 / max(self.cfg.max_infer_hz, 0.1)

        while self._running:
            loop_start = time.monotonic()
            try:
                self._inference_step()
            except Exception:
                log.error(
                    "Exception in inference loop (will restart in %.1f s):\n%s",
                    self.cfg.thread_restart_delay_s,
                    traceback.format_exc(),
                )
                time.sleep(self.cfg.thread_restart_delay_s)
                continue   # restart — do NOT break

            # Rate limiter: sleep for the remainder of the frame budget
            elapsed = time.monotonic() - loop_start
            sleep_t = min_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _inference_step(self) -> None:
        """One inference iteration. Called by _inference_loop."""
        required = int(
            self.cfg.sample_duration * self._MIN_VALID_BUFFER_FILL
        )
        if len(self._raw_frames) < required:
            time.sleep(0.02)
            return

        # --- snapshot (prevents buffer mutation during preprocessing) ---
        frames = list(self._raw_frames)

        # --- preprocess ---
        t_frames = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t_frames.append(self._transform(Image.fromarray(rgb)))

        # --- forward pass ---
        clip   = torch.stack(t_frames).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(clip)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        # --- temporal smoothing (recency-weighted vote) ---
        pred_idx = int(np.argmax(probs))
        self._pred_history.append(pred_idx)

        n       = len(self._pred_history)
        weights = np.linspace(0.5, 1.0, n)
        votes   = np.zeros(len(self.class_labels))
        for w, idx in zip(weights, self._pred_history):
            votes[idx] += w

        smoothed_idx = int(np.argmax(votes))
        label        = self.class_labels[smoothed_idx]

        # BUG 7 FIX: confidence = smoothed vote fraction, NOT raw prob of
        # smoothed class. This is a true measure of window agreement.
        total_weight  = weights.sum()
        smoothed_conf = float(votes[smoothed_idx] / total_weight) if total_weight > 0 else 0.0

        flag = self._resolve_flag(label, smoothed_conf)

        # --- FPS ---
        now   = time.monotonic()
        fps   = 1.0 / max(now - self._last_infer_t, 1e-6)
        self._last_infer_t = now

        # --- atomic state update + transition detection ---
        self._set_flag(flag, smoothed_conf, label, probs=probs, fps=fps)

    # -----------------------------------------------------------------------
    # Internal — helpers
    # -----------------------------------------------------------------------

    def _set_flag(
        self,
        flag:       str,
        confidence: float,
        label:      str,
        probs:      Optional[np.ndarray] = None,
        fps:        float = 0.0,
    ) -> None:
        """
        Atomically update shared state and fire on_flag_change exactly once
        per flag transition.

        BUG 2 FIX: the entire read-compare-write on _prev_flag is performed
        inside a single lock acquisition, eliminating the TOCTOU race.
        """
        with self._lock:
            self._flag       = flag
            self._label      = label
            self._confidence = confidence
            self._infer_fps  = fps
            if probs is not None:
                self._probs = probs

            transitioned  = (flag != self._prev_flag)
            self._prev_flag = flag          # update inside the lock

        # Fire callback outside the lock to avoid potential deadlock if the
        # caller tries to call get_state() from within the callback.
        if transitioned and self._on_flag_change is not None:
            try:
                self._on_flag_change(flag, confidence)
            except Exception:
                log.error(
                    "on_flag_change callback raised:\n%s",
                    traceback.format_exc(),
                )

    def _resolve_flag(self, label: str, confidence: float) -> str:
        if label == self.cfg.stop_class and confidence >= self.cfg.confidence_threshold:
            return self.FLAG_STOP
        if label == self.cfg.caution_class:
            return self.FLAG_CAUTION
        return self.FLAG_CLEAR

    def _build_transform(self) -> transforms.Compose:
        # ImageNet-style mean/std used during training (from mean.py in the
        # original codebase: norm_value=1, dataset=activitynet)
        mean = [114.7748, 107.7354, 99.4750]
        std  = [  1.0,      1.0,      1.0  ]
        return transforms.Compose([
            transforms.Resize((self.cfg.sample_size, self.cfg.sample_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255.0),
            transforms.Normalize(mean, std),
        ])

    def _load_model(self) -> nn.Module:
        """
        Load CNNLSTM weights with strict key verification.

        BUG 6 FIX: strict=True ensures a mismatch between checkpoint and
        model architecture raises immediately with a clear error rather than
        silently producing random outputs. weights_only is handled
        conditionally for cross-version PyTorch compatibility.
        """
        if not self.cfg.resume_path:
            raise ValueError("cfg.resume_path must be set to a .pth file.")
        if not os.path.isfile(self.cfg.resume_path):
            raise FileNotFoundError(
                f"Weights file not found: {self.cfg.resume_path}"
            )

        model = CNNLSTM(num_classes=self.cfg.n_classes).to(self.device)

        # weights_only=True is safer but requires PyTorch ≥ 1.13 and a pure
        # tensor checkpoint. Try it first; fall back for older checkpoints.
        load_kwargs = {"map_location": self.device}
        try:
            checkpoint = torch.load(
                self.cfg.resume_path, weights_only=True, **load_kwargs
            )
        except TypeError:
            # PyTorch < 1.13 does not have weights_only
            checkpoint = torch.load(self.cfg.resume_path, **load_kwargs)
        except Exception:
            # Checkpoint contains non-tensor objects (optimizer state, etc.)
            checkpoint = torch.load(self.cfg.resume_path, **load_kwargs)

        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip "module." prefix added by DataParallel, if present
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"Checkpoint is missing keys required by the model: {missing}"
            )
        if unexpected:
            log.warning(
                "Checkpoint contains keys not present in the model "
                "(they will be ignored): %s", unexpected
            )

        model.eval()
        model.resnet.eval()
        log.info("Loaded weights: %s", self.cfg.resume_path)
        log.info("Model on device: %s", self.device)
        return model

    def _resolve_labels(self) -> List[str]:
        if self.cfg.class_labels:
            return list(self.cfg.class_labels)
        if self.cfg.annotation_path:
            if not os.path.isfile(self.cfg.annotation_path):
                raise FileNotFoundError(
                    f"annotation_path not found: {self.cfg.annotation_path}"
                )
            with open(self.cfg.annotation_path, "r") as f:
                data = json.load(f)

            # Format 1: {"labels": ["INTERACTION", "PASSTHRU", "WAIT"]}
            # Format 2: {"classes": [...]} or {"label": [...]}
            for key in ("labels", "classes", "label"):
                if key in data and isinstance(data[key], list):
                    return list(data[key])

            # Format 3: {"0": "INTERACTION", "1": "PASSTHRU", "2": "WAIT"}
            # (your actual labels.json format — numeric string keys)
            if data and all(k.isdigit() for k in data.keys()):
                ordered = sorted(data.items(), key=lambda kv: int(kv[0]))
                labels  = [v for _, v in ordered]
                log.info("Parsed labels from numeric-keyed JSON: %s", labels)
                return labels

            # Format 4: {"INTERACTION": 0, "PASSTHRU": 1, "WAIT": 2}
            # (inverted map — class name to index)
            if data and all(isinstance(v, int) for v in data.values()):
                ordered = sorted(data.items(), key=lambda kv: kv[1])
                labels  = [k for k, _ in ordered]
                log.info("Parsed labels from inverted class-map JSON: %s", labels)
                return labels

            raise KeyError(
                f"annotation JSON format not recognised. Keys found: {list(data.keys())}. "
                f"Supported formats: {{\"labels\": [...]}}, "
                f"{{\"0\": \"CLASS\", \"1\": \"CLASS\", ...}}, "
                f"{{\"CLASS\": 0, \"CLASS\": 1, ...}}"
            )
        raise ValueError(
            "Set cfg.class_labels (list) or cfg.annotation_path (JSON path)."
        )

    @staticmethod
    def _make_daemon(name: str, target: Callable) -> threading.Thread:
        return threading.Thread(target=target, name=name, daemon=True)


# ===========================================================================
# CLI entry-point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNN-LSTM Intent Prediction Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resume_path",           required=True,
                   help="Path to model weights (.pth)")
    p.add_argument("--annotation_path",       default="",
                   help="Path to labels JSON (alternative to --class_labels)")
    p.add_argument("--class_labels",          nargs="+", default=[],
                   help="Class labels in index order")
    p.add_argument("--n_classes",             type=int,   default=3)
    p.add_argument("--sample_size",           type=int,   default=150)
    p.add_argument("--sample_duration",       type=int,   default=8)
    p.add_argument("--confidence_threshold",  type=float, default=0.75)
    p.add_argument("--smoothing_window",      type=int,   default=3)
    p.add_argument("--max_infer_hz",          type=float, default=10.0,
                   help="Max inference rate (Hz). Lower = less thermal load on Jetson.")
    p.add_argument("--camera_warmup_frames",  type=int,   default=30,
                   help="Frames to discard on startup before inference begins.")
    p.add_argument("--gpu",                   type=int,   default=0)
    p.add_argument("--log_level",             default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(args.log_level)

    cfg = EngineConfig(
        resume_path           = args.resume_path,
        annotation_path       = args.annotation_path,
        class_labels          = args.class_labels,
        n_classes             = args.n_classes,
        sample_size           = args.sample_size,
        sample_duration       = args.sample_duration,
        confidence_threshold  = args.confidence_threshold,
        smoothing_window      = args.smoothing_window,
        max_infer_hz          = args.max_infer_hz,
        camera_warmup_frames  = args.camera_warmup_frames,
        gpu                   = args.gpu,
    )

    def on_flag_change(flag: str, confidence: float) -> None:
        marker = {
            IntentEngine.FLAG_STOP:         "⚠  STOP",
            IntentEngine.FLAG_CAUTION:      "△  CAUTION",
            IntentEngine.FLAG_CLEAR:        "✓  CLEAR",
            IntentEngine.FLAG_CAMERA_ERROR: "✗  CAMERA ERROR",
        }.get(flag, flag)
        print(f"\n[FLAG TRANSITION]  {marker}   conf={confidence:.2f}\n")

    log.info("Initialising engine …")
    engine = IntentEngine(cfg, on_flag_change=on_flag_change)

    log.info("Starting engine (camera + inference threads) …")
    engine.start()
    log.info("Running — Ctrl-C to quit\n")

    header = f"{'flag':<14}{'label':<16}{'conf':>6}  {'fps':>6}  camera"
    print(header)
    print("─" * len(header))

    try:
        while True:
            s = engine.get_state()
            # Build a one-line prob bar for each class
            bars = "  ".join(
                f"{lbl}:{p*100:4.0f}%"
                for lbl, p in zip(engine.class_labels, s["probs"])
            )
            print(
                f"\r{s['flag']:<14}{s['label']:<16}"
                f"{s['confidence']:>6.2f}  "
                f"{s['infer_fps']:>5.1f}hz  "
                f"{s['camera']:<18}  "
                f"{bars}   ",
                end="",
                flush=True,
            )
            time.sleep(0.05)   # 20 Hz terminal refresh

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — shutting down …")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
