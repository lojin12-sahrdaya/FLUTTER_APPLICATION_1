"""Sign language translator module.

This contains only the ASL translator pipeline + model loading.
It is intentionally separated from mouse-control code to keep responsibilities clear.

Public API (used by main_server.py via sign_language_mouse.py wrapper):
- OptimizedSignLanguageController
- start_sign_language_translator()
- get_translator_snapshot()
"""

from __future__ import annotations

import datetime
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# Simplified ASL label mapping used by the translator state machine.
# NOTE: Letters are produced by the trained model. SPACE/DELETE/CLEAR are control gestures.
ASL_ALPHABET: Dict[str, str] = {
    "asl_a": "A",
    "asl_b": "B",
    "asl_c": "C",
    "asl_d": "D",
    "asl_e": "E",
    "asl_f": "F",
    "asl_g": "G",
    "asl_h": "H",
    "asl_i": "I",
    "asl_j": "J",
    "asl_k": "K",
    "asl_l": "L",
    "asl_m": "M",
    "asl_n": "N",
    "asl_o": "O",
    "asl_p": "P",
    "asl_q": "Q",
    "asl_r": "R",
    "asl_s": "S",
    "asl_t": "T",
    "asl_u": "U",
    "asl_v": "V",
    "asl_w": "W",
    "asl_x": "X",
    "asl_y": "Y",
    "asl_z": "Z",
    "space": " ",
    "delete": "DELETE",
    "clear": "CLEAR",
}


@dataclass(frozen=True)
class FramePrediction:
    label: str
    confidence: float
    detection_status: str  # NO_HAND | HAND_DETECTED | ERROR


@dataclass
class TranslationState:
    current_word: str = ""
    full_sentence: str = ""
    last_action_time: float = 0.0
    last_clear_time: float = 0.0
    cooldown_seconds: float = 1.5
    clear_cooldown_seconds: float = 3.0

    # Hold-to-clear (safety)
    clear_hold_start: Optional[float] = None
    clear_hold_duration_seconds: float = 3.0
    is_holding_clear: bool = False


class HandLandmarkExtractor:
    """MediaPipe wrapper: image -> hand landmarks."""

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._mp_drawing = mp.solutions.drawing_utils

    @property
    def mp_hands(self):
        return self._mp_hands

    @property
    def mp_drawing(self):
        return self._mp_drawing

    def extract(self, image_bgr) -> Tuple[Optional[object], str]:
        """Returns (hand_landmarks, detection_status)."""
        try:
            rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb_image)
            if results.multi_hand_landmarks:
                return results.multi_hand_landmarks[0], "HAND_DETECTED"
            return None, "NO_HAND"
        except Exception as e:
            print(f"⚠️ Hand landmark extraction error: {e}")
            return None, "ERROR"


class LandmarkPreprocessor:
    """Converts MediaPipe landmarks into a normalized feature vector."""

    def __init__(self, *, unmirror_x: Optional[bool] = None):
        # When the camera feed is mirrored for UX (cv2.flip(image, 1)), the
        # landmark x-coordinates are mirrored too. Most models are trained on
        # non-mirrored datasets, so we optionally flip x back for inference.
        if unmirror_x is None:
            # Default assumes the camera image is mirrored (see OptimizedSignLanguageController.run_camera).
            self.unmirror_x = os.environ.get("SIGN_UNMIRROR_X", "1") == "1"
        else:
            self.unmirror_x = bool(unmirror_x)

    def landmarks_to_features(self, hand_landmarks) -> np.ndarray:
        # Shape: (21, 3) -> flatten to (63,)
        pts = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)

        # Normalize: translate wrist to origin and scale by max distance.
        pts = pts - pts[0:1]
        if self.unmirror_x:
            pts[:, 0] *= -1.0

        scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
        if scale > 0:
            pts = pts / scale
        return pts.flatten()


class SequenceBuffer:
    """Fixed-length FIFO buffer for temporal models."""

    def __init__(self, sequence_len: int = 16):
        self.sequence_len = int(sequence_len)
        self._buf: Deque[np.ndarray] = deque(maxlen=self.sequence_len)

    def reset(self) -> None:
        self._buf.clear()

    def append(self, features: np.ndarray) -> None:
        self._buf.append(features)

    def is_ready(self) -> bool:
        return len(self._buf) == self.sequence_len

    def to_array(self) -> np.ndarray:
        # Shape: (T, F)
        return np.stack(list(self._buf), axis=0)


class ModelBackend:
    """Interface for models (frame or temporal)."""

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        raise NotImplementedError


class KerasH5ModelBackend(ModelBackend):
    """TensorFlow/Keras backend that supports both frame and temporal models.

    - Frame model: input shape (None, 63) -> uses the most recent frame.
    - Temporal model: input shape (None, T, 63) -> uses the full sequence.

    `sequence_len` is used by the pipeline to know how many frames to buffer.
    """

    def __init__(self, model_path: str, labels: List[str], default_sequence_len: int = 16):
        self.model_path = model_path
        self.labels = list(labels)
        self.sequence_len = int(default_sequence_len)

        try:
            from tensorflow.keras.models import load_model  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "TensorFlow is not available. Install tensorflow-intel to use .h5 models."
            ) from e

        self._model = load_model(model_path)

        # Auto-detect whether this is a per-frame classifier or a temporal model.
        input_shape = getattr(self._model, "input_shape", None)
        self._expects_rank = len(input_shape) if isinstance(input_shape, tuple) else 3

        if self._expects_rank == 2 and isinstance(input_shape, tuple) and len(input_shape) == 2:
            # (None, 63)
            self.sequence_len = 1
        elif self._expects_rank == 3 and isinstance(input_shape, tuple) and len(input_shape) == 3:
            # (None, T, 63) where T can be None
            t = input_shape[1]
            if isinstance(t, int) and t > 0:
                self.sequence_len = int(t)

        # Auto-detect class count to align labels when none provided.
        output_shape = getattr(self._model, "output_shape", None)
        if isinstance(output_shape, tuple) and len(output_shape) >= 2 and isinstance(output_shape[-1], int):
            c = int(output_shape[-1])
            if c == 26 and (not self.labels or len(self.labels) != 26):
                self.labels = [
                    "asl_a",
                    "asl_b",
                    "asl_c",
                    "asl_d",
                    "asl_e",
                    "asl_f",
                    "asl_g",
                    "asl_h",
                    "asl_i",
                    "asl_j",
                    "asl_k",
                    "asl_l",
                    "asl_m",
                    "asl_n",
                    "asl_o",
                    "asl_p",
                    "asl_q",
                    "asl_r",
                    "asl_s",
                    "asl_t",
                    "asl_u",
                    "asl_v",
                    "asl_w",
                    "asl_x",
                    "asl_y",
                    "asl_z",
                ]

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        if sequence.ndim == 1:
            sequence = np.asarray(sequence, dtype=np.float32).reshape(1, -1)
        if sequence.ndim != 2:
            raise ValueError(f"Expected (T, F) sequence, got shape {sequence.shape}")

        if self._expects_rank == 2:
            vec = np.asarray(sequence[-1], dtype=np.float32).reshape(1, -1)  # (1, 63)
            probs = self._model.predict(vec, verbose=0)
        else:
            x = np.expand_dims(np.asarray(sequence, dtype=np.float32), axis=0)  # (1, T, F)
            probs = self._model.predict(x, verbose=0)

        probs = np.asarray(probs)
        if probs.ndim == 3:
            probs = probs[:, -1, :]
        probs = probs.reshape(-1)

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        if class_idx < 0 or class_idx >= len(self.labels):
            return "none", 0.0
        return self.labels[class_idx], confidence


class HeuristicFallbackModel(ModelBackend):
    """Rule-based control + coarse letter heuristics.

    This is used when the trained model is not available, and also to detect
    control gestures (space/delete/clear).
    """

    def predict(self, sequence: np.ndarray) -> Tuple[str, float]:
        return "none", 0.0

    def classify_from_landmarks(self, landmarks) -> Tuple[str, float]:
        """Frame-level heuristic classification. Returns (label, confidence)."""
        try:
            thumb_tip = landmarks.landmark[4]
            thumb_mcp = landmarks.landmark[2]
            index_tip = landmarks.landmark[8]
            index_pip = landmarks.landmark[6]
            index_mcp = landmarks.landmark[5]
            middle_tip = landmarks.landmark[12]
            middle_pip = landmarks.landmark[10]
            middle_mcp = landmarks.landmark[9]
            ring_tip = landmarks.landmark[16]
            ring_pip = landmarks.landmark[14]
            pinky_tip = landmarks.landmark[20]
            pinky_pip = landmarks.landmark[18]
            wrist = landmarks.landmark[0]

            def dist(a, b) -> float:
                dx = float(a.x - b.x)
                dy = float(a.y - b.y)
                return (dx * dx + dy * dy) ** 0.5

            # Scale reference: roughly forearm-to-palm size
            hand_scale = dist(wrist, middle_mcp)
            if hand_scale <= 1e-6:
                hand_scale = dist(wrist, index_mcp)
            if hand_scale <= 1e-6:
                hand_scale = 0.2

            def is_finger_extended(tip, pip) -> bool:
                # Works better than a fixed threshold across different camera distances.
                return float(tip.y) < float(pip.y) - (0.10 * hand_scale)

            def is_finger_folded(tip, pip) -> bool:
                return float(tip.y) > float(pip.y) - (0.02 * hand_scale)

            index_up = is_finger_extended(index_tip, index_pip)
            middle_up = is_finger_extended(middle_tip, middle_pip)
            ring_up = is_finger_extended(ring_tip, ring_pip)
            pinky_up = is_finger_extended(pinky_tip, pinky_pip)

            fingers_extended = [index_up, middle_up, ring_up, pinky_up]
            extended_count = sum(fingers_extended)

            # --- CONTROL GESTURES (only) ---
            # DELETE: pinch (thumb tip close to index tip) while other fingers folded.
            pinch = dist(thumb_tip, index_tip)
            others_folded = (
                is_finger_folded(middle_tip, middle_pip)
                and is_finger_folded(ring_tip, ring_pip)
                and is_finger_folded(pinky_tip, pinky_pip)
            )
            if pinch < (0.35 * hand_scale) and others_folded:
                return "delete", 0.92

            # SPACE: open palm (all 4 fingers extended) and fingers far from wrist.
            avg_tip_to_wrist = (dist(index_tip, wrist) + dist(middle_tip, wrist) + dist(ring_tip, wrist) + dist(pinky_tip, wrist)) / 4.0
            if extended_count == 4 and avg_tip_to_wrist > (1.55 * hand_scale):
                return "space", 0.90

            # CLEAR: fist (no fingers extended) and fingertips close to wrist (held for 3s by state machine).
            all_folded = (
                is_finger_folded(index_tip, index_pip)
                and is_finger_folded(middle_tip, middle_pip)
                and is_finger_folded(ring_tip, ring_pip)
                and is_finger_folded(pinky_tip, pinky_pip)
            )
            if extended_count == 0 and all_folded and avg_tip_to_wrist < (1.15 * hand_scale):
                return "clear", 0.95

            # Coarse letter heuristics (fallback only)
            return "none", 0.0

        except Exception as e:
            print(f"Gesture detection error: {e}")
            return "none", 0.0


class PredictionSmoother:
    """Majority vote + confidence filtering over a rolling window."""

    def __init__(self, window_size: int = 9, min_votes: int = 6, min_confidence: float = 0.65):
        self.window_size = int(window_size)
        self.min_votes = int(min_votes)
        self.min_confidence = float(min_confidence)
        self._labels: Deque[str] = deque(maxlen=self.window_size)
        self._confs: Deque[float] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._labels.clear()
        self._confs.clear()

    def update(self, pred: FramePrediction) -> Optional[Tuple[str, float]]:
        if pred.detection_status != "HAND_DETECTED":
            self.reset()
            return None

        if not pred.label or pred.label == "none":
            self._labels.append("none")
            self._confs.append(0.0)
            return None

        self._labels.append(pred.label)
        self._confs.append(float(pred.confidence))

        if len(self._labels) < self.window_size:
            return None

        label_counts = Counter(self._labels)
        top_label, top_count = label_counts.most_common(1)[0]
        if top_label == "none" or top_count < self.min_votes:
            return None

        confs = [c for (l, c) in zip(self._labels, self._confs) if l == top_label]
        avg_conf = float(np.mean(confs)) if confs else 0.0
        if avg_conf < self.min_confidence:
            return None

        return top_label, avg_conf


class TextBuilder:
    """Applies stable gestures into word/sentence state."""

    def __init__(self, state: TranslationState):
        self.state = state
        self.current_gesture_display: str = "Detecting..."

    def apply_stable_label(self, stable_label: str, confidence: float, now: float) -> bool:
        self.current_gesture_display = stable_label.replace("asl_", "").upper() if stable_label else "Detecting..."

        if stable_label not in ASL_ALPHABET:
            if self.state.is_holding_clear:
                self.state.is_holding_clear = False
                self.state.clear_hold_start = None
            return False

        translation = ASL_ALPHABET[stable_label]

        # CLEAR: hold-to-clear safety
        if translation == "CLEAR":
            if not self.state.is_holding_clear:
                self.state.is_holding_clear = True
                self.state.clear_hold_start = now
                return False

            hold_time = now - (self.state.clear_hold_start or now)
            if hold_time < self.state.clear_hold_duration_seconds:
                return False

            if now - self.state.last_clear_time < self.state.clear_cooldown_seconds:
                return False

            self.state.current_word = ""
            self.state.full_sentence = ""
            self.state.last_clear_time = now
            self.state.last_action_time = now
            self.state.is_holding_clear = False
            self.state.clear_hold_start = None
            return True

        # Any non-clear label cancels clear hold
        if self.state.is_holding_clear:
            self.state.is_holding_clear = False
            self.state.clear_hold_start = None

        # Cooldown for normal actions
        if now - self.state.last_action_time < self.state.cooldown_seconds:
            return False

        changed = False
        if translation == "DELETE":
            if self.state.current_word:
                self.state.current_word = self.state.current_word[:-1]
                changed = True
        elif translation == " ":
            if self.state.current_word:
                self.state.full_sentence += self.state.current_word + " "
                self.state.current_word = ""
                changed = True
            else:
                self.state.full_sentence += " "
                changed = True
        else:
            self.state.current_word += translation
            changed = True

        if changed:
            self.state.last_action_time = now
        return changed


class SignTranslatorPipeline:
    """Orchestrates preprocessing, inference, smoothing, and text building."""

    def __init__(
        self,
        extractor: HandLandmarkExtractor,
        preprocessor: LandmarkPreprocessor,
        temporal_model: Optional[ModelBackend],
        fallback_model: HeuristicFallbackModel,
        smoother: PredictionSmoother,
        state: TranslationState,
        sequence_len: int = 16,
    ):
        self.extractor = extractor
        self.preprocessor = preprocessor
        self.temporal_model = temporal_model
        self.fallback_model = fallback_model
        self.smoother = smoother
        self.state = state
        self.text_builder = TextBuilder(state)
        self.seq = SequenceBuffer(sequence_len=sequence_len)

        self.last_stable_label: Optional[str] = None
        self.last_stable_confidence: float = 0.0
        self.detection_status: str = "NO_HAND"

        # Debug/telemetry
        self.last_raw_label: Optional[str] = None
        self.last_raw_confidence: float = 0.0
        self.last_inference_source: str = "none"  # control | model | fallback | none

        # Debounce controls so they respond faster than letter smoothing.
        self._control_streak_label: Optional[str] = None
        self._control_streak_count: int = 0

    def process_frame(self, image_bgr) -> Tuple[Optional[object], str]:
        now = time.time()
        hand_landmarks, status = self.extractor.extract(image_bgr)
        self.detection_status = status

        if hand_landmarks is None or status != "HAND_DETECTED":
            self.seq.reset()
            self.smoother.reset()
            self.last_raw_label = None
            self.last_raw_confidence = 0.0
            self.last_inference_source = "none"
            if self.state.is_holding_clear:
                self.state.is_holding_clear = False
                self.state.clear_hold_start = None
            return None, "No Hand" if status == "NO_HAND" else "Error"

        features = self.preprocessor.landmarks_to_features(hand_landmarks)
        self.seq.append(features)

        model_loaded = self.temporal_model is not None
        allow_fallback_letters = os.environ.get("SIGN_ALLOW_FALLBACK_LETTERS", "0") == "1"
        model_min_conf = float(os.environ.get("SIGN_MODEL_MIN_CONF", "0.55"))

        # Model prediction (letters only)
        model_label: str = "none"
        model_conf: float = 0.0
        if model_loaded and self.seq.is_ready():
            try:
                model_label, model_conf = self.temporal_model.predict(self.seq.to_array())
            except Exception as e:
                print(f"⚠️ Model inference failed; will fall back. Error: {e}")
                model_label, model_conf = "none", 0.0

        model_label_is_letter = isinstance(model_label, str) and model_label.startswith("asl_")
        model_is_accepted_letter = model_label_is_letter and float(model_conf) >= model_min_conf

        # Control prediction (heuristics)
        control_label, control_conf = self.fallback_model.classify_from_landmarks(hand_landmarks)
        is_control = control_label in ("space", "delete", "clear") and float(control_conf) >= 0.85

        # If the model is very confident it's a letter, do NOT let delete/clear override it.
        veto_threshold = float(os.environ.get("SIGN_CONTROL_MODEL_VETO_CONF", "0.85"))
        model_is_confident_letter = model_is_accepted_letter and float(model_conf) >= veto_threshold

        if is_control:
            if control_label in ("delete", "clear") and model_is_confident_letter:
                label, conf = model_label, float(model_conf)
                self.last_inference_source = "model"
            else:
                label, conf = control_label, float(control_conf)
                self.last_inference_source = "control"
        else:
            if model_is_accepted_letter:
                label, conf = model_label, float(model_conf)
                self.last_inference_source = "model"
            else:
                if model_loaded and not allow_fallback_letters:
                    # Strict mode: when the trained model is available, do not
                    # emit heuristic A-Z letters. Only the model can emit letters.
                    label, conf = "none", 0.0
                    self.last_inference_source = "none"
                else:
                    # No model available (or explicitly allowed): fallback can emit letters.
                    label, conf = self.fallback_model.classify_from_landmarks(hand_landmarks)
                    self.last_inference_source = "fallback"

        self.last_raw_label = label
        self.last_raw_confidence = float(conf)

        # Fast-path control gestures using a short debounce streak.
        control_threshold_frames = int(os.environ.get("SIGN_CONTROL_STREAK_FRAMES", "3"))
        if label in ("space", "delete", "clear") and float(conf) >= 0.85:
            if self._control_streak_label == label:
                self._control_streak_count += 1
            else:
                self._control_streak_label = label
                self._control_streak_count = 1

            # Keep letter smoother from mixing with controls.
            self.smoother.reset()

            if self._control_streak_count >= control_threshold_frames:
                self.last_stable_label = label
                self.last_stable_confidence = float(conf)
                self.text_builder.apply_stable_label(label, float(conf), now)

            return hand_landmarks, self.text_builder.current_gesture_display
        else:
            self._control_streak_label = None
            self._control_streak_count = 0

        frame_pred = FramePrediction(label=label, confidence=float(conf), detection_status=status)

        if frame_pred.label in ASL_ALPHABET and frame_pred.confidence >= 0.50:
            if frame_pred.label.startswith("asl_"):
                self.text_builder.current_gesture_display = frame_pred.label.replace("asl_", "").upper()
            else:
                self.text_builder.current_gesture_display = frame_pred.label.upper()

        stable = self.smoother.update(frame_pred)
        if stable is not None:
            stable_label, stable_conf = stable
            self.last_stable_label = stable_label
            self.last_stable_confidence = stable_conf
            self.text_builder.apply_stable_label(stable_label, stable_conf, now)

        return hand_landmarks, self.text_builder.current_gesture_display


class OptimizedSignLanguageController:
    """Translator UI + camera loop."""

    def __init__(self):
        self.cap = None

        self._stop_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._snapshot = {
            "running": False,
            "detection_status": "NO_HAND",
            "gesture": "Detecting...",
            "current_word": "",
            "full_sentence": "",
            "live_text": "",
            "last_stable_label": None,
            "last_stable_confidence": 0.0,
            "raw_label": None,
            "raw_confidence": 0.0,
            "inference_source": "none",
            "updated_at": time.time(),
        }

        self.extractor = HandLandmarkExtractor()
        self.preprocessor = LandmarkPreprocessor()
        self.fallback_model = HeuristicFallbackModel()
        self.temporal_model = self._try_load_model()
        self.smoother = PredictionSmoother(window_size=7, min_votes=4, min_confidence=0.55)
        self.state = TranslationState(
            cooldown_seconds=1.5,
            clear_cooldown_seconds=3.0,
            clear_hold_duration_seconds=3.0,
        )

        sequence_len = 16
        if self.temporal_model is not None and hasattr(self.temporal_model, "sequence_len"):
            try:
                sequence_len = int(getattr(self.temporal_model, "sequence_len"))
            except Exception:
                sequence_len = 16

        self.pipeline = SignTranslatorPipeline(
            extractor=self.extractor,
            preprocessor=self.preprocessor,
            temporal_model=self.temporal_model,
            fallback_model=self.fallback_model,
            smoother=self.smoother,
            state=self.state,
            sequence_len=sequence_len,
        )

        self.current_gesture = "Detecting..."

        # Tk UI is optional; disabled when running from Flask thread.
        self._ui_enabled = False
        self.root = None
        self.text_area = None
        self._setup_ui()

    def _try_load_model(self) -> Optional[ModelBackend]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(base_dir, "models", "sign_model.h5")
        model_path = os.environ.get("SIGN_MODEL_PATH", default_path)

        if not os.path.exists(model_path):
            print(f"ℹ️ No trained sign model found at: {model_path} (using heuristic fallback)")
            return None

        # Labels: load from training metrics (preferred) to guarantee correct mapping.
        labels: List[str] = []
        try:
            metrics_path = os.path.join(os.path.dirname(model_path), "sign_model_metrics.json")
            if os.path.exists(metrics_path):
                import json

                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                label_map = metrics.get("label_map") or {}
                if isinstance(label_map, dict) and label_map:
                    inv = {int(v): str(k) for k, v in label_map.items() if isinstance(v, int) or str(v).isdigit()}
                    max_idx = max(inv.keys())
                    for i in range(max_idx + 1):
                        letter = inv.get(i)
                        if isinstance(letter, str) and len(letter) == 1 and letter.isalpha():
                            labels.append(f"asl_{letter.lower()}")
                        else:
                            labels.append("none")
        except Exception as e:
            print(f"ℹ️ Could not read sign model metrics; using default labels. Error: {e}")

        if not labels:
            labels = [
                "asl_a",
                "asl_b",
                "asl_c",
                "asl_d",
                "asl_e",
                "asl_f",
                "asl_g",
                "asl_h",
                "asl_i",
                "asl_j",
                "asl_k",
                "asl_l",
                "asl_m",
                "asl_n",
                "asl_o",
                "asl_p",
                "asl_q",
                "asl_r",
                "asl_s",
                "asl_t",
                "asl_u",
                "asl_v",
                "asl_w",
                "asl_x",
                "asl_y",
                "asl_z",
            ]

        try:
            backend = KerasH5ModelBackend(model_path=model_path, labels=labels)
            print(f"✅ Loaded sign model: {model_path} (sequence_len={getattr(backend, 'sequence_len', 'n/a')})")
            return backend
        except Exception as e:
            print(f"⚠️ Could not load sign model ({model_path}). Using fallback. Error: {e}")
            return None

    def get_snapshot(self) -> Dict[str, object]:
        with self._snapshot_lock:
            return dict(self._snapshot)

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        with self._snapshot_lock:
            self._snapshot["running"] = False
            self._snapshot["updated_at"] = time.time()

    def _setup_ui(self) -> None:
        # Avoid Tk when started by Flask (worker thread)
        if threading.current_thread() is not threading.main_thread():
            print("ℹ️ Translator started from a non-main thread: Tk UI disabled; using camera overlay only.")
            self._ui_enabled = False
            return

        try:
            import tkinter as tk
            from tkinter import scrolledtext
        except Exception as e:
            print(f"ℹ️ Tk UI not available: {e}")
            self._ui_enabled = False
            return

        self.root = tk.Tk()
        self.root.title("ASL Text Builder")
        self.root.geometry("500x300")

        self.text_area = scrolledtext.ScrolledText(self.root, height=10, font=("Arial", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)

        tk.Button(button_frame, text="Clear All", command=self._clear_text).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Text", command=self._save_text).pack(side=tk.LEFT, padx=5)

        self._ui_enabled = True

    def _clear_text(self) -> None:
        self.state.current_word = ""
        self.state.full_sentence = ""
        self.state.is_holding_clear = False
        self.state.clear_hold_start = None

    def _save_text(self) -> None:
        full_text = f"{self.state.full_sentence}{self.state.current_word}"
        if full_text.strip():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"asl_text_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(full_text.strip())
            print(f"Text saved to {filename}")

    def _update_ui(self) -> None:
        if not self._ui_enabled or self.text_area is None:
            return

        import tkinter as tk  # safe if UI enabled

        hold_status = ""
        if self.state.is_holding_clear and self.state.clear_hold_start:
            hold_time = time.time() - self.state.clear_hold_start
            remaining = max(self.state.clear_hold_duration_seconds - hold_time, 0)
            hold_status = f"\n🔥 HOLDING TO CLEAR: {remaining:.1f}s remaining..."

        display_str = f"""
Current Gesture: {self.current_gesture}
Building: {self.state.current_word}
Text: {self.state.full_sentence.strip()}

LIVE: {self.state.full_sentence}{self.state.current_word}{hold_status}

Commands:
- Show ASL letters A-Z to spell words
- OPEN PALM = SPACE (finish word)
- POINT with INDEX+THUMB = DELETE last letter
- Hold tight FIST for 3 seconds = CLEAR all text (safe)
        """

        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(1.0, display_str)

    def _draw_camera_text(self, image) -> None:
        height, width = image.shape[:2]

        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        y_pos = 35
        texts = [
            f"STATUS: {self.pipeline.detection_status}",
            f"GESTURE: {self.current_gesture}",
            f"WORD: {self.state.current_word}",
            f"SENTENCE: {self.state.full_sentence.strip()[:40]}",
            f"LIVE: {(self.state.full_sentence + self.state.current_word)[:35]}",
        ]
        colors = [(200, 200, 200), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

        for i, text in enumerate(texts):
            cv2.putText(image, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 2)
            y_pos += 24

        if self.state.is_holding_clear and self.state.clear_hold_start:
            now = time.time()
            hold_time = now - self.state.clear_hold_start
            progress = min(hold_time / self.state.clear_hold_duration_seconds, 1.0)
            remaining = max(self.state.clear_hold_duration_seconds - hold_time, 0)

            bar_width = 240
            bar_height = 14
            bar_x = 20
            bar_y = y_pos + 6
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            fill_width = int(bar_width * progress)
            color = (0, 0, 255) if progress < 1.0 else (0, 255, 0)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            cv2.putText(
                image,
                f"HOLD TO CLEAR: {remaining:.1f}s",
                (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.putText(
            image,
            "FIST 3s=CLEAR | OPEN PALM=SPACE | POINT+THUMB=DELETE",
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            image,
            "Show ASL letters A-Z to spell words",
            (20, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

    def run_camera(self) -> None:
        self.cap = cv2.VideoCapture(0)

        with self._snapshot_lock:
            self._snapshot["running"] = True
            self._snapshot["updated_at"] = time.time()

        headless = os.environ.get("SIGN_TRANSLATOR_HEADLESS", "0") == "1"
        flip_camera = os.environ.get("SIGN_FLIP_CAMERA", "1") == "1"

        while self.cap.isOpened() and not self._stop_event.is_set():
            success, image = self.cap.read()
            if not success:
                continue

            if flip_camera:
                image = cv2.flip(image, 1)

            hand_landmarks, gesture_display = self.pipeline.process_frame(image)
            self.current_gesture = gesture_display

            if hand_landmarks is not None:
                self.extractor.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.extractor.mp_hands.HAND_CONNECTIONS,
                )

            self._update_ui()
            self._draw_camera_text(image)

            with self._snapshot_lock:
                self._snapshot.update(
                    {
                        "running": True,
                        "detection_status": self.pipeline.detection_status,
                        "gesture": self.current_gesture,
                        "current_word": self.state.current_word,
                        "full_sentence": self.state.full_sentence,
                        "live_text": (self.state.full_sentence + self.state.current_word),
                        "last_stable_label": self.pipeline.last_stable_label,
                        "last_stable_confidence": float(self.pipeline.last_stable_confidence),
                        "raw_label": self.pipeline.last_raw_label,
                        "raw_confidence": float(self.pipeline.last_raw_confidence),
                        "inference_source": getattr(self.pipeline, "last_inference_source", "none"),
                        "updated_at": time.time(),
                    }
                )

            if not headless:
                cv2.imshow("ASL Sign Language Translator", image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        try:
            self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        with self._snapshot_lock:
            self._snapshot["running"] = False
            self._snapshot["updated_at"] = time.time()

    def start(self) -> None:
        camera_thread = threading.Thread(target=self.run_camera, daemon=True)
        camera_thread.start()

        if self._ui_enabled and self.root is not None:
            self.root.mainloop()
        else:
            camera_thread.join()


# Global controller instance (used by polling endpoint)
controller: Optional[OptimizedSignLanguageController] = None


def get_translator_snapshot() -> Dict[str, object]:
    """Return latest translator state for the webapp (safe, non-throwing)."""
    global controller
    if controller is None:
        return {"running": False}
    try:
        return controller.get_snapshot()
    except Exception as e:
        return {"running": False, "error": str(e)}


def start_sign_language_translator() -> None:
    """Start the sign language translator (blocking in the calling thread)."""
    global controller
    controller = OptimizedSignLanguageController()
    controller.start()
