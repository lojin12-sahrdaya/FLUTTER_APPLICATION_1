"""Compatibility wrapper for legacy imports.

Historically this project used a single large file (sign_language_mouse.py) that
contained BOTH:
- Sign language translator (ASL letters -> text)
- Sign language mouse controller

To reduce clutter and keep responsibilities separate, the implementation is now
split into:
- sign_language_translator.py

Mouse control remains in this file to keep a single entrypoint for the server.

This file remains as a thin wrapper to preserve the public API expected by
main_server.py.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import mediapipe as mp
import pyautogui
import time


def start_sign_language_translator() -> None:
    from sign_language_translator import start_sign_language_translator as _start

    _start()


def get_translator_snapshot():
    from sign_language_translator import get_translator_snapshot as _get

    return _get()


def start_simple_mouse_control() -> None:
    """Model-driven ASL mouse controller.

    This is intentionally different from the generic hand-gesture module:
    - Mouse ACTIONS are triggered by ASL LETTERS (from your trained model).
    - Cursor movement uses landmarks only when you enable MOVE mode.

    Default letter -> action mapping (tunable later):
    - A: toggle MOVE mode (cursor follows index fingertip)
    - L: left click
    - R: right click
    - D: double click
    - G: drag toggle (mouse down/up)
    - U: scroll up
    - O: scroll down

    Press ESC in the camera window to exit.
    """

    # Import translator internals lazily (loads TensorFlow only when needed).
    from sign_language_translator import LandmarkPreprocessor, KerasH5ModelBackend

    def _load_backend() -> Optional[KerasH5ModelBackend]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(base_dir, "models", "sign_model.h5")
        model_path = os.environ.get("SIGN_MODEL_PATH", default_path)
        if not os.path.exists(model_path):
            print(f"❌ No trained sign model found at: {model_path}")
            return None

        labels = []
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
            labels = [f"asl_{chr(ord('a') + i)}" for i in range(26)]

        try:
            backend = KerasH5ModelBackend(model_path=model_path, labels=labels)
            print(f"✅ Loaded sign model for mouse control: {model_path} (sequence_len={getattr(backend, 'sequence_len', 'n/a')})")
            return backend
        except Exception as e:
            print(f"❌ Failed to load sign model for mouse control: {e}")
            return None

    backend = _load_backend()
    if backend is None:
        print("Mouse control requires the trained model. Aborting.")
        return

    preprocessor = LandmarkPreprocessor()
    sequence_len = int(getattr(backend, "sequence_len", 1))
    seq = []  # list of np arrays, length <= sequence_len

    # Smoothing for letters
    window_size = int(os.environ.get("SL_ASL_WINDOW", "7"))
    min_votes = int(os.environ.get("SL_ASL_MIN_VOTES", "4"))
    min_conf = float(os.environ.get("SL_ASL_MIN_CONF", "0.65"))
    label_window = []
    conf_window = []

    action_cooldown = float(os.environ.get("SL_ASL_ACTION_COOLDOWN", "0.7"))
    last_action_time = 0.0
    last_fired_label: Optional[str] = None

    move_enabled = False
    is_dragging = False

    screen_width, screen_height = pyautogui.size()
    pyautogui.PAUSE = 0
    smoothing_factor = float(os.environ.get("SL_MOUSE_SMOOTHING", "6"))
    prev_x, prev_y = 0.0, 0.0

    # Map stable ASL letter -> action
    action_map = {
        "asl_a": "toggle_move",
        "asl_l": "left_click",
        "asl_r": "right_click",
        "asl_d": "double_click",
        "asl_g": "toggle_drag",
        "asl_u": "scroll_up",
        "asl_o": "scroll_down",
    }

    # Actions that should only fire once per held sign (require the sign to change).
    edge_trigger_actions = {
        "toggle_move",
        "left_click",
        "right_click",
        "double_click",
        "toggle_drag",
    }

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Model inference (A-Z)
            import numpy as np

            features = preprocessor.landmarks_to_features(hand_landmarks)
            seq.append(features)
            if len(seq) > sequence_len:
                seq = seq[-sequence_len:]

            raw_label = "none"
            raw_conf = 0.0
            if len(seq) == sequence_len:
                raw_label, raw_conf = backend.predict(np.stack(seq, axis=0))

            # Update smoothing window
            label_window.append(raw_label)
            conf_window.append(float(raw_conf))
            if len(label_window) > window_size:
                label_window = label_window[-window_size:]
                conf_window = conf_window[-window_size:]

            stable_label = None
            stable_conf = 0.0
            if len(label_window) == window_size:
                # majority vote
                counts = {}
                for l in label_window:
                    counts[l] = counts.get(l, 0) + 1
                best_label = max(counts.items(), key=lambda kv: kv[1])[0]
                best_votes = counts[best_label]
                if best_label != "none" and best_votes >= min_votes:
                    confs = [c for (l, c) in zip(label_window, conf_window) if l == best_label]
                    stable_conf = float(sum(confs) / max(len(confs), 1))
                    if stable_conf >= min_conf:
                        stable_label = best_label

            now = time.time()
            action_text = "MOVE" if move_enabled else "IDLE"

            # Apply action from stable label
            if stable_label is not None:
                action = action_map.get(stable_label)
                should_fire = True
                if action in edge_trigger_actions and last_fired_label == stable_label:
                    should_fire = False

                if action and should_fire and (now - last_action_time) >= action_cooldown:
                    try:
                        if action == "toggle_move":
                            move_enabled = not move_enabled
                            action_text = f"MOVE={'ON' if move_enabled else 'OFF'}"
                        elif action == "left_click":
                            pyautogui.click(button="left")
                            action_text = "LEFT_CLICK"
                        elif action == "right_click":
                            pyautogui.click(button="right")
                            action_text = "RIGHT_CLICK"
                        elif action == "double_click":
                            pyautogui.doubleClick(button="left")
                            action_text = "DOUBLE_CLICK"
                        elif action == "toggle_drag":
                            if not is_dragging:
                                pyautogui.mouseDown(button="left")
                                is_dragging = True
                                action_text = "DRAG_START"
                            else:
                                pyautogui.mouseUp(button="left")
                                is_dragging = False
                                action_text = "DRAG_END"
                        elif action == "scroll_up":
                            pyautogui.scroll(240)
                            action_text = "SCROLL_UP"
                        elif action == "scroll_down":
                            pyautogui.scroll(-240)
                            action_text = "SCROLL_DOWN"
                        last_action_time = now
                        last_fired_label = stable_label
                    except Exception as e:
                        cv2.putText(
                            image,
                            f"Mouse action error: {e}",
                            (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

            # Move cursor only when enabled
            if move_enabled:
                index_tip = hand_landmarks.landmark[8]
                target_x = float(screen_width) * float(index_tip.x)
                target_y = float(screen_height) * float(index_tip.y)
                curr_x = prev_x + (target_x - prev_x) / smoothing_factor
                curr_y = prev_y + (target_y - prev_y) / smoothing_factor
                try:
                    pyautogui.moveTo(curr_x, curr_y)
                except Exception:
                    pass
                prev_x, prev_y = curr_x, curr_y

            cv2.putText(
                image,
                "Sign Language Mouse Control - ESC to exit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"ACTION: {action_text} | MOVE={'ON' if move_enabled else 'OFF'} | DRAG={'ON' if is_dragging else 'OFF'}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                image,
                "ASL: A=MoveToggle L=Left R=Right D=Double G=Drag U=ScrollUp O=ScrollDown",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                image,
                f"RAW: {raw_label.replace('asl_', '').upper() if raw_label.startswith('asl_') else raw_label} ({raw_conf:.2f})",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
            if stable_label is not None:
                cv2.putText(
                    image,
                    f"STABLE: {stable_label.replace('asl_', '').upper()} ({stable_conf:.2f})",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                )
        else:
            cv2.putText(
                image,
                "No hand detected - show your hand",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Sign Language Mouse Control", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()


class SignLanguageMouseController:
    """Legacy controller used by main_server.py.

    main_server.py creates this controller, sets `mode = "translator"`,
    runs it on a background thread, and later calls `stop()`.

    We keep the same surface area but delegate to sign_language_translator.
    """

    def __init__(self, mode: str = "translator"):
        self.mode = mode
        self._controller = None

    def run(self) -> None:
        if self.mode != "translator":
            # Not used by current server routes, but keep predictable behavior.
            start_simple_mouse_control()
            return

        from sign_language_translator import OptimizedSignLanguageController

        self._controller = OptimizedSignLanguageController()
        # Replace the module-level controller used by polling endpoint.
        import sign_language_translator as _translator

        _translator.controller = self._controller
        self._controller.start()

    def stop(self) -> None:
        # Stop the running translator controller if present.
        try:
            if self._controller is not None:
                self._controller.stop()
                return
        except Exception:
            pass

        # Fallback: stop global translator controller if it exists.
        try:
            import sign_language_translator as _translator

            if getattr(_translator, "controller", None) is not None:
                _translator.controller.stop()  # type: ignore[attr-defined]
        except Exception:
            pass


__all__ = [
    "SignLanguageMouseController",
    "start_simple_mouse_control",
    "start_sign_language_translator",
    "get_translator_snapshot",
]
