import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import speech_recognition as sr
import threading
import time
import platform
import os
import screen_brightness_control as sbc


class EyeController:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.active = False
        self.voice_command_active = False
        self.voice_thread = None
        # Noise cancellation settings
        self.noise_sensitivity = "medium"  # low, medium, high
        self.energy_thresholds = {
            "low": 200,     # More sensitive, picks up more sounds
            "medium": 400,  # Balanced
            "high": 600     # Less sensitive, filters more noise
        }

    def set_noise_sensitivity(self, level):
        """Set noise sensitivity level: 'low', 'medium', or 'high'"""
        if level in self.energy_thresholds:
            self.noise_sensitivity = level
            print(f"🔊 Noise sensitivity set to: {level}")
        else:
            print("❌ Invalid sensitivity level. Use 'low', 'medium', or 'high'")

    def stop(self):
        if self.active:
            print("🔴 Stopping Eye Controller...")
            self.active = False
            self.voice_command_active = False
            # Give threads a moment to clean up
            time.sleep(0.5)
            print("✅ Eye Controller stopped successfully")

    def change_volume(self, up=True):
        system = platform.system()
        if system == "Windows":
            pyautogui.press("volumeup" if up else "volumedown")
        elif system == "Darwin":
            op = "+" if up else "-"
            os.system(f"osascript -e 'set volume output volume ((output volume of (get volume settings)) {op} 10)'")
        else:  # Assume Linux
            op = "+" if up else "-"
            os.system(f"amixer -D pulse sset Master 10%{op}")

    def change_brightness(self, up=True):
        try:
            current_brightness = sbc.get_brightness()
            # sbc.get_brightness() returns a list of values (one for each display)
            if isinstance(current_brightness, list):
                if len(current_brightness) > 0:
                    current_brightness = current_brightness[0]
                else:
                    print("⚠️ No displays detected for brightness control.")
                    return
            
            # Ensure it's an integer
            try:
                current_brightness = int(current_brightness)
            except (ValueError, TypeError):
                print(f"⚠️ Unexpected brightness value format: {current_brightness}")
                return

            step = 10
            new_brightness = min(100, current_brightness + step) if up else max(0, current_brightness - step)
            sbc.set_brightness(new_brightness)
            print(f"🔆 Brightness changed to {new_brightness}%")
        except Exception as e:
            print(f"⚠️ Error changing brightness: {e}")

    def _listen_for_commands(self):
        r = sr.Recognizer()
        pyautogui.PAUSE = 0

        with sr.Microphone() as source:
            print("🎤 Calibrating for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=2)
            
            # Enhanced noise cancellation settings with dynamic threshold
            r.energy_threshold = self.energy_thresholds[self.noise_sensitivity]
            r.dynamic_energy_threshold = True  # Automatically adjust energy threshold
            r.dynamic_energy_adjustment_damping = 0.15
            r.dynamic_energy_ratio = 1.5
            r.pause_threshold = 0.8  # Shorter pause threshold for quicker response
            r.operation_timeout = None  # No timeout for better noise handling
            
            print(f"✅ Voice commands with noise cancellation are active for Eye Control!")
            print(f"🔊 Noise sensitivity: {self.noise_sensitivity}")
            print("📢 Available commands: left click, right click, double click, scroll up, scroll down, drag start, drag end, VU/VD (volume), BU/BD (brightness)")
            print("🎚️ Say 'noise low/medium/high' to adjust sensitivity")

            while self.voice_command_active:
                try:
                    # Use longer timeout and phrase limit for better command capture
                    audio = r.listen(source, timeout=3, phrase_time_limit=3)
                    print("...Processing command...")
                    command = r.recognize_google(audio, language='en-US').lower()
                    print(f"✅ Command: '{command}'")

                    if "left click" in command:
                        pyautogui.click(button='left')
                        print("🖱️ Left click executed")
                    elif "right click" in command:
                        pyautogui.click(button='right')
                        print("🖱️ Right click executed")
                    elif "double click" in command:
                        pyautogui.doubleClick()
                        print("🖱️ Double click executed")
                    elif "scroll up" in command:
                        pyautogui.scroll(200)
                        print("📜 Scroll up executed")
                    elif "scroll down" in command:
                        pyautogui.scroll(-200)
                        print("📜 Scroll down executed")
                    elif "drag start" in command:
                        pyautogui.mouseDown()
                        print("🫳 Drag start executed")
                    elif "drag end" in command:
                        pyautogui.mouseUp()
                        print("🫴 Drag end executed")
                    elif "vu" in command or "v u" in command or "volume up" in command:
                        self.change_volume(up=True)
                        print("🔊 Volume up executed")
                    elif "vd" in command or "v d" in command or "volume down" in command:
                        self.change_volume(up=False)
                        print("🔉 Volume down executed")
                    elif "bu" in command or "b u" in command or "brightness up" in command:
                        self.change_brightness(up=True)
                    elif "bd" in command or "b d" in command or "brightness down" in command:
                        self.change_brightness(up=False)
                    elif "noise low" in command:
                        self.set_noise_sensitivity("low")
                        r.energy_threshold = self.energy_thresholds["low"]
                    elif "noise medium" in command:
                        self.set_noise_sensitivity("medium")
                        r.energy_threshold = self.energy_thresholds["medium"]
                    elif "noise high" in command:
                        self.set_noise_sensitivity("high")
                        r.energy_threshold = self.energy_thresholds["high"]
                    elif any(word in command for word in ["stop", "quit", "exit", "close", "deactivate"]):
                        print("🛑 Stop command received - shutting down eye control")
                        print("🔄 Cleaning up resources...")
                        self.stop()
                        print("✅ Eye control stopped - you can now use voice assistant")
                        break
                    else:
                        print(f"❓ Unknown command: '{command}'")

                except sr.WaitTimeoutError:
                    # Timeout is normal, continue listening
                    continue
                except sr.UnknownValueError:
                    # Could not understand audio - this is normal with noise cancellation
                    print("...Filtering background noise...")
                    pass
                except sr.RequestError as e:
                    print(f"⚠️ Speech recognition service error: {e}")
                    # Wait a bit before retrying to avoid hammering the service
                    time.sleep(2)
                except Exception as e:
                    print(f"⚠️ Error in voice listener: {e}")
                    # Continue listening despite errors
                    time.sleep(1)
        
        print("🔴 Eye control voice listener stopped")

    def run(self):
        if self.active:
            print("⚠️ Eye Controller is already running.")
            return

        print("🟢 Starting Enhanced Eye Controller...")
        self.active = True

        self.voice_command_active = True
        self.voice_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self.voice_thread.start()

        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Eye Control (Head Tracking)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Eye Control (Head Tracking)', 640, 480)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        if not cap.isOpened():
            print("❌ Error: Could not open camera for eye control.")
            self.stop()
            return

        while self.active and cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            cam_height, cam_width, _ = image.shape
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_brow = landmarks[65]
                right_brow = landmarks[295]

                head_x_cam = int(((left_brow.x + right_brow.x) / 2) * cam_width)
                head_y_cam = int(((left_brow.y + right_brow.y) / 2) * cam_height)
                cv2.circle(image, (head_x_cam, head_y_cam), 5, (0, 255, 0), -1)

                roi_width, roi_height = 400, 250
                roi_x_min = (cam_width - roi_width) // 2
                roi_y_min = (cam_height - roi_height) // 2
                roi_x_max = roi_x_min + roi_width
                roi_y_max = roi_y_min + roi_height
                cv2.rectangle(image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), (0, 255, 255), 2)

                clamped_x = np.clip(head_x_cam, roi_x_min, roi_x_max)
                clamped_y = np.clip(head_y_cam, roi_y_min, roi_y_max)

                target_x = np.interp(clamped_x, [roi_x_min, roi_x_max], [0, self.screen_width])
                target_y = np.interp(clamped_y, [roi_y_min, roi_y_max], [0, self.screen_height])

                pyautogui.moveTo(target_x, target_y, duration=0.0)

            # Add noise sensitivity indicator on the image
            cv2.putText(image, f"Noise Filter: {self.noise_sensitivity.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Commands: Click, Scroll, VU/VD, BU/BD, Stop", 
                       (10, cam_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "Say 'noise low/medium/high' to adjust sensitivity", 
                       (10, cam_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Eye Control (Head Tracking)', image)
            
            # Check for stop more frequently and with shorter wait
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or not self.active:
                break

        # Cleanup
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=1)

        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Eye Controller stopped and resources released.")


if __name__ == '__main__':
    controller = EyeController()
    controller.run()
