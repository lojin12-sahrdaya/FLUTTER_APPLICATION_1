import cv2
import mediapipe as mp
import pyautogui
import math
import time

class GestureController:
    """
    This class merges your advanced gesture logic (fist control, volume, clicks, etc.)
    into a structure that the server can start and stop in a thread.
    """
    def __init__(self):
        # Correcting your original typo from _init_ to the proper __init__
        print("Initializing Advanced Gesture Controller Settings...")
        
        # --- Your Custom Settings ---
        self.SMOOTHING_FACTOR = 5
        self.LEFT_CLICK_THRESHOLD = 0.04
        self.RIGHT_CLICK_THRESHOLD = 0.04
        self.DOUBLE_CLICK_THRESHOLD = 0.04
        self.DRAG_THRESHOLD = 0.05
        self.VOLUME_THRESHOLD = 0.05
        self.CLOSED_FIST_THRESHOLD = 0.08
        self.SLIDE_MOVE_Y_THRESHOLD = 0.03

        # --- Your State Variables ---
        self.prev_x, self.prev_y = 0, 0
        self.curr_x, self.curr_y = 0, 0
        self.click_lock_time = 0
        self.CLICK_COOLDOWN = 0.3
        self.double_click_start_time = 0
        self.DOUBLE_CLICK_HOLD_TIME = 1.0 # Reduced for better responsiveness
        self.is_dragging = False
        self.is_volume_controlling = False
        self.initial_volume_y = None
        self.is_slide_controlling = False
        self.initial_fist_y = None
        self.last_slide_action_time = 0
        self.SLIDE_COOLDOWN = 0.5
        self.last_left_click_time = 0
        self.last_right_click_time = 0
        
        # --- Double Click with Thumb-Ring Finger ---
        self.THUMB_RING_DOUBLE_CLICK_THRESHOLD = 0.04
        self.THUMB_RING_HOLD_TIME = 1.5  # 1.5 seconds hold for double click
        self.thumb_ring_pinch_start_time = 0
        self.is_thumb_ring_pinching = False
        self.double_click_executed = False
        
        # --- Controller State ---
        self.active = False # This controls the main run() loop

    def stop(self):
        """Signals the run loop to stop."""
        if self.active:
            print("🔴 Stopping Advanced Gesture Controller.")
            self.active = False

    # --- All of your gesture logic is now part of the class ---
    def _is_fist_closed(self, hand_landmarks):
        try:
            # Using specific finger joint distances for more reliability
            wrist = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            
            # If fingertips are closer to the wrist than the base of the fingers, it's likely a fist
            dist_index = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
            dist_middle = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
            dist_ring = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y)
            
            # A simple average distance threshold can work well
            return (dist_index + dist_middle + dist_ring) / 3 < 0.25 # Adjust this threshold
        except:
            return False

    def _move_mouse(self, hand_landmarks):
        index_finger_tip = hand_landmarks.landmark[8]
        screen_width, screen_height = pyautogui.size()
        target_x = screen_width * index_finger_tip.x
        target_y = screen_height * index_finger_tip.y
        self.curr_x = self.prev_x + (target_x - self.prev_x) / self.SMOOTHING_FACTOR
        self.curr_y = self.prev_y + (target_y - self.prev_y) / self.SMOOTHING_FACTOR
        if not self.is_dragging and not self.is_slide_controlling:
            pyautogui.moveTo(self.curr_x, self.curr_y)
        self.prev_x, self.prev_y = self.curr_x, self.curr_y

    def _check_clicks_and_gestures(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_finger_tip = hand_landmarks.landmark[8]
        middle_finger_tip = hand_landmarks.landmark[12]
        ring_finger_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        wrist = hand_landmarks.landmark[0]
        current_time = time.time()
        
        # --- Your gesture logic hierarchy ---
        # Note: This logic is complex and gesture conflicts are possible.
        # The order of these checks is very important.

        # 1. Slide Control (Highest Priority)
        if self._is_fist_closed(hand_landmarks):
            if not self.is_slide_controlling:
                self.is_slide_controlling = True
                self.initial_fist_y = wrist.y
                print("Slide Control ON")
            else:
                delta_y = wrist.y - self.initial_fist_y
                if current_time - self.last_slide_action_time > self.SLIDE_COOLDOWN:
                    if delta_y < -self.SLIDE_MOVE_Y_THRESHOLD:
                        pyautogui.press('up')
                        print("Slide UP")
                        self.last_slide_action_time = current_time
                        self.initial_fist_y = wrist.y
                    elif delta_y > self.SLIDE_MOVE_Y_THRESHOLD:
                        pyautogui.press('down')
                        print("Slide DOWN")
                        self.last_slide_action_time = current_time
                        self.initial_fist_y = wrist.y
            return # Exit to prioritize this gesture

        elif self.is_slide_controlling:
            self.is_slide_controlling = False
            self.click_lock_time = current_time + self.CLICK_COOLDOWN
            print("Slide Control OFF")
            return

        # 2. Volume Control
        volume_dist = math.hypot(thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y)
        if volume_dist < self.VOLUME_THRESHOLD:
            if not self.is_volume_controlling:
                self.is_volume_controlling = True
                self.initial_volume_y = pinky_tip.y
                print("Volume Control ON")
            else:
                delta_y = pinky_tip.y - self.initial_volume_y
                if abs(delta_y) > 0.02: # Sensitivity
                    if delta_y < 0: pyautogui.press('volumeup')
                    else: pyautogui.press('volumedown')
                    self.initial_volume_y = pinky_tip.y
            return
        elif self.is_volume_controlling:
            self.is_volume_controlling = False
            print("Volume Control OFF")
            return

        # 3. Dragging
        drag_dist = math.hypot(thumb_tip.x - index_finger_tip.x, thumb_tip.y - index_finger_tip.y)
        if drag_dist < self.DRAG_THRESHOLD and not self.is_dragging:
             # Check if we are close enough in distance to start a drag
            self.is_dragging = True
            pyautogui.mouseDown(button='left')
            print("Drag STARTED")
        elif self.is_dragging and drag_dist >= self.DRAG_THRESHOLD:
            pyautogui.mouseUp(button='left')
            self.is_dragging = False
            self.click_lock_time = current_time + self.CLICK_COOLDOWN
            print("Drag ENDED")
        if self.is_dragging:
            pyautogui.moveTo(self.curr_x, self.curr_y) # Ensure mouse moves while dragging
            return # Prioritize dragging over clicks

        # If we are in a cooldown period, do nothing else
        if current_time < self.click_lock_time:
            return

        # 4. Double Click with Thumb-Ring Finger Pinch (1.5 seconds hold)
        thumb_ring_dist = math.hypot(thumb_tip.x - ring_finger_tip.x, thumb_tip.y - ring_finger_tip.y)
        if thumb_ring_dist < self.THUMB_RING_DOUBLE_CLICK_THRESHOLD:
            if not self.is_thumb_ring_pinching:
                # Start timing the pinch
                self.is_thumb_ring_pinching = True
                self.thumb_ring_pinch_start_time = current_time
                self.double_click_executed = False
                print("Thumb-Ring pinch detected, holding for double click...")
            else:
                # Check if we've held long enough for double click
                hold_duration = current_time - self.thumb_ring_pinch_start_time
                if hold_duration >= self.THUMB_RING_HOLD_TIME and not self.double_click_executed:
                    pyautogui.doubleClick(button='left')
                    print("Double Click! (Thumb-Ring 1.5s hold)")
                    self.double_click_executed = True
                    self.click_lock_time = current_time + self.CLICK_COOLDOWN
                    return
        else:
            # Reset thumb-ring pinch state when fingers are no longer pinched
            if self.is_thumb_ring_pinching:
                self.is_thumb_ring_pinching = False
                self.thumb_ring_pinch_start_time = 0
                self.double_click_executed = False

        # 5. Quick Clicks
        right_click_dist = math.hypot(thumb_tip.x - middle_finger_tip.x, thumb_tip.y - middle_finger_tip.y)
        if right_click_dist < self.RIGHT_CLICK_THRESHOLD:
            pyautogui.click(button='right')
            print("Right Click!")
            self.click_lock_time = current_time + self.CLICK_COOLDOWN
            return

        # Left click should be checked last as it's part of the drag gesture
        if drag_dist < self.LEFT_CLICK_THRESHOLD:
            pyautogui.click(button='left')
            print("Left Click!")
            self.click_lock_time = current_time + self.CLICK_COOLDOWN
            return

    def process_gestures_wrapper(self, hand_landmarks):
        """A wrapper to handle the main logic and reset states if hand is lost."""
        if not hand_landmarks:
            if self.is_dragging:
                pyautogui.mouseUp(button='left')
                print("Drag Ended (Hand Lost)!")
            self.is_dragging = False
            self.is_volume_controlling = False
            self.is_slide_controlling = False
            self.double_click_start_time = 0
            # Reset thumb-ring finger pinch state
            self.is_thumb_ring_pinching = False
            self.thumb_ring_pinch_start_time = 0
            self.double_click_executed = False
            return
        
        self._move_mouse(hand_landmarks)
        self._check_clicks_and_gestures(hand_landmarks)

    def run(self):
        """The main execution loop that opens the camera and processes gestures."""
        if self.active:
            print("⚠️ Gesture Controller is already running.")
            return

        print("🟢 Starting Advanced Gesture Controller...")
        self.active = True
        
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        if not cap.isOpened():
            print("❌ Error: Could not open camera for gesture control.")
            self.active = False
            return

        while self.active and cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                # Process the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                self.process_gestures_wrapper(hand_landmarks)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            else:
                # Call the wrapper with None to reset states
                self.process_gestures_wrapper(None)
            
            cv2.imshow('Advanced Hand Gesture Control', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                self.stop() # Signal a stop if 'q' is pressed
        
        # Cleanup
        self.active = False
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Advanced Gesture Controller stopped and resources released.")


if __name__ == '__main__':
    # This part allows you to test the script directly
    print("Testing Advanced Gesture Controller directly...")
    controller = GestureController()
    controller.run()