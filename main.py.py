
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import os
import time

# Fix pyautogui lag
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Configuration
SMOOTHING_FACTOR = 0.5
CLICK_THRESHOLD = 30
SCROLL_THRESHOLD = 50
SCROLL_SPEED = 10
SCROLL_DELAY = 0.1
VOLUME_DELAY = 0.3
CLICK_DELAY = 0.4       # ✅ Added
MODEL_PATH = 'hand_landmarker.task'

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit(1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

screen_width, screen_height = pyautogui.size()
prev_mouse_x, prev_mouse_y = screen_width // 2, screen_height // 2

control_enabled = True
scroll_enabled = True
volume_enabled = True

last_scroll_time = 0
last_volume_time = 0
last_left_click_time = 0    # ✅ Added
last_right_click_time = 0   # ✅ Added

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # ✅ Lower resolution = faster
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)            # ✅ Cap FPS

def is_hand_open(hand_landmarks):
    index_extended = hand_landmarks[8].y < hand_landmarks[5].y
    middle_extended = hand_landmarks[12].y < hand_landmarks[9].y
    ring_extended = hand_landmarks[16].y < hand_landmarks[13].y
    pinky_extended = hand_landmarks[20].y < hand_landmarks[17].y
    return index_extended and middle_extended and ring_extended and pinky_extended

def is_hand_closed(hand_landmarks):
    index_closed = hand_landmarks[8].y > hand_landmarks[5].y
    middle_closed = hand_landmarks[12].y > hand_landmarks[9].y
    ring_closed = hand_landmarks[16].y > hand_landmarks[13].y
    pinky_closed = hand_landmarks[20].y > hand_landmarks[17].y
    return index_closed and middle_closed and ring_closed and pinky_closed

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        gesture_status = "Idle"

        if results.hand_landmarks and control_enabled:
            for hand_landmarks in results.hand_landmarks:
                h, w, _ = img.shape

                thumb = hand_landmarks[4]
                thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
                cv2.circle(img, (thumb_x, thumb_y), 8, (0, 0, 255), -1)

                index_finger = hand_landmarks[8]
                index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
                cv2.circle(img, (index_x, index_y), 8, (255, 0, 0), -1)

                middle_finger = hand_landmarks[12]
                middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)
                cv2.circle(img, (middle_x, middle_y), 8, (0, 255, 0), -1)

                wrist = hand_landmarks[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(img, (wrist_x, wrist_y), 8, (0, 255, 255), -1)

                raw_mouse_x = np.interp(index_x, [0, w], [0, screen_width])
                raw_mouse_y = np.interp(index_y, [0, h], [0, screen_height])

                mouse_x = SMOOTHING_FACTOR * prev_mouse_x + (1 - SMOOTHING_FACTOR) * raw_mouse_x
                mouse_y = SMOOTHING_FACTOR * prev_mouse_y + (1 - SMOOTHING_FACTOR) * raw_mouse_y
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                pyautogui.moveTo(mouse_x, mouse_y)

                distance_thumb_index = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                distance_thumb_middle = np.sqrt((thumb_x - middle_x)**2 + (thumb_y - middle_y)**2)

                now = time.time()

                # ✅ Left click with cooldown, no scroll conflict
                if distance_thumb_index < CLICK_THRESHOLD and now - last_left_click_time > CLICK_DELAY:
                    pyautogui.click()
                    gesture_status = "Left-Clicking"
                    last_left_click_time = now

                # ✅ Right click with cooldown
                elif distance_thumb_middle < CLICK_THRESHOLD and now - last_right_click_time > CLICK_DELAY:
                    pyautogui.rightClick()
                    gesture_status = "Right-Clicking"
                    last_right_click_time = now

                # ✅ Scroll only when NOT clicking (separate gesture)
                elif scroll_enabled and now - last_scroll_time > SCROLL_DELAY:
                    if index_y < h // 2 - SCROLL_THRESHOLD:
                        pyautogui.scroll(SCROLL_SPEED)
                        gesture_status = "Scrolling Up"
                        last_scroll_time = now
                    elif index_y > h // 2 + SCROLL_THRESHOLD:
                        pyautogui.scroll(-SCROLL_SPEED)
                        gesture_status = "Scrolling Down"
                        last_scroll_time = now

                # Volume controls
                if volume_enabled and now - last_volume_time > VOLUME_DELAY:
                    if is_hand_open(hand_landmarks):
                        pyautogui.press('volumeup')
                        gesture_status = "Volume Up"
                        last_volume_time = now
                    elif is_hand_closed(hand_landmarks):
                        pyautogui.press('volumedown')
                        gesture_status = "Volume Down"
                        last_volume_time = now

        status_text = f"Control: {'ON' if control_enabled else 'OFF'} | Scroll: {'ON' if scroll_enabled else 'OFF'} | Volume: {'ON' if volume_enabled else 'OFF'} | {gesture_status}"
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, "t=control  s=scroll  v=volume  q=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Hand Mouse Control", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            control_enabled = not control_enabled
        elif key == ord('s'):
            scroll_enabled = not scroll_enabled
        elif key == ord('v'):
            volume_enabled = not volume_enabled

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    detector.close()