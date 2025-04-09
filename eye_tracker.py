import cv2
import numpy as np
import pyautogui
import dlib
import imutils
from imutils import face_utils
import time
import pytesseract
from PIL import Image
import pyperclip
import keyboard
import threading
import subprocess
import os
from gtts import gTTS
from playsound import playsound
import tempfile

TTS_AVAILABLE = True
TTS_LANGUAGE = 'en'
TTS_SPEED = 1.0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

screen_width, screen_height = pyautogui.size()

CALIBRATION_POINTS = [
    (0.1, 0.1),
    (0.9, 0.1),
    (0.1, 0.9),
    (0.9, 0.9),
    (0.5, 0.5)
]

SENSITIVITY = 1.5
TEXT_REGION_SIZE = 400
TEXT_DETECTION_COOLDOWN = 1.0

should_quit = False
should_detect_text = False
should_adjust_sensitivity = False

def eye_centre(eye_points):
    eye_center = np.mean(eye_points, axis=0).astype(int)
    return eye_center

def eye_mapping(eye_center, frame_shape):
    h, w = frame_shape[:2]
    x, y = eye_center
    x_norm = (x / w - 0.5) * 2
    y_norm = (y / h - 0.5) * 2
    x_norm *= SENSITIVITY
    y_norm *= SENSITIVITY
    screen_x = int((x_norm + 1) * screen_width / 2)
    screen_y = int((y_norm + 1) * screen_height / 2)
    screen_x = max(0, min(screen_x, screen_width))
    screen_y = max(0, min(screen_y, screen_height))
    return screen_x, screen_y

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(enhanced, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

def detect_text(region):
    try:
        processed = preprocess_image(region)
        pil_image = Image.fromarray(processed)
        custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%&*()[]{}:;/\\-_=+<>'
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        text = text.strip()
        text = ' '.join(text.split())
        text = ''.join(char for char in text if char.isprintable())
        return text
    except Exception as e:
        print(f"Error in text detection: {e}")
        return ""

def speak_text(text):
    if not TTS_AVAILABLE:
        print("Text-to-speech is not available. Text will only be copied to clipboard.")
        return
    try:
        if text:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
            tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=False)
            tts.save(temp_filename)
            playsound(temp_filename)
            os.unlink(temp_filename)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        print("Text will only be copied to clipboard.")

def hotkeys():
    global SENSITIVITY, should_quit, should_detect_text, should_adjust_sensitivity

    def on_q():
        global should_quit
        should_quit = True

    def on_t():
        global should_detect_text
        should_detect_text = True

    def on_s():
        global should_adjust_sensitivity, SENSITIVITY
        should_adjust_sensitivity = True
        SENSITIVITY += 1.0 if SENSITIVITY <= 10.0 else SENSITIVITY -= 1.0
        print(f"Sensitivity adjusted to: {SENSITIVITY}")

    keyboard.on_press_key('q', lambda _: on_q())
    keyboard.on_press_key('t', lambda _: on_t())
    keyboard.on_press_key('s', lambda _: on_s())

    while not should_quit:
        time.sleep(0.1)

def main():
    global SENSITIVITY, should_quit, should_detect_text, should_adjust_sensitivity

    print("Starting eye tracking...")
    print("Press 'q' to quit")
    print("Press 't' to detect and speak text at current cursor position")
    print("Press 's' to adjust sensitivity (current: {:.1f})".format(SENSITIVITY))
    if not TTS_AVAILABLE:
        print("Note: Text-to-speech is disabled. Text will only be copied to clipboard.")

    hotkey_thread = threading.Thread(target=handle_hotkeys)
    hotkey_thread.daemon = True
    hotkey_thread.start()

    pyautogui.FAILSAFE = False

    last_text_detection_time = 0
    last_spoken_text = ""

    while not should_quit:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from webcam")
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            left_eye_center = eye_center(left_eye)
            right_eye_center = eye_center(right_eye)
            eye_center = (left_eye_center + right_eye_center) // 2
            screen_x, screen_y = eye_mapping(eye_center, frame.shape)
            current_x, current_y = pyautogui.position()
            target_x = int(current_x + (screen_x - current_x) * 0.3)
            target_y = int(current_y + (screen_y - current_y) * 0.3)
            pyautogui.moveTo(target_x, target_y, duration=0.1)
            cv2.circle(frame, tuple(left_eye_center), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_eye_center), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(eye_center), 3, (0, 255, 0), -1)

        if should_detect_text:
            current_time = time.time()
            if current_time - last_text_detection_time >= TEXT_DETECTION_COOLDOWN:
                cursor_x, cursor_y = pyautogui.position()
                screenshot = pyautogui.screenshot(region=(cursor_x - TEXT_REGION_SIZE//2, 
                                                        cursor_y - TEXT_REGION_SIZE//2,
                                                        TEXT_REGION_SIZE, TEXT_REGION_SIZE))
                region = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                text = detect_text(region)
                if text and text != last_spoken_text:
                    print(f"Detected text: {text}")
                    pyperclip.copy(text)
                    speak_text(text)
                    last_spoken_text = text
                    last_text_detection_time = current_time
            should_detect_text = False

        cv2.imshow("Eye Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_quit = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
