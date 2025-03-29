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

# Initialize text-to-speech settings
TTS_AVAILABLE = True
TTS_LANGUAGE = 'en'
TTS_SPEED = 1.0  # Normal speed

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get screen size for cursor movement scaling
screen_width, screen_height = pyautogui.size()

# Calibration points (you can adjust these)
CALIBRATION_POINTS = [
    (0.1, 0.1),    # Top-left
    (0.9, 0.1),    # Top-right
    (0.1, 0.9),    # Bottom-left
    (0.9, 0.9),    # Bottom-right
    (0.5, 0.5)     # Center
]

# Sensitivity adjustment for eye tracking
SENSITIVITY = 1.5  # Increase this value to make the cursor more sensitive to eye movement

# Text detection settings
TEXT_REGION_SIZE = 400  # Increased region size for better text detection
TEXT_DETECTION_COOLDOWN = 1.0  # seconds

# Global flags for hotkey actions
should_quit = False
should_detect_text = False
should_adjust_sensitivity = False

def get_eye_center(eye_points):
    """Calculate the center point of the eye."""
    eye_center = np.mean(eye_points, axis=0).astype(int)
    return eye_center

def map_eye_to_screen(eye_center, frame_shape):
    """Map eye position to screen coordinates with improved sensitivity."""
    h, w = frame_shape[:2]
    x, y = eye_center
    
    # Normalize coordinates to [-1, 1] range
    x_norm = (x / w - 0.5) * 2
    y_norm = (y / h - 0.5) * 2
    
    # Apply sensitivity adjustment
    x_norm *= SENSITIVITY
    y_norm *= SENSITIVITY
    
    # Map back to screen coordinates
    screen_x = int((x_norm + 1) * screen_width / 2)
    screen_y = int((y_norm + 1) * screen_height / 2)
    
    # Ensure coordinates stay within screen bounds
    screen_x = max(0, min(screen_x, screen_width))
    screen_y = max(0, min(screen_y, screen_height))
    
    return screen_x, screen_y

def preprocess_image(image):
    """Preprocess the image for better OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(thresh)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated = cv2.dilate(enhanced, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return eroded

def detect_text(region):
    """Detect text in a given region using Tesseract OCR with improved accuracy."""
    try:
        # Preprocess the image
        processed = preprocess_image(region)
        
        # Convert to PIL Image for Tesseract
        pil_image = Image.fromarray(processed)
        
        # Perform OCR with optimized configuration
        custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%&*()[]{}:;/\\-_=+<>'
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        
        # Clean up the detected text
        text = text.strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Remove any non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        return text
    except Exception as e:
        print(f"Error in text detection: {e}")
        return ""

def speak_text(text):
    """Speak the detected text using Google Text-to-Speech."""
    if not TTS_AVAILABLE:
        print("Text-to-speech is not available. Text will only be copied to clipboard.")
        return
        
    try:
        if text:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
            
            # Generate speech
            tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=False)
            tts.save(temp_filename)
            
            # Play the audio
            playsound(temp_filename)
            
            # Clean up the temporary file
            os.unlink(temp_filename)
            
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        print("Text will only be copied to clipboard.")

def handle_hotkeys():
    """Handle global hotkeys in a separate thread."""
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
        SENSITIVITY = 2.0 if SENSITIVITY == 1.0 else 1.0
        print(f"Sensitivity adjusted to: {SENSITIVITY}")
    
    # Register hotkeys
    keyboard.on_press_key('q', lambda _: on_q())
    keyboard.on_press_key('t', lambda _: on_t())
    keyboard.on_press_key('s', lambda _: on_s())
    
    # Keep the thread running
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
    
    # Start hotkey handler thread
    hotkey_thread = threading.Thread(target=handle_hotkeys)
    hotkey_thread.daemon = True
    hotkey_thread.start()
    
    # Disable pyautogui's fail-safe
    pyautogui.FAILSAFE = False
    
    # Initialize variables for text detection
    last_text_detection_time = 0
    last_spoken_text = ""  # To avoid repeating the same text
    
    while not should_quit:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame from webcam")
            break
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = detector(gray, 0)
        
        for face in faces:
            # Determine facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Get the left and right eye points
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # Calculate eye centers
            left_eye_center = get_eye_center(left_eye)
            right_eye_center = get_eye_center(right_eye)
            
            # Calculate the average eye position
            eye_center = (left_eye_center + right_eye_center) // 2
            
            # Map eye position to screen coordinates
            screen_x, screen_y = map_eye_to_screen(eye_center, frame.shape)
            
            # Move the cursor with smoothing
            current_x, current_y = pyautogui.position()
            target_x = int(current_x + (screen_x - current_x) * 0.3)
            target_y = int(current_y + (screen_y - current_y) * 0.3)
            pyautogui.moveTo(target_x, target_y, duration=0.1)
            
            # Draw the eye centers on the frame
            cv2.circle(frame, tuple(left_eye_center), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(right_eye_center), 3, (0, 0, 255), -1)
            cv2.circle(frame, tuple(eye_center), 3, (0, 255, 0), -1)
        
        # Handle text detection if requested
        if should_detect_text:
            current_time = time.time()
            if current_time - last_text_detection_time >= TEXT_DETECTION_COOLDOWN:
                # Get the current cursor position
                cursor_x, cursor_y = pyautogui.position()
                
                # Capture a region around the cursor
                screenshot = pyautogui.screenshot(region=(cursor_x - TEXT_REGION_SIZE//2, 
                                                        cursor_y - TEXT_REGION_SIZE//2,
                                                        TEXT_REGION_SIZE, TEXT_REGION_SIZE))
                
                # Convert to OpenCV format
                region = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                # Detect text
                text = detect_text(region)
                if text and text != last_spoken_text:
                    print(f"Detected text: {text}")
                    # Copy text to clipboard
                    pyperclip.copy(text)
                    print("Text copied to clipboard!")
                    
                    # Speak the text
                    speak_text(text)
                    
                    # Draw a rectangle around the text region on the frame
                    cv2.rectangle(frame, 
                                (cursor_x - TEXT_REGION_SIZE//2, cursor_y - TEXT_REGION_SIZE//2),
                                (cursor_x + TEXT_REGION_SIZE//2, cursor_y + TEXT_REGION_SIZE//2),
                                (0, 255, 0), 2)
                    
                    last_spoken_text = text
                elif text == last_spoken_text:
                    print("Same text detected, not repeating...")
                else:
                    print("No text detected in the region")
                
                last_text_detection_time = current_time
                should_detect_text = False
        
        # Display the frame
        cv2.imshow("Eye Tracking", frame)
        
        # Handle OpenCV window
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 