# Eye Tracking Cursor Control

This program allows you to control your computer's cursor using eye movements. It uses your webcam to track your eyes and translates the eye position into cursor movements.

## Prerequisites

- Python 3.7 or higher
- Webcam
- The shape predictor file for facial landmarks

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the shape predictor file:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Usage

1. Run the program:
```bash
python eye_tracker.py
```

2. Position yourself in front of your webcam, ensuring good lighting and that your face is clearly visible.

3. The program will track your eyes and move the cursor accordingly. The cursor movement is mapped to your eye position relative to the webcam frame.

4. Press 'q' to quit the program.

## Notes

- The program uses your webcam, so make sure it's properly connected and accessible.
- Good lighting conditions will improve tracking accuracy.
- The cursor movement is smoothed to prevent jittery movements.
- You may need to adjust the `CALIBRATION_POINTS` in the code to better match your setup and preferences.

## Troubleshooting

If you experience issues:
1. Make sure your webcam is properly connected and accessible
2. Check if all dependencies are correctly installed
3. Ensure the shape predictor file is in the same directory as the script
4. Try adjusting the lighting conditions
5. Make sure you're positioned correctly in front of the webcam 