 # Drowsiness_Detect_Guard
 This Python script implements a real-time drowsiness detection system using computer vision techniques. The program utilizes the dlib library for facial landmark detection and calculates the Eye Aspect Ratio (EAR) to determine if a person is showing signs of drowsiness.
# Key Features:
 - ## Eye Aspect Ratio (EAR):
    The EAR is computed based on the distances between facial landmarks around the eyes, allowing the system to monitor eye behavior.
 - ## Sound Alarm:
    When drowsiness is detected, an alarm sound (specified by the user) is played to alert the individual.
 - ## Video Stream Analysis:
    The system continuously analyzes a video stream from a webcam, drawing contours around the eyes and providing a visual indication of drowsiness.
# Dependencies:
- numpy
- pygame-for sound alarm
- imutils-for image processing utilities
- time-for time-related functions
- dlib-for facial landmark detection
- cv2 (OpenCV)-for computer vision tasks
# How to Use:
1. Install the required dependencies: pip install numpy pygame imutils dlib opencv-python.
2. Download the shape_predictor_68_face_landmarks.dat file (facial landmark predictor) from the dlib website and place it in the script's directory.
3. Specify the paths for the shape_predictor file and the alarm sound file in the script.
4. Run the script, and it will initiate the webcam to start the drowsiness detection system.
# Notes:
- Adjust the EYE_AR_THRESH and EYE_AR_CONSEC_FRAMES parameters based on your requirements.
- The script uses multithreading to play the alarm sound concurrently with video processing.
# License
- This project is licensed under the MIT License - see the LICENSE file for details.
