# Import necessary libraries
from scipy.spatial import distance as dist  # Importing the distance module from scipy library for calculating distances
from imutils.video import VideoStream  # Importing VideoStream class from imutils library for video streaming
from imutils import face_utils  # Importing face_utils module from imutils library for working with facial landmarks
from threading import Thread  # Importing Thread class from threading library for concurrent execution
import numpy as np  # Importing numpy library for numerical operations
import pygame  # Importing pygame library for audio feedback
import imutils  # Importing imutils library for resizing frames
import time  # Importing time library for time-related operations
import dlib  # Importing dlib library for facial detection and landmark prediction
import cv2  # Importing OpenCV library for computer vision tasks

# Function to play alarm sound
def sound_alarm(path):
    pygame.mixer.init()  # Initialize Pygame mixer for playing sounds
    pygame.mixer.music.load(path)  # Load the alarm sound file
    pygame.mixer.music.play()  # Play the alarm sound

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculating Euclidean distances between points on the eye
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # Compute and return the eye aspect ratio using the formula
    ear = (A + B) / (2.04 * C)
    return ear

# Function to detect drowsiness
def detect_drowsiness():
    # Path to the shape predictor file and alarm sound file
    shape_predictor_path = r"Path_of_the_Shape_Predictor"
    alarm_sound_path = r"Path_of_the_Alarm_Sound"
    webcam_index = 0  # Index of the webcam

    # Constants for drowsiness detection
    EYE_AR_THRESH = 0.3  # Threshold for eye aspect ratio indicating drowsiness
    EYE_AR_CONSEC_FRAMES = 48  # Number of consecutive frames for drowsiness detection
    COUNTER = 0  # Counter for consecutive drowsy frames
    ALARM_ON = False  # Flag to indicate if the alarm is activated
    LAST_DETECTION = time.time()  # Time of the last detection

    # Load facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()  # Initialize face detector
    predictor = dlib.shape_predictor(shape_predictor_path)  # Initialize shape predictor
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # Indices of left eye landmarks
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # Indices of right eye landmarks

    # Start video stream thread
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=webcam_index).start()  # Initialize video stream
    time.sleep(1.0)  # Wait for camera to warm up
    
    # Initialize Pygame mixer for sound
    pygame.mixer.init()

    # Infinite loop for continuously processing frames
    while True:
        frame = vs.read()  # Read frame from video stream
        frame = imutils.resize(frame, width=450)  # Resize frame for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        rects = detector(gray, 0)  # Detect faces in the grayscale frame
        drowsy = False  # Initialize drowsiness flag as False

        # Loop over detected faces
        for rect in rects:
            shape = predictor(gray, rect)  # Predict facial landmarks
            shape = face_utils.shape_to_np(shape)  # Convert landmarks to NumPy array
            leftEye = shape[lStart:lEnd]  # Extract left eye landmarks
            rightEye = shape[rStart:rEnd]  # Extract right eye landmarks
            leftEAR = eye_aspect_ratio(leftEye)  # Calculate EAR for left eye
            rightEAR = eye_aspect_ratio(rightEye)  # Calculate EAR for right eye
            ear = (leftEAR + rightEAR) / 2.0  # Average EAR of both eyes

            # Compute convex hull and draw contours around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if EAR falls below threshold
            if ear < EYE_AR_THRESH:
                COUNTER += 1  # Increment counter for consecutive drowsy frames
                # Check if drowsiness is detected for consecutive frames
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    drowsy = True  # Set drowsiness flag to True
                    # Activate alarm if not already on
                    if not ALARM_ON:
                        ALARM_ON = True
                        if alarm_sound_path != "":
                            t = Thread(target=sound_alarm, args=(alarm_sound_path,))
                            t.daemon = True
                            t.start()
                    # Display drowsiness alert on frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    LAST_DETECTION = time.time()  # Update time of last detection
            else:
                COUNTER = 0  # Reset counter if EAR is above threshold
                # Turn off alarm if it was on
                if ALARM_ON:
                    if time.time() - LAST_DETECTION > 3:  # Adjust this threshold as needed
                        ALARM_ON = False
                        pygame.mixer.music.stop()
                # Display EAR value on frame
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw face recognition box based on drowsiness state
            if drowsy:
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255), 2)  # Red box for drowsy
            else:
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)  # Green box for non-drowsy

        # Turn off alarm if no longer drowsy but alarm is still on
        if not drowsy and ALARM_ON:
            ALARM_ON = False
            pygame.mixer.music.stop()

        # Display frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit if 'q' key is pressed
            break

    # Stop alarm and clean up
    pygame.mixer.music.stop()
    cv2.destroyAllWindows()
    vs.stop()

# Main function invocation
if __name__ == "__main__":
    detect_drowsiness()
