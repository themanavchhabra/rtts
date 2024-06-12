import cv2
import numpy as np

# Load the pre-trained Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a MOSSE tracker
tracker = cv2.TrackerMOSSE_create()

# Create a Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * 0.03

# Open a video capture stream (you can replace '0' with the video file path)
cap = cv2.VideoCapture(0)

# Initialize variables for tracking
tracking = False
bbox = None
kalman_state = np.zeros((4, 1), np.float32)

while True:
    # Read a frame from the video capture stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If not currently tracking, detect faces and start tracking the first face found
    if not tracking:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            bbox = tuple(faces[0])
            tracking = True
            tracker.init(frame, bbox)

            # Initialize Kalman filter state
            kalman_state[0] = bbox[0] + bbox[2] / 2
            kalman_state[1] = bbox[1] + bbox[3] / 2

    # If currently tracking, update the tracker and draw the bounding box
    if tracking:
        success, bbox = tracker.update(frame)
        
        if success:
            (x, y, w, h) = tuple(map(int, bbox))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Kalman filter prediction and correction
            kalman_prediction = kalman.predict()
            kalman_corrected = kalman.correct(np.array([[x + w / 2], [y + h / 2]], np.float32))

            # Update bounding box based on Kalman filter output
            x_kalman, y_kalman = int(kalman_corrected[0][0] - w / 2), int(kalman_corrected[1][0] - h / 2)
            cv2.rectangle(frame, (x_kalman, y_kalman), (x_kalman + w, y_kalman + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Face Tracking with Haar Cascade, MOSSE, and Kalman Filter', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
