import cv2

# Load the pre-trained Haar cascade XML file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a MOSSE tracker
tracker = cv2.TrackerMOSSE_create()

# Open a video capture stream (you can replace '0' with the video file path)
cap = cv2.VideoCapture(0)

# Initialize variables for tracking
tracking = False
bbox = None

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

    # If currently tracking, update the tracker and draw the bounding box
    if tracking:
        success, bbox = tracker.update(frame)
        
        if success:
            (x, y, w, h) = tuple(map(int, bbox))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Tracking with Haar Cascade and MOSSE', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
