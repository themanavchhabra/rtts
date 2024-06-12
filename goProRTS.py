'''Go Pro Imports'''
import cv2
from time import time
import socket
from goprocam import GoProCamera, constants
'''**************'''
'''PCA 9685 Imports'''
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
'''****************'''
'''R.T.S. Imports'''
import numpy as np
#import time
from imutils.video import FPS
from time import sleep
import RPi.GPIO as GPIO
'''**************'''

'''Driver Code'''
'''***********'''
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("BEGIN")
initBB = None
bbox = None
gpCam = GoProCamera.GoPro(constants.gpcontrol)
gpCam.overview()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t=time()
gpCam.livestream("start")
gpCam.video_settings(res='1080p', fps='30')
gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
cap = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)
tracker = cv2.legacy.TrackerMOSSE_create()
success = 0
while(1):
    start = time.time()
    _,frame = vid.read()
    print(frame.shape)
    W = frame.shape[0]
    H = frame.shape[1]
    center_x = W/2
    center_y = H/2
    if success==0 :
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(faces)
        print("FACES MEIN KOI HAI????", len(faces))
        (H, W) = frame.shape[:2]
        if len(faces) == 1:
            for x,y,w,h in faces:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                bbox=(x,y,w,h)
                tracker = cv2.legacy.TrackerMOSSE_create()
                print(bbox)
        else:
            pass
        
    for i in faces:
        print(i.shape)
    
    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
            errorPan = x - center_x
            errorTilt = y- center_y
            
            
            thetaX = int(errorPan*56.71524/W)
            print("###############  THETA X ###############", thetaX)
            kit.servo[0].angle = 90 - thetaX
            thetaY = int(errorTilt*42.7428/H)
            print("############### THETA y ###############", thetaY)
            kit.servo[4].angle = 180 + thetaY 

        fps.update()
        fps.stop()
        info = [
            ("Tracker", tracker),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    key = cv2.waitKey(1) & 0xFF

    if bbox != None:
        print("INTO BBOX")
        initBB = bbox
        tracker.init(frame, initBB)
        print(tracker.init(frame, initBB))
        fps = FPS().start() 
    else:
        bbox = None
        pass
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end = time.time()
    print("##########FPS##########", 1/(end - start))
 
cv2.destroyAllWindows()
 