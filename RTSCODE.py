#!/usr/bin/env python3


import numpy as np
import cv2
import time
from imutils.video import FPS
# import RPi.GPIO as GPIO
import time     
import RPi.GPIO as GPIO
from time import sleep
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)


def motorInput(angle):
    # Direction pin from controller
    DIR = 10
    # Step pin from controller
    STEP = 8
    # 0/1 used to signify clockwise or counterclockwise.
    CW = 1
    CCW = 0

    # Setup pin layout on PI
    GPIO.setmode(GPIO.BOARD)

    # Establish Pins in software
    GPIO.setup(DIR, GPIO.OUT)
    GPIO.setup(STEP, GPIO.OUT)

    # Set the first direction you want it to spin
    GPIO.output(DIR, CW)

    try:
        
        x = (200/360)*angle
        x = int(x)
        """Change Direction: Changing direction requires time to switch. The
        time is dictated by the stepper motor and controller. """
        sleep(1.0)
        # Esablish the direction you want to go
        GPIO.output(DIR,CW)

        # Run for 200 steps. This will change based on how you set you controller
        if x<0:
            y = x*(-1)
            for i in range(y):

                # Set one coil winding to high
                GPIO.output(STEP,GPIO.LOW)
                # Allow it to get there.
                sleep(.005) # Dictates how fast stepper motor will run
                # Set coil winding to low
                GPIO.output(STEP,GPIO.HIGH)
                sleep(.005) # Dictates how fast stepper motor will run

        """Change Direction: Changing direction requires time to switch. The
        time is dictated by the stepper motor and controller. """
        sleep(1.0)
        GPIO.output(DIR,CCW)
        if x>0:
            print("enter")
            
            for x in range(x):
                GPIO.output(STEP,GPIO.HIGH)
                sleep(.005)
                GPIO.output(STEP,GPIO.LOW)
                sleep(.005)

    # Once finished clean everything up
    except KeyboardInterrupt:
        print("cleanup")
        GPIO.cleanup()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
print("BEGIN")
initBB = None
bbox = None
vid = cv2.VideoCapture(0)
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
        # faces = np.array(faces)
        # faces = list(faces)
        print(faces)
        print("Checking for faces in frame", len(faces))
        (H, W) = frame.shape[:2]
        if len(faces) == 1:
            # print("HAAN BHAI BOL")
            for x,y,w,h in faces:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                bbox=(x,y,w,h)
                tracker = cv2.legacy.TrackerMOSSE_create()
                print(bbox)
        else:
            # print("NIKAL LAVEDAY")
            pass
        
       
    # if len()>1:
    #     bbox = bbox[0]
    #     print(bbox)
    for i in faces:
    # frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        print(i.shape)
    # bbox=(x,y,w,h)

    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
            # print(x,y,w,h)
            errorPan = x - center_x
            errorTilt = y- center_y
            
            
            #thetaX = int(errorPan*(56.71524/W)*1.4)
            #thetaX = int(errorPan*1.2)
            #thetaX = int(errorPan*(48.71524/W))
            thetaX = int(errorPan*(48.71524/W))
            print("###############  THETA X ###############", thetaX)
            kit.servo[0].angle = 90 - thetaX*0.9
            # value_x = motorInput(thetaX)

            #thetaY = int(errorTilt*(42.7428/H)*1.2)
            #thetaY = int(errorTilt*1.4)
            thetaY = int(errorTilt*(42.71524/H))
            print("############### THETA y ###############", thetaY)
            kit.servo[4].angle = 110 - thetaY*0.9

            # value_y = motorInput(thetaX)


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
    # show the output frame

    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if bbox != None:
        print("INTO BBOX")
        # print(bbox_touple)
        initBB = bbox
        tracker.init(frame, initBB)
        print(tracker.init(frame, initBB))
        fps = FPS().start() 
    else:
        bbox = None
        pass
    cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    end = time.time()
    print("##########FPS##########", 1/(end - start))
 
cv2.destroyAllWindows()
 
