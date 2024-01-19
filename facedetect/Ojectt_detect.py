# loding model
import cv2
from ultralytics import YOLO
import os
import numpy as np
model = YOLO('yolov8n.pt')

# loading video
path = './test.mp4'

# read frmaes
cap = cv2.VideoCapture(path)
ret = True
while ret:
    ret,frame = cap.read()
    if ret==True:
        # detecting objects from frame
        # track object
        result = model.track(frame, persist=True)

        #plot results
        frame_ = result[0].plot()

        # visulaize
        cv2.imshow('frame',frame_)
        if cv2.waitKey(1)==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()