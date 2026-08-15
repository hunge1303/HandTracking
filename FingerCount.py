import cv2
import numpy as np
import mediapipe as mp
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionCon=0.75, maxHands=1)

tipIds = [4, 8, 12, 16, 20]

while cap.isOpened():
    success, frame = cap.read()
    detections, lmList = detector.findHands(frame)

    fingers = []
    x1 = [0] * 5
    y1 = [0] * 5
    x2 = [0] * 5
    y2 = [0] * 5

    for lm in lmList:
        for i in range(5):
            x1[i], y1[i] = lm[tipIds[i]]
            x2[i], y2[i] = lm[tipIds[i]-2]
        if x1[0] > x2[0]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if y1[id] < y2[id]:
                fingers.append(1)
            else:
                fingers.append(0)

    totalFingers = fingers.count(1)
    print(totalFingers)


    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break