import cv2
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm
import math
from pycaw.pycaw import AudioUtilities



device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
print(f"Audio output: {device.FriendlyName}")
print(f"- Muted: {bool(volume.GetMute())}")
print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")


volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
volBar = 400
volPer = 0

cap = cv2.VideoCapture(0)

detector = htm.handDetector(detectionCon=0.7)

while cap.isOpened():
    success, frame = cap.read()

    frame, lmList = detector.findHands(frame)

    x1, y1 = (0, 0)
    x2, y2 = (0, 0)

    for lm in lmList:
        print(lm[4], lm[8])
        x1, y1 = lm[4]
        x2, y2 = lm[8]

    cx, cy = (x1+x2)//2, (y1+y2)//2

    cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)

    length = math.hypot(x2-x1, y2-y1)
    #print(length)

    vol = np.interp(length, [15, 170], [minVol, maxVol])
    volBar = np.interp(length, [15, 170], [400, 150])
    volPer = np.interp(length, [15, 170], [0, 100])
    print(vol)
    volume.SetMasterVolumeLevel(vol, None)

    if length < 40:
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

    cv2.rectangle(frame, (50, 150), (85, 400), (250, 0, 0), 2)
    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (250, 0, 0), cv2.FILLED)
    cv2.putText(frame, f'{int(volPer)}%', (40,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (250,0,0), 2)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break