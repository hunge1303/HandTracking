import cv2
import numpy as np
import mediapipe as mp
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(overlayList)

header = overlayList[0]

drawColor = (0, 0, 255)
brushThickness = 12
eraserThickness = 40
xp, yp = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.75, maxHands=1)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    detections, lmList = detector.findHands(frame)
    x1, y1 = (0, 0)
    x2, y2 = (0, 0)

    for lm in lmList:
        x1, y1 = lm[8]
        x2, y2 = lm[12]

    fingers = detector.fingersUp()
    if len(fingers) != 0:
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            if y1 < 125:
                if 220 < x1 < 370:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 475 < x1 < 635:
                    header = overlayList[1]
                    drawColor = (240, 30, 0)
                elif 735 < x1 < 890:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 980 < x1 < 1075:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
                cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, imgCanvas)

    frame[0:84, 0:1280] = header


    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break