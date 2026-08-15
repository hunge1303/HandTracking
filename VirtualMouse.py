import cv2
import numpy as np
import HandTrackingModule as htm
import pyautogui as pag
import time

wCam, hCam = 640, 480
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

frameR = 100
smoothening = 5

detector = htm.handDetector(detectionCon=0.75, maxHands=1)
wScreen, hScreen = pag.size()
print(wScreen, hScreen)

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
    cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 2)

    if len(fingers) != 0:
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pag.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
            length = detector.findDistance(int(x1), int(y1), int(x2), int(y2))

            if length < 25:
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
                pag.click()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(frame, "FPS: {0:.2f}".format(fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break