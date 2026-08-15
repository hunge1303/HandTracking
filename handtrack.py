import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = "D:\\coding\\models\\handtrack\\hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (0,9),(9,10),(10,11),(0,13),(13,14),(14,15),(15,16),
               (0,17),(17,18),(18,19),(5,9),(9,13),(13,17)]

cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    result = detector.detect(mp_image)

    h, w = frame.shape[:2]
    count=0
    count_2=0
    for hand in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            if count==8 or count_2==29:
                cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            count_2+=1
            count+=1



    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()