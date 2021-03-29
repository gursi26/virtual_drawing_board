import cv2
import mediapipe as mp 
import time 
import numpy as np 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
circles = []

while True : 
    success, frame = cap.read()
    h,w,c = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks : 
        for hand_landmarks in results.multi_hand_landmarks :
            mp_draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            index_x = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
            index_y = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y
            pos = (int(index_x*w), int(index_y*h))
            circles.append(pos)

    for position in circles : 
        frame = cv2.circle(frame, position, 10,(255,0,0), -1)
    
    cv2.imshow('output', frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
            break

