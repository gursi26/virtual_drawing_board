import cv2
import mediapipe as mp 
import time 

#img_path = input('Image path : ')
img_path = '/Users/gursi/Desktop/ASL/asl_dataset/A/A1949.jpg'
frame = cv2.imread(img_path)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(img_rgb)

if results.multi_hand_landmarks : 
    for hand_landmarks in results.multi_hand_landmarks :
        mp_draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            
cv2.imshow('output', frame)
cv2.waitKey(0)

