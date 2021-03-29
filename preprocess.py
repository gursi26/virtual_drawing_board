import cv2
import mediapipe as mp 
import numpy as np
import pandas as pd 

# ---------------------------------------------- Single run ----------------------------------------- #
cols = []
cols.append('label')
for i in range(63):
    cols.append(str(i))

dataset = pd.DataFrame(columns=cols)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

lm_list = [
    mpHands.HandLandmark.WRIST, 
    mpHands.HandLandmark.THUMB_CMC, 
    mpHands.HandLandmark.THUMB_MCP,
    mpHands.HandLandmark.THUMB_IP, 
    mpHands.HandLandmark.THUMB_TIP, 
    mpHands.HandLandmark.INDEX_FINGER_MCP,
    mpHands.HandLandmark.INDEX_FINGER_DIP, 
    mpHands.HandLandmark.INDEX_FINGER_PIP, 
    mpHands.HandLandmark.INDEX_FINGER_TIP,
    mpHands.HandLandmark.MIDDLE_FINGER_MCP,
    mpHands.HandLandmark.MIDDLE_FINGER_DIP, 
    mpHands.HandLandmark.MIDDLE_FINGER_PIP, 
    mpHands.HandLandmark.MIDDLE_FINGER_TIP, 
    mpHands.HandLandmark.RING_FINGER_MCP, 
    mpHands.HandLandmark.RING_FINGER_DIP,
    mpHands.HandLandmark.RING_FINGER_PIP, 
    mpHands.HandLandmark.RING_FINGER_TIP, 
    mpHands.HandLandmark.PINKY_MCP,
    mpHands.HandLandmark.PINKY_DIP, 
    mpHands.HandLandmark.PINKY_PIP, 
    mpHands.HandLandmark.PINKY_TIP
]

# ---------------------------------------------- Loop -------------------------------------------- #

img_path = '/Users/gursi/Desktop/ASL/asl_dataset/A/A627.jpg'
frame = cv2.imread(img_path)

img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(img_rgb)

letter = 'sample'

if results.multi_hand_landmarks : 
    for hand_landmarks in results.multi_hand_landmarks :
        landmark_list = [letter]
        mp_draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

        for lm in lm_list : 
            lms = hand_landmarks.landmark[lm]
            landmark_list.append(lms.x)
            landmark_list.append(lms.y)
            landmark_list.append(lms.z)

        landmark_df = pd.DataFrame([landmark_list], columns=cols)
        dataset = dataset.append(landmark_df)
            


