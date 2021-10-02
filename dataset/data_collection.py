import cv2
import mediapipe as mp 
import pandas as pd 

# ---------------------------------------------- Single run ----------------------------------------- #
cols = ['label', *map(str, range(63))]
dataset = pd.DataFrame(columns=cols)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

_lm_list = [
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

def landmark_extract(hand_lms, mpHands):
    output_lms = []

    for lm in _lm_list : 
        lms = hand_lms.landmark[lm]
        output_lms.append(lms.x)
        output_lms.append(lms.y)
        output_lms.append(lms.z)

    return output_lms


# ---------------------------------------------- Loop -------------------------------------------- #

vid = cv2.VideoCapture(0)
action = input('Enter action name : ')

try : 
    while True : 

        success, frame = vid.read()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks : 
            for hand_landmarks in results.multi_hand_landmarks :
                mp_draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                landmark_list = [action, *landmark_extract(hand_landmarks, mpHands)]

                landmark_df = pd.DataFrame([landmark_list], columns=cols)
                dataset = dataset.append(landmark_df)
        
        cv2.imshow('output', frame)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

except KeyboardInterrupt : 
    pass

dataset.to_csv(f'{action}.csv')
            


