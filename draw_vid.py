import cv2
import mediapipe as mp 
from model import Model 
import torch 

cam_number = 0
flip = True 
min_conf = 0.75
max_hands = 2
model_path = '/users/gursi/desktop/virtual_drawing_board/models/120.pt'
pen_color = (255,0,0)
eraser_size = 80

cap = cv2.VideoCapture(cam_number)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=max_hands,
    min_detection_confidence=min_conf,
    min_tracking_confidence=min_conf
)
mp_draw = mp.solutions.drawing_utils


## Extract landmark positions as array 
def landmark_extract(hand_lms, mpHands):
    output_lms = []

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

    for lm in lm_list : 
        lms = hand_lms.landmark[lm]
        output_lms.append(lms.x)
        output_lms.append(lms.y)
        output_lms.append(lms.z)

    return output_lms


## Loading torch model
model = Model()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

action_map = {0:'Draw', 1:'Erase', 2:'None'}

## cv2 text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,255,255)
lineType = 4

## Stores previously drawing circles to give continous lines
circles = []

## Video feed loop
while True : 
    success, frame = cap.read()
    if flip : 
        frame = cv2.flip(frame,1)
    h,w,c = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks : 
        cv2.putText(frame, 'No hand in frame', (20, h-35), font, fontScale, fontColor, lineType)
    else : 
        for hand_landmarks in results.multi_hand_landmarks :
            mp_draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            ## Mode check 
            landmark_list = landmark_extract(hand_landmarks, mpHands)
            model_input = torch.tensor(landmark_list, dtype=torch.float).unsqueeze(0)
            action = action_map[torch.argmax(model.forward(model_input)).item()]
            cv2.putText(frame, f'Mode : {action}', (20, h-50), font, fontScale, fontColor, lineType)

            ## Draw mode
            if action == 'Draw': 
                index_x = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
                index_y = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y
                pos = (int(index_x*w), int(index_y*h))
                cv2.circle(frame, pos, 20, (255,0,0), 2)
                circles.append(pos)

            ## Erase mode
            if action == 'Erase':
                eraser_mid = [
                        int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x * w),
                        int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
                    ]

                bottom_right = (eraser_mid[0]+eraser_size, eraser_mid[1]+eraser_size)
                top_left = (eraser_mid[0]-eraser_size, eraser_mid[1]-eraser_size)

                cv2.rectangle(frame, top_left, bottom_right, (0,0,255), 5)
                
                try : 
                    for pt in range(len(circles)):
                        if circles[pt][0]>top_left[0] and circles[pt][0]<bottom_right[0]: 
                            if circles[pt][1]>top_left[1] and circles[pt][1]<bottom_right[1]:
                                circles.pop(pt)
                except IndexError : 
                    pass

    ## Draws all stored circles 
    for position in circles : 
        frame = cv2.circle(frame, position, 10, pen_color, -1)
    
    cv2.imshow('output', frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
            break

