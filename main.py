import cv2, torch
import time, os
import mediapipe as mp
import numpy as np
from model import Model

current_path = os.getcwd()

# Camera number, can be varied if using multiple webcams
cam_number = 0
# Laterally inverting video stream
flip = True
# Minimum confidence score required for detecting and marking hand landmarks
min_conf = 0.75
max_hands = 2
# Path of trained model. Can be changed to point to a custom model
model_path = os.path.join(current_path, 'models/120.pt')
# Pen parameters
pen_color = (255, 0, 0)
eraser_size = 80
pen_size = 10
# The density of the line. Smaller values make the line more smooth.
intermediate_step_gap = 4
# Create Control window to change color and size of pen
cv2.namedWindow('control')
# This show the color
img = np.zeros((200, 600, 3), np.uint8)


def nothing(x):
    pass


# This create trackbar to adjust various values.
cv2.createTrackbar('Red', 'control', 0, 255, nothing)
cv2.createTrackbar('Blue', 'control', 0, 255, nothing)
cv2.createTrackbar('Green', 'control', 0, 255, nothing)
cv2.createTrackbar('pen_thickness', 'control', 5, 30, nothing)
cv2.createTrackbar('Imd_step_gap', 'control', 10, 29, nothing)
# Button size
button = [20, 60, 145, 460]

# Function to click button
def process_click(event, x, y, flags, params):
    # check if the click is within the dimensions of the button
    if event == cv2.EVENT_LBUTTONDOWN:
        if button[0] < y < button[1] and button[2] < x < button[3]:
            cv2.imwrite('Image' + str(fps) + '.png', frame)
            img[:80, :] = (0, 0, 255)


cv2.setMouseCallback('control', process_click)
cap = cv2.VideoCapture(cam_number)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=max_hands,
    min_detection_confidence=min_conf,
    min_tracking_confidence=min_conf
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


## Extract landmark positions as array
def landmark_extract(hand_lms, mpHands):
    output_lms = []

    for lm in _lm_list:
        lms = hand_lms.landmark[lm]
        output_lms.append(lms.x)
        output_lms.append(lms.y)
        output_lms.append(lms.z)

    return output_lms


## Checks if the position is out of bounds or not
def is_position_out_of_bounds(position, top_left, bottom_right):
    return (
            position[0] > top_left[0] and position[0] < bottom_right[0]
            and position[1] > top_left[1] and position[1] < bottom_right[1]
    )


## Loading torch model
model = Model()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

action_map = {0: 'Draw', 1: 'Erase', 2: 'None'}

## cv2 text parameters
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 4
## Stores previously drawn circles to give continous lines and also store current color and size of pen
circles = []

was_drawing_last_frame = False

ptime = 0
ctime = 0

## Video feed loop
while True:
    success, frame = cap.read()
    if flip:
        frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    cv2.rectangle(frame, (w, h), (w - 320, h - 90), (0, 0, 0), -1, 1)
    b = cv2.getTrackbarPos('Blue', 'control')
    g = cv2.getTrackbarPos('Green', 'control')
    r = cv2.getTrackbarPos('Red', 'control')
    t = cv2.getTrackbarPos('pen_thickness', 'control')
    # Added 1 to make range of imd_step_gap equal to [1, 30].
    imd_step_gap = (cv2.getTrackbarPos('Imd_step_gap', 'control')+1)/10
    
    intermediate_step_gap = imd_step_gap
    if not results.multi_hand_landmarks:
        was_drawing_last_frame = False
        cv2.putText(frame, 'No hand in frame', (w - 300, h - 50), font, fontScale, fontColor, lineType)
    else:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            ## Mode check
            landmark_list = landmark_extract(hand_landmarks, mpHands)
            model_input = torch.tensor(landmark_list, dtype=torch.float).unsqueeze(0)
            action = action_map[torch.argmax(model.forward(model_input)).item()]
            cv2.putText(frame, f"Mode : {action}", (w - 300, h - 50), font, fontScale, fontColor, lineType)

            ## Draw mode
            if action == 'Draw':
                pen_color = (b, g, r)
                pen_size = t
                index_x = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
                index_y = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y
                pos = (int(index_x * w), int(index_y * h))
                cv2.circle(frame, pos, 20, (255, 0, 0), 2)
                if was_drawing_last_frame:
                    prev_pos = circles[-1][0]
                    x_distance = pos[0] - prev_pos[0]
                    y_distance = pos[1] - prev_pos[1]
                    distance = (x_distance ** 2 + y_distance ** 2) ** 0.5
                    num_step_points = int(int(distance) // intermediate_step_gap) - 1
                    if num_step_points > 0:
                        x_normalized = x_distance / distance
                        y_normalized = y_distance / distance
                        for i in range(1, num_step_points + 1):
                            step_pos_x = prev_pos[0] + int(x_normalized * i)
                            step_pos_y = prev_pos[1] + int(y_normalized * i)
                            step_pos = (step_pos_x, step_pos_y)
                            circles.append((step_pos, pen_color, pen_size))
                
                circles.append((pos, pen_color, pen_size))
                was_drawing_last_frame = True
            else:
                was_drawing_last_frame = False

            ## Erase mode
            if action == 'Erase':
                eraser_mid = [
                    int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x * w),
                    int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
                ]

                bottom_right = (eraser_mid[0] + eraser_size, eraser_mid[1] + eraser_size)
                top_left = (eraser_mid[0] - eraser_size, eraser_mid[1] - eraser_size)

                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 5)

                circles = [
                    (position, pen_color, pen_size)
                    for position, pen_color, pen_size in circles
                    if not is_position_out_of_bounds(position, top_left, bottom_right)
                ]

    ## Draws all stored circles
    for position, pen_color, pen_size in circles:
        frame = cv2.circle(frame, position, pen_size, pen_color, -1)

    ctime = time.time()
    fps = round(1 / (ctime - ptime), 2)
    ptime = ctime
    cv2.putText(frame, f'FPS : {fps}', (w - 300, h - 20), font, fontScale, fontColor, lineType)

    cv2.imshow('output', frame)
    # contol_image = img[:80, :]
    img[button[0]:button[1], button[2]:button[3]] = 180
    cv2.putText(img, 'Click_to_save_img', (148, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.0, 0, 1)
    cv2.imshow('control', img)
    img[:80, :] = [255, 255, 255]
    img[80:, :] = [b, g, r]

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
