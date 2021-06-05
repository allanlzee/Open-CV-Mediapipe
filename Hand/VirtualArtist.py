import cv2 as cv
import numpy as np
import time
import os
import HandTrackModule as htm
import math

# Constants
brush_thickness = 5
eraser_thickness = 100
pos_x, pos_y = 0, 0
canvas = np.ones((720, 1227, 3), np.uint8)
draw_color = (0, 0, 0)

folder_path = "Artist"
images = os.listdir(folder_path)
print(images)

images_list = []
for image_path in images:
    image = cv.imread(f'{folder_path}/{image_path}')
    images_list.append(image)

print(len(images_list))

header = images_list[6]
red = images_list[4]
blue = images_list[5]
green = images_list[0]
orange = images_list[1]
purple = images_list[3]
black = images_list[2]

capture = cv.VideoCapture(0)
capture.set(3, 1227)
capture.set(4, 720)

detector = htm.HandsDetector(detection_con=0.85)

header_color = [True, False, False, False, False, False, False]

while True:
    is_true, frame = capture.read()
    # Resize frame to fit the art bar
    frame = cv.resize(frame, (1227, 720))
    frame = cv.flip(frame, 1)

    # Set Header
    if header_color[1]:
        frame[0:118, 0:1227] = red
    elif header_color[2]:
        frame[0:118, 0:1227] = blue
    elif header_color[3]:
        frame[0:118, 0:1227] = green
    elif header_color[4]:
        frame[0:118, 0:1227] = orange
    elif header_color[5]:
        frame[0:118, 0:1227] = purple
    elif header_color[6]:
        frame[0:118, 0:1227] = black
    else:
        frame[0:119, 0:1227] = header

    # Hand Landmarks and Check Whether Fingers are up
    frame = detector.detect_hands(frame)
    hand_landmarks = detector.find_position(frame, draw=False)

    if len(hand_landmarks) > 0:
        # print(hand_landmarks)

        # Finger Tips: Index, Middle
        index_x, index_y = hand_landmarks[8][1], hand_landmarks[8][2]
        middle_x, middle_y = hand_landmarks[12][1], hand_landmarks[12][2]

        # Draw Circles on the Finger Tips
        cv.circle(frame, (index_x, index_y), 5, (255, 255, 0), -1)
        cv.circle(frame, (middle_x, middle_y), 5, (255, 255, 0), -1)

        fingers_up = detector.fingers_up(frame, hand_landmarks, flipped=True)
        # print(fingers_up)

        # Check for finger positions
        if fingers_up[1] and fingers_up[2]:
            # Selection Mode (two fingers are up)
            # print("Selection Mode")
            # Draw Line Between Fingers
            cv.line(frame, (index_x, index_y), (middle_x, middle_y), (80, 127, 255), 2)
            length = math.sqrt((middle_x - index_x) ** 2 + (middle_y - middle_x) ** 2) + 25

            print(index_x, index_y, middle_x, middle_y)

            # Select Colors
            if index_y < 140 and middle_y < 135:
                for i in range(0, 7):
                    header_color[i] = False
                # Red
                if index_x < 260 and middle_x < index_x + length:
                    header_color[1] = True
                    draw_color = (0, 0, 255)
                # Blue
                elif index_x < 425 and middle_x < index_x + length:
                    header_color[2] = True
                    draw_color = (255, 0, 0)
                # Green
                elif index_x < 590 and middle_x < index_x + length:
                    header_color[3] = True
                    draw_color = (0, 255, 0)
                # Orange
                elif index_x < 765 and middle_x < index_x + length:
                    header_color[4] = True
                    draw_color = (80, 127, 255)
                # Purple
                elif index_x < 920 and middle_x < index_x + length:
                    header_color[5] = True
                    draw_color = (226, 37, 173)
                # Black
                elif index_x < 1060 and middle_x < index_x + length:
                    header_color[6] = True
                    draw_color = (0, 0, 0)
                # Eraser
                else:
                    header_color[0] = True
                    draw_color = (0, 0, 0)

        elif fingers_up[1]:
            # Drawing Mode (index finger up only)
            print("Drawing Mode")
            # Draw Circle on the Index Finger Tip
            cv.circle(frame, (index_x, index_y), 8, draw_color, -1)

            if pos_x == 0 and pos_y == 0:
                pos_x, pos_y = index_x, index_y

            if draw_color == (0, 0, 0):
                cv.line(frame, (pos_x, pos_y), (index_x, index_y), draw_color, eraser_thickness)
                cv.line(canvas, (pos_x, pos_y), (index_x, index_y), draw_color, eraser_thickness)
            else:
                cv.line(frame, (pos_x, pos_y), (index_x, index_y), draw_color, brush_thickness)
                cv.line(canvas, (pos_x, pos_y), (index_x, index_y), draw_color, brush_thickness)

            pos_x, pos_y = index_x, index_y

        else:
            print("Nothing Selected")

    cv.imshow("Virtual Artist", frame)
    cv.imshow("Draw Pad", canvas)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
