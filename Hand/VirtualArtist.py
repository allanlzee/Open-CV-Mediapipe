import cv2 as cv
import numpy as np
import time
import os
import HandTrackModule as htm

folder_path = "Artist"
images = os.listdir(folder_path)
print(images)

images_list = []
for image_path in images:
    image = cv.imread(f'{folder_path}/{image_path}')
    images_list.append(image)

print(len(images_list))

header = images_list[0]
capture = cv.VideoCapture(0)
capture.set(3, 1227)
capture.set(4, 720)

detector = htm.HandsDetector(detection_con=0.85)

while True:
    is_true, frame = capture.read()
    # Resize frame to fit the art bar
    frame = cv.resize(frame, (1227, 720))
    frame = cv.flip(frame, 1)

    # Set Header
    frame[0:118, 0:1227] = header

    # Hand Landmarks and Check Whether Fingers are up
    frame = detector.detect_hands(frame)
    hand_landmarks = detector.find_position(frame, draw=False)

    if len(hand_landmarks) > 0:
        # print(hand_landmarks)

        # Finger Tips: Index, Middle
        index_x, index_y = hand_landmarks[8][1], hand_landmarks[8][2]
        middle_x, middle_y = hand_landmarks[12][1], hand_landmarks[12][2]

        # Draw Circles on the Finger Tips
        cv.circle(frame, (index_x, index_y), 8, (255, 255, 0), -1)
        cv.circle(frame, (middle_x, middle_y), 8, (255, 255, 0), -1)

        fingers_up = detector.fingers_up(frame, hand_landmarks, flipped=True)
        print(fingers_up)

        # Check for finger positions
        if fingers_up[1] and fingers_up[2]:
            # Selection Mode (two fingers are up)
            print("Selection Mode")
        elif fingers_up[1]:
            # Drawing Mode (index finger up only)
            print("Drawing Mode")
        else:
            print("Nothing Selected")

    cv.imshow("Virtual Artist", frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
