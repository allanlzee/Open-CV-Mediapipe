import cv2 as cv
import time
import HandTrackModule as HandTrack
import os

capture = cv.VideoCapture(0)
frame_width, frame_height = 1000, 750
capture.set(3, frame_width)
capture.set(4, frame_height)

folder_path = "Fingers"
images = os.listdir(folder_path)
print(images)

images_list = []
counter = 1
for image_path in images:
    image = cv.imread(f'{folder_path}/{str(counter)}.jpeg')
    print(f'{folder_path}/{str(counter)}.jpeg')
    images_list.append(image)
    counter += 1

# print(images_list)

prev_time = 0

# Use hand tracking module
detector = HandTrack.handsDetector(detection_con=0.8)

finger_tips = [4, 8, 12, 16, 20]

while True:
    is_true, frame = capture.read()
    frame = detector.detect_hands(frame)
    hand_landmarks = detector.find_position(frame, draw=False)

    if hand_landmarks:
        # print(hand_landmarks)
        # Find the Finger Tips
        # 4, 8, 12, 16, 20 -> Tips
        # 3, 7, 11, 15, 19 -> Knuckles
        # Use 0 to track hand orientation

        # print(hand_landmarks[0])
        pos_wrist_x = hand_landmarks[0][1]
        pos_wrist_y = hand_landmarks[0][2]

        fingers = []

        # Hand's Orientation is Normal
        if pos_wrist_y - 100 > hand_landmarks[12][2]:
            # Special Case for the Thumb
            if hand_landmarks[finger_tips[0]][1] < hand_landmarks[finger_tips[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for tip in range(1, 5):
                finger_tip = finger_tips[tip]
                if hand_landmarks[finger_tip][2] < hand_landmarks[finger_tip - 1][2]:
                    # negative values -> higher on frame
                    fingers.append(1)
                else:
                    fingers.append(0)

        # Hand's Orientation is Upside Down
        elif pos_wrist_y + 100 < hand_landmarks[12][2]:
            # Special case for the thumb
            if hand_landmarks[finger_tips[0]][1] < hand_landmarks[finger_tips[0] - 1][1]:
                fingers.append(1)
            else:

                fingers.append(0)
            for tip in range(1, 5):
                finger_tip = finger_tips[tip]
                if hand_landmarks[finger_tip][2] > hand_landmarks[finger_tip - 1][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        # If fingers point to the left
        elif pos_wrist_x + 100 < hand_landmarks[12][1]:

            if hand_landmarks[finger_tips[0]][2] < hand_landmarks[finger_tips[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            for tip in range(1, 5):
                finger_tip = finger_tips[tip]
                if hand_landmarks[finger_tip][1] > hand_landmarks[finger_tip - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        # If the fingers point to the right
        elif pos_wrist_x - 100 > hand_landmarks[12][1]:

            if hand_landmarks[finger_tips[0]][2] < hand_landmarks[finger_tips[0] - 1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            for tip in range(1, 5):
                finger_tip = finger_tips[tip]
                if hand_landmarks[finger_tip][1] < hand_landmarks[finger_tip - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        if fingers:
            print(fingers)

    image_height, image_width, image_channels = images_list[0].shape
    frame[0:image_height, 0:image_width] = images_list[0]

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv.putText(frame, f'FPS: {int(fps)}', (frame_width + 125, 50), cv.FONT_HERSHEY_SIMPLEX,
               1, (255, 0, 0), 3)

    cv.imshow("Video Capture", frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
