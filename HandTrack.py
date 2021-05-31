import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDrawLine = mp.solutions.drawing_utils

# Frame Rates
prev_time = 0
curr_time = 0

while True:
    isTrue, frame = capture.read()

    # mediapipe: hands only uses RGB images
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Deal with multiple hands in frame
        # Draw lines between keypoints on the hand
        for hand in results.multi_hand_landmarks:
            # Get information for each hand (ID, Landmark Info.)
            for id, landmark in enumerate(hand.landmark):
                # print(id, landmark)
                height, width, channels = frame.shape
                pos_x, pos_y = int(landmark.x * width), int(landmark.y * height)

                print(id, pos_x, pos_y)

                # First Landmark
                if id == 0:
                    cv.circle(frame, (pos_x, pos_y), 15, (0, 0, 255), cv.FILLED)

                # Thumb Tip
                if id == 4:
                    cv.circle(frame, (pos_x, pos_y), 15, (0, 0, 255), cv.FILLED)

            mpDrawLine.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX,
               2, (0, 255, 0), 2)

    cv.imshow("Video Capture", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()