import cv2 as cv
import time as time
import mediapipe as mp
import HandTrackModule

capture = cv.VideoCapture(0)
# Frame Rates
prev_time = 0
curr_time = 0
detector = HandTrackModule.HandsDetector() # Use default parameters
while True:
    isTrue, frame = capture.read()
    frame = detector.detect_hands(frame)

    lmList_hand1 = detector.find_position(frame)
    # lmList_hand2 = detector.find_position(frame, hand_mark=1, color=(255, 0, 0))
    # print(detector.results.multi_hand_landmarks)

    if len(lmList_hand1) != 0:
        print(lmList_hand1[0]) # Base of Hand (wrist)
        # This will make sure the console only prints where '0'
        # or whatever hand landmark is specified

    landmarks = detector.find_position(frame)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX,
            2, (0, 255, 0), 2)
    cv.imshow("Video Capture", frame)

    # Press d to stop the program
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()