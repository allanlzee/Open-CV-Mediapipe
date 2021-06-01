import cv2 as cv
import mediapipe as mp
import time
import HandTrackModule

camera_width, camera_height = 2000, 1000

capture = cv.VideoCapture(0)
capture.set(3, camera_width)
capture.set(4, camera_height)

# Frame Rates
prev_time = 0
curr_time = 0

detector = HandTrackModule.handsDetector()  # Use default parameters

while True:
    is_true, frame = capture.read()

    frame = detector.detect_hands(frame)
    # detector.detectHands(frame, draw=False) -> Prevents Drawing of Lines

    lmList = detector.find_position(frame)
    # detector.findPosition(frame, draw = False) -> Prevents Drawing of Landmarks

    if len(lmList) != 0:
        print(lmList[4])  # Base of Hand (wrist)
        # This will make sure the console only prints the specified landmark

    landmarks = detector.find_position(frame)
    # landmarks = detector.findPosition(frame, draw=False) -> Prevents Drawing of Hand Marks

    # Frame Rate
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv.putText(frame, f'FPS: {str(int(fps))}', (10, 70), cv.FONT_HERSHEY_COMPLEX,
               2, (0, 255, 0), 2)

    cv.imshow("Video Capture", frame)

    # Press d to stop the program
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
