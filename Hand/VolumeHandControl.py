import cv2 as cv
import mediapipe as mp
import time
import HandTrackModule
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL # Windows :(
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# pycaw for Volume Control
##############
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
print(volume.GetVolumeRange())
volume.SetMasterVolumeLevel(-20.0, None)
###############

camera_width, camera_height = 2000, 1000

capture = cv.VideoCapture(0)
capture.set(3, camera_width)
capture.set(4, camera_height)

# Frame Rates
prev_time = 0
curr_time = 0

detector = HandTrackModule.handsDetector(detection_con=0.8)  # Use default parameters

while True:
    is_true, frame = capture.read()

    frame = detector.detect_hands(frame)
    # detector.detectHands(frame, draw=False) -> Prevents Drawing of Lines

    lmList = detector.find_position(frame, draw=False)
    # detector.findPosition(frame, draw=False) -> Prevents Drawing of Landmarks

    if len(lmList) != 0:
        # print(lmList[0], lmList[4], lmList[8])

        pos_wrist_x, pos_wrist_y = lmList[0][1], lmList[0][2]
        pos_thumb_x, pos_thumb_y = lmList[4][1], lmList[4][2]
        pos_index_x, pos_index_y = lmList[8][1], lmList[8][2]

        # Center of line between thumb and index
        mid_x, mid_y = (pos_thumb_x + pos_index_x) // 2, (pos_thumb_y + pos_index_y) // 2

        cv.circle(frame, (pos_wrist_x, pos_wrist_y), 15, (255, 0, 0), -1)
        cv.circle(frame, (pos_thumb_x, pos_thumb_y), 15, (255, 0, 255), -1)
        cv.circle(frame, (pos_index_x, pos_index_y), 15, (255, 0, 255), -1)

        # Length of the Line
        line_length = math.hypot(pos_index_x - pos_thumb_x, pos_index_y - pos_thumb_y)
        print(line_length)

        if line_length < 50:
            cv.circle(frame, (mid_x, mid_y), 15, (0, 255, 0), -1)
        elif line_length < 175:
            cv.circle(frame, (mid_x, mid_y), 15, (255, 0, 0), -1)
        elif line_length < 300:
            cv.circle(frame, (mid_x, mid_y), 15, (0, 0, 255), -1)
        else:
            cv.circle(frame, (mid_x, mid_y), 15, (255, 0, 255), -1)


        # Line between index tip and thumb top
        cv.line(frame, (pos_thumb_x, pos_thumb_y), (pos_index_x, pos_index_y), (255, 0, 255), 3)

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
