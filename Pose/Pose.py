import cv2 as cv
import time
import PoseEstimationModule as pm

capture = cv.VideoCapture(0)
prev_time = 0

# Use class
detector = pm.poseDetector()

while True:
    is_true, frame = capture.read()
    frame = detector.detect_pose(frame)
    # use draw=False to disable drawing

    landmarks = detector.detect_position(frame)
    # use draw=False to disable drawing

    if landmarks:
        print(landmarks[0])
        cv.circle(frame, (landmarks[0][1], landmarks[0][2]), 10, (0, 0, 255), cv.FILLED)

    # (landmarks[n][1], landmarks[n][2]) -> n is an int from 0 - 32 for various landmarks
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv.putText(frame, str(int(fps)), (70, 50), cv.FONT_HERSHEY_COMPLEX, 3,
                       (255, 255, 255), 2)

        cv.imshow("Video Capture", frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break

capture.release()
cv.destroyAllWindows()
