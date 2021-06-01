import cv2 as cv
import time
import FaceDetectionModule as fd

capture = cv.VideoCapture(0)
prev_time = 0

# Instantiate new detector object
detector = fd.FaceDetector()

while True:
    is_true, frame = capture.read()

    frame, bound_box = detector.detect_faces(frame)

    if bound_box:
        print(bound_box)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv.putText(frame, f'FPS: {str(int(fps))}', (20, 70), cv.FONT_HERSHEY_SIMPLEX,
                   2, (0, 255, 0), 2)
    cv.imshow("Video Capture", frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
