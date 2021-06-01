import cv2 as cv
import time
import FaceMeshModule as fm

capture = cv.VideoCapture(0)
prev_time = 0

detector = fm.FaceMeshDetector()

while True:
    is_true, frame = capture.read()

    frame, faces = detector.detect_face_mesh(frame)

    if faces:
        print(faces[0])

    # Frames Per Second
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv.putText(frame, f'FPS: {str(int(fps))}', (20, 70), cv.FONT_HERSHEY_PLAIN,
               2, (0, 255, 0), 3)
    cv.imshow("Video Capture", frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
