import cv2 as cv
import mediapipe as mp
import time

# Webcam
capture = cv.VideoCapture(0)
prev_time = 0

# Initialize Media Pipe Functions
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    is_true, frame = capture.read()

    # Note that face detection occurs in RGB instead of BGR
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceDetection.process(frame_rgb)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            # mpDraw.draw_detection(frame, detection)

            bounding_box_class = detection.location_data.relative_bounding_box
            height, width, channels = frame.shape
            bounding_box = int(bounding_box_class.xmin * width), int(bounding_box_class.ymin * height), \
                           int(bounding_box_class.width * width), int(bounding_box_class.height * height)
            cv.rectangle(frame, bounding_box, (255, 0, 255), 2)
            cv.putText(frame, f'Confidence: {int(detection.score[0] * 100)}%',
                       (bounding_box[0], bounding_box[1] - 20),
                       cv.FONT_HERSHEY_PLAIN,
                       2, (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time
    cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN,
               3, (0, 0, 255), 2)

    cv.imshow("Video Capture", frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
