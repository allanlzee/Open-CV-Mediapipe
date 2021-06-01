import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)

prev_time = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    is_true, frame = capture.read()
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = faceMesh.process(frame_rgb)

    if results.multi_face_landmarks:
        # Landmarks for multiple faces
        for face_mark in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, face_mark, mpFaceMesh.FACE_CONNECTIONS,
                                 drawSpecs, drawSpecs)

            for id, landmark in enumerate(face_mark.landmark):

                # Conversion to pixels
                height, width, channels = frame.shape
                pos_x, pos_y, pos_z = int(landmark.x * width), int(landmark.y * height), int(landmark.z * channels)

                print(id, pos_x, pos_y)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv.putText(frame, f'FPS: {str(int(fps))}', (20, 70), cv.FONT_HERSHEY_PLAIN,
               2, (0, 255, 0), 3)
    cv.imshow("Video Capture", frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
