import cv2 as cv
import mediapipe as mp
import time

# mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

"""
def __init__(self,
    static_image_mode=False,       <- Tries to detect based on tracking confidence
    upper_body_only=False,    
    smooth_landmarks=True,
    min_detection_confidence=0.5,  <- Goes back to detection or keeps on tracking 
    min_tracking_confidence=0.5)   <- based on confidence from the ML model
"""

# Captures video from the webcam
capture = cv.VideoCapture(0)

curr_time = 0
prev_time = 0
while True:
    isTrue, frame = capture.read()

    # Convert from BGR to RGB
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = pose.process(frameRGB)
    print(results.pose_landmarks)

    # Draw in connections
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, width, channels = frame.shape
            print(id, landmark)

            # To get the pixel value
            pos_x, pos_y = int(landmark.x * width), int(landmark.y * height)

            # Draw in circles on each body landmark
            cv.circle(frame, (pos_x, pos_y), 5, (255, 0, 0), cv.FILLED)

    curr_time = time.time()
    fps = 1/(curr_time - prev_time)
    prev_time = curr_time

    cv.putText(frame, str(int(fps)), (70, 50), cv.FONT_HERSHEY_COMPLEX, 3,
        (255, 255, 255), 2)

    cv.imshow("Video Capture", frame)
    cv.waitKey(1)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
