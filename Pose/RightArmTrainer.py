import cv2 as cv
import numpy as np
import time
import PoseEstimationModule

capture = cv.VideoCapture(0)
prev_time = 0

detector = PoseEstimationModule.PoseDetector(detection_con=0.8, track_con=0.8)

rep_counter_left_arm = 0
rep_counter_right_arm = 0

# Boolean for the rep
rep_left_arm = False
rep_right_arm = False

while True:
    is_true, frame = capture.read()
    frame = detector.detect_pose(frame, draw=False)

    width, height, channels = frame.shape
    width *= 0.75
    height *= 0.75

    frame = cv.resize(frame, (int(height), int(width)))

    landmarks = detector.detect_position(frame, draw=False)
    if landmarks:
        # print(landmarks)
        # Find the Angles of the Important Points
        pass

    # Arms
    # frame, joint_angle_left_arm = detector.calculate_angle(frame, 11, 13, 15)

    frame, joint_angle_right_arm = detector.calculate_angle(frame, 12, 14, 16)

    # Legs
    # frame = detector.calculate_angle(frame, 23, 25, 27, color=(255, 0, 255))
    # frame = detector.calculate_angle(frame, 24, 26, 28, color=(255, 0, 255))

    # Calculate Repetitions
    if int(joint_angle_right_arm) <= 35 and rep_left_arm is False:
        rep_counter_right_arm += 1
        rep_left_arm = True
    elif int(joint_angle_right_arm) <= 35 and rep_left_arm is True:
        rep_left_arm = True
    else:
        rep_left_arm = False

    # Display Reps
    cv.putText(frame, f'Right Arm Reps: {str(rep_counter_right_arm)}', (10, 80), cv.FONT_HERSHEY_COMPLEX, 1,
               (0, 255, 0), 2)

    # Frame Rates
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv.putText(frame, f'FPS: {str(int(fps))}', (10, 40), cv.FONT_HERSHEY_COMPLEX, 1,
               (0, 255, 0), 2)

    cv.imshow("Workout Trainer", frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break
