import cv2 as cv
import numpy as np
import time
import PoseEstimationModule

capture = cv.VideoCapture(0)
prev_time = 0

detector = PoseEstimationModule.PoseDetector(detection_con=0.5, track_con=0.5)

rep_counter_left_leg = 0
rep_counter_right_leg = 0

# Boolean for the rep
rep_left_leg = False
rep_right_leg = False

while True:
    is_true, frame = capture.read()
    frame = detector.detect_pose(frame, draw=False)

    width, height, channels = frame.shape
    width *= 0.75
    height *= 0.75

    frame = cv.resize(frame, (int(height), int(width)))

    landmarks = detector.detect_position(frame, draw=False)

    # Arms
    # frame, joint_angle_left_arm = detector.calculate_angle(frame, 11, 13, 15)

    # frame, joint_angle_right_arm = detector.calculate_angle(frame, 12, 14, 16)

    # Legs
    # frame, joint_angle_left_leg = detector.calculate_angle(frame, 23, 25, 27, color=(255, 0, 255))
    frame, joint_angle_right_leg = detector.calculate_angle(frame, 24, 26, 28, color=(255, 0, 255))

    # Calculate Repetitions
    if int(joint_angle_right_leg) <= 100 and rep_right_leg is False:
        rep_counter_right_leg += 1
        rep_left_leg = True
    elif int(joint_angle_right_leg) <= 100 and rep_right_leg is True:
        rep_right_leg = True
    else:
        rep_right_leg = False

    # Display Reps
    cv.putText(frame, f'Right Leg Reps: {str(rep_counter_right_leg)}', (10, 80), cv.FONT_HERSHEY_COMPLEX, 1,
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
