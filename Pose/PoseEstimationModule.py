import cv2 as cv
import mediapipe as mp
import time
import math

class PoseDetector():

    def __init__(self, mode=False, upper_body=False, smooth=True,
                 detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.upper_body = upper_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        # Media Pipe Functions
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upper_body, self.smooth,
                                     self.detection_con, self.track_con)

    def detect_pose(self, frame, draw=True):

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(frame_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return frame

    def detect_position(self, frame, draw=True):
        self.landmarks = []

        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channels = frame.shape
                # print(id, landmark)

                pos_x, pos_y = int(landmark.x * width), int(landmark.y * height)

                self.landmarks.append([id, pos_x, pos_y])

                if draw:
                    cv.circle(frame, (pos_x, pos_y), 5, (255, 0, 0), -1)

        return self.landmarks

    def calculate_angle(self, frame, landmark_1, landmark_2, landmark_3, draw=True, color=(0, 255, 0)):

        # Average the three landmarks to get the joint
        self.joint = int((landmark_1 + landmark_2 + landmark_3) / 3)

        length = len(self.landmarks)
        if self.landmarks:
            self.lm1_pos_x, self.lm1_pos_y = self.landmarks[landmark_1][1:]
            self.lm2_pos_x, self.lm2_pos_y = self.landmarks[landmark_2][1:]
            self.lm3_pos_x, self.lm3_pos_y = self.landmarks[landmark_3][1:]
            self.joint_pos_x, self.joint_pos_y = self.landmarks[self.joint][1:]

        if draw:
            cv.line(frame, (self.lm1_pos_x, self.lm1_pos_y), (self.lm2_pos_x, self.lm2_pos_y), color, 3)
            cv.line(frame, (self.lm2_pos_x, self.lm2_pos_y), (self.lm3_pos_x, self.lm3_pos_y), color, 3)
            cv.line(frame, (self.lm1_pos_x, self.lm1_pos_y), (self.lm3_pos_x, self.lm3_pos_y), color, 3)

            cv.circle(frame, (self.lm1_pos_x, self.lm1_pos_y), 10, (0, 0, 255), -1)
            cv.circle(frame, (self.lm2_pos_x, self.lm2_pos_y), 10, (0, 0, 255), -1)
            cv.circle(frame, (self.lm3_pos_x, self.lm3_pos_y), 10, (0, 0, 255), -1)
            cv.circle(frame, (self.joint_pos_x, self.joint_pos_y), 10, (255, 0, 0), -1)

        # Trigonometry to find angle of joints
        self.landmark_joints = [13, 14, 25, 26]

        # Left Arm
        length1 = math.sqrt(
            (self.lm1_pos_x - self.lm2_pos_x) * (self.lm1_pos_x - self.lm2_pos_x) + (self.lm1_pos_y - self.lm2_pos_y) * (self.lm1_pos_y - self.lm2_pos_y))
        length2 = math.sqrt(
            (self.lm1_pos_x - self.lm3_pos_x) * (self.lm1_pos_x - self.lm3_pos_x) + (self.lm1_pos_y - self.lm3_pos_y) * (self.lm1_pos_y - self.lm3_pos_y))
        length3 = math.sqrt(
            (self.lm2_pos_x - self.lm3_pos_x) * (self.lm2_pos_x - self.lm3_pos_x) + (self.lm2_pos_y - self.lm3_pos_y) * (self.lm2_pos_y - self.lm3_pos_y))
        # print(length1, length2, length3)

        # if int(length1) > 0 & int(length3) > 0:
        angle = (length2 ** 2 - length1 ** 2 - length3 ** 2) / (-2 * length1 * length3)
        joint_angle = math.degrees(math.acos(angle))
        joint_angle = int(joint_angle)
        print(joint_angle)

        return frame, joint_angle


def main():
    capture = cv.VideoCapture(0)
    prev_time = 0

    # Use class
    detector = PoseDetector()

    while True:
        is_true, frame = capture.read()
        frame = detector.detect_pose(frame)
        # use draw=False to disable drawing

        landmarks = detector.detect_position(frame)
        # use draw=False to disable drawing

        if landmarks:
            print(landmarks[0])
            # cv.circle(frame, (landmarks[0][1], landmarks[0][2]), 10, (0, 0, 255), cv.FILLED)

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


if __name__ == "__main__":
    main()