import cv2 as cv
import mediapipe as mp
import time


class poseDetector():

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
        landmarks = []

        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channels = frame.shape
                # print(id, landmark)

                pos_x, pos_y = int(landmark.x * width), int(landmark.y * height)

                landmarks.append([id, pos_x, pos_y])

                if draw:
                    cv.circle(frame, (pos_x, pos_y), 5, (255, 0, 0), -1)

        return landmarks


def main():
    capture = cv.VideoCapture(0)
    prev_time = 0

    # Use class
    detector = poseDetector()

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


if __name__ == "__main__":
    main()
