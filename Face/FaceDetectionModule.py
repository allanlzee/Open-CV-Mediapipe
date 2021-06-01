import cv2 as cv
import mediapipe as mp
import time


class FaceDetector():

    def __init__(self, min_detection_con=0.5):

        self.min_detections_con = min_detection_con

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_con)

    def detect_faces(self, frame, draw=True):
        # Convert to RGB Image
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(frame_rgb)

        # Return the bounding boxes by appending them to the array
        bound_box_return = []

        if self.results.detections:
            # Find all faces in frame
            for id, detection in enumerate(self.results.detections):
                bound_box_class = detection.location_data.relative_bounding_box
                height, width, channels = frame.shape
                bound_box = int(bound_box_class.xmin * width), int(bound_box_class.ymin * height), \
                            int(bound_box_class.width * width), int(bound_box_class.height * height)

                bound_box_return.append([id, bound_box, detection.score])

                frame = self.draw(frame, bound_box)
                if draw:
                    # cv.rectangle(frame, bound_box, (255, 0, 255), 2)
                    cv.putText(frame, f' Confidence: {int(detection.score[0] * 100)}',
                               (bound_box[0], bound_box[1] - 20), cv.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2)

        return frame, bound_box_return

    # Draws additional lines on the box containing the face
    def draw(self, frame, bounding_box, length=40, thickness=10, draw=True):
        x, y, width, height = bounding_box
        corner_x, corner_y = x + width, y + height

        if draw:
            cv.rectangle(frame, bounding_box, (255, 0, 255), 2)

            # Top Left
            cv.line(frame, (x, y), (x + length, y), (255, 0, 255), thickness)
            cv.line(frame, (x, y), (x, y + length), (255, 0, 255), thickness)

            # Top Right
            cv.line(frame, (corner_x, y), (corner_x - length, y), (255, 0, 255), thickness)
            cv.line(frame, (corner_x, y), (corner_x, y + length), (255, 0, 255), thickness)

            # Bottom Left
            cv.line(frame, (x, corner_y), (x + length, corner_y), (255, 0, 255), thickness)
            cv.line(frame, (x, corner_y), (x, corner_y - length), (255, 0, 255), thickness)

            # Bottom Right
            cv.line(frame, (corner_x, corner_y), (corner_x - length, corner_y), (255, 0, 255), thickness)
            cv.line(frame, (corner_x, corner_y), (corner_x, corner_y - length), (255, 0, 255), thickness)

        return frame


def main():
    capture = cv.VideoCapture(0)
    prev_time = 0

    # Instantiate new detector object
    detector = FaceDetector()

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


if __name__ == "__main__":
    main()
