import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, static_image_mode=False, max_faces=2,
                 min_detect_con=0.5, min_track_con=0.5):

        self.static_image_mode = static_image_mode
        self.max_faces = max_faces
        self.min_detect_con = min_detect_con
        self.min_track_con = min_track_con

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_faces,
                                                 self.min_detect_con, self.min_track_con)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def detect_face_mesh(self, frame, draw=True):

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        self.results = self.faceMesh.process(frame_rgb)

        faces = []

        if self.results.multi_face_landmarks:
            # Landmarks for multiple faces
            for face_mark in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, face_mark, self.mpFaceMesh.FACE_CONNECTIONS,
                                        self.drawSpecs, self.drawSpecs)

                face_values = []
                for id, landmark in enumerate(face_mark.landmark):
                    # Conversion to pixels
                    height, width, channels = frame.shape
                    pos_x, pos_y, pos_z = int(landmark.x * width), int(landmark.y * height), int(landmark.z * channels)

                    # print(id, pos_x, pos_y)

                    face_values.append([id, pos_x, pos_y])
            faces.append(face_values)

        return frame, faces


def main():
    capture = cv.VideoCapture(0)
    prev_time = 0

    detector = FaceMeshDetector()

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


if __name__ == "__main__":
    main()
