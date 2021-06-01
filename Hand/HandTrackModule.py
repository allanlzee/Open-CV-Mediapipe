import cv2 as cv
import mediapipe as mp
import time


class handsDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDrawLine = mp.solutions.drawing_utils

    def detectHands(self, frame, draw=True):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        self.results = self.hands.process(rgb_frame)

        if self.results.multi_hand_landmarks:
            # Deal with multiple hands in frame
            # Draw lines between key points on the hand
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDrawLine.draw_landmarks(frame, hand, self.mpHands.HAND_CONNECTIONS)

                """ 
                # Get information for each hand (ID, Landmark Info.)
                    # First Landmark
                    if id == 0:
                        cv.circle(frame, (pos_x, pos_y), 15, (0, 0, 255), cv.FILLED)
                    # Thumb Tip 
                    if id == 4: 
                        cv.circle(frame, (pos_x, pos_y), 15, (0, 0, 255), cv.FILLED)
                """

        return frame

    def findPosition(self, frame, hand_mark=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hand_mark]

            for id, landmark in enumerate(myHand.landmark):
                # print(id, landmark)
                height, width, channels = frame.shape
                pos_x, pos_y = int(landmark.x * width), int(landmark.y * height)

                lmList.append([id, pos_x, pos_y])

                if draw:
                    # print(id, pos_x, pos_y)
                    cv.circle(frame, (pos_x, pos_y), 15, (0, 0, 255), cv.FILLED)

        return lmList


def main():
    capture = cv.VideoCapture(0)
    # Frame Rates
    prev_time = 0
    curr_time = 0

    detector = handsDetector()  # Use default parameters

    while True:
        is_true, frame = capture.read()

        frame = detector.detectHands(frame)
        # detector.detectHands(frame, draw=False) -> Prevents Drawing of Lines

        lmList = detector.findPosition(frame)
        # detector.findPosition(frame, draw = False) -> Prevents Drawing of Landmarks

        if len(lmList) != 0:
            print(lmList[4])  # Base of Hand (wrist)
            # This will make sure the console only prints the specified landmark

        landmarks = detector.findPosition(frame)
        # landmarks = detector.findPosition(frame, draw=False) -> Prevents Drawing of Hand Marks

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX,
                   2, (0, 255, 0), 2)

        cv.imshow("Video Capture", frame)

        # Press d to stop the program
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

# External Usages:
# To use the module in a new file, project etc.
# import cv2
# import time
# import mediapipe as mp
# import HandTrackModule as ...

""" 
capture = cv.VideoCapture(0)
    # Frame Rates
    prev_time = 0
    curr_time = 0 
    detector = handsDetector() # Use default parameters
    while True:
        isTrue, frame = capture.read()
        frame = detector.detectHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0: 
            print(lmList[4]) # Base of Hand (wrist)
            # This will make sure the console only prints where '0' 
            # or whatever hand landmark is specified
        landmarks = detector.findPosition(frame)
        curr_time = time.time() 
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX,
            2, (0, 255, 0), 2)
        cv.imshow("Video Capture", frame)

        # Press d to stop the program
        if cv.waitKey(20) & 0xFF == ord('d'):
            break 

    capture.release()
    cv.destroyAllWindows() 
"""