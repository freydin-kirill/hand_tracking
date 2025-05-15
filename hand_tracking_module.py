import cv2
import time
import math
import numpy as np
import mediapipe as mp

from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from typing import NamedTuple, Tuple
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from hand_landmarks import HandLandmarkIndex


min_color = (0, 255, 0)
mark_color = (255, 0, 255)


class HandDetector:
    def __init__(self,
                 mode=False,
                 num_hands=2,
                 complexity=1,
                 detection_conf=0.5,
                 tracking_conf=0.5):

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, num_hands, complexity, detection_conf, tracking_conf)

    def find_hands(self, frame) -> NamedTuple:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb_frame)

    def draw_hands(self, frame, multi_hand_landmarks):
        for hand_landmarks in multi_hand_landmarks:
            self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)


def find_position(frame, hand_landmarks, mark_number: int) -> Tuple[int, int]:
    h, w, c = frame.shape
    lm = hand_landmarks.landmark[mark_number]
    return int(lm.x * w), int(lm.y * h)


def capture_video(func, detector: HandDetector, w_cam: int = 640, h_cam: int = 480, args: Tuple = ()):
    c_time, p_time = 0, 0

    # define a video capture object
    video = cv2.VideoCapture(0)
    video.set(3, w_cam)
    video.set(4, h_cam)
    print("Camera Opened: ", video.isOpened())

    while True:
        # Capture the video frame
        success, frame = video.read()

        func(frame, detector, *args)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('Test Window', frame)
        if not success:
            print("Error Drawing Frame")

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    # After the loop release the cap object
    video.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def detect_on_video(frame, detector: HandDetector):
    results = detector.find_hands(frame)
    if results.multi_hand_landmarks:
        detector.draw_hands(frame, results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            x, y = find_position(frame, hand_landmarks, HandLandmarkIndex.Index_Finger_TIP)
            cv2.circle(frame, (x, y), 10, (255, 0, 255), cv2.FILLED)


def volume_control(frame, detector: HandDetector, min_vol, max_vol):
    results = detector.find_hands(frame)
    if results.multi_hand_landmarks:
        detector.draw_hands(frame, results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            x1, y1 = find_position(frame, hand_landmarks, HandLandmarkIndex.Index_Finger_TIP)
            x2, y2 = find_position(frame, hand_landmarks, HandLandmarkIndex.Thumb_TIP)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(frame, (x1, y1), 10, mark_color, cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, mark_color, cv2.FILLED)
            cv2.circle(frame, (cx, cy), 10, mark_color, cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), mark_color, 3)

            length = math.hypot(x2 - x1, y2 - y1)
            print(length)
            if length < 50:
                cv2.circle(frame, (cx, cy), 10, min_color, cv2.FILLED)

            # Finger range 50 -- 300
            vol = np.interp(length, [50, 300], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)


if __name__ == '__main__':
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    vol_range = volume.GetVolumeRange()
    volume.SetMasterVolumeLevel(0.0, None)
    min_volume = vol_range[0]
    max_volume = vol_range[1]

    capture_video(volume_control, HandDetector(detection_conf=0.7), args=(min_volume, max_volume))
