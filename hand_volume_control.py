import cv2

from hand_tracking_module import HandDetector, capture_video

##########################
wCam, hCam = 640, 480
##########################


if __name__ == '__main__':
    capture_video(detect_on_video, HandDetector(), wCam, hCam)
