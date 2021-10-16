"""
Main App
"""
import cv2
from src import help_functions
from src import pose_detetion_module

P_TIME = 0
video = cv2.VideoCapture(0)
pose_detector = pose_detetion_module.PoseDetector()

while True:
    ret, img = video.read()
    img = cv2.resize(img, (1360, 765))

    img = pose_detector.find_pose(img)
    lm_list = pose_detector.get_pixel_positions(img)

    fps, p_time = help_functions.get_fps(P_TIME)
    cv2.putText(img, str(int(fps)), (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()

cv2.destroyAllWindows()
