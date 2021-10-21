"""
Main App
"""
import time
import cv2
from src import help_functions
from src import pose_detetion_module
from src import constants

P_TIME = 0
video = cv2.VideoCapture(0)
pose_detector = pose_detetion_module.PoseDetector()
time_start = time.time()

result = []
# result[i] = [time, [[angle_value_1, [[point_1_1], [point_1_2], [point_1_3]]],
# [angle_value_2, [[point_2_1], [point_2_2], [point_2_3]]], ...]],
# each point = [ID, Y, X]

while True:
    ret, img = video.read()
    time_curr = time.time()
    #print("{:10.2f}".format(time_curr - time_start))
    img = cv2.resize(img, (1360, 765))

    img = pose_detector.find_pose(img)
    lm_list = pose_detector.get_pixel_positions(img)
    angles = pose_detector.get_all_angles(img, lm_list, constants.JOINTS)

    result.append([time_curr-time_start, angles])


    fps, p_time = help_functions.get_fps(P_TIME)
    cv2.putText(img, str(int(fps)), (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(result[0])
        print()
        print(result[1])
        break

video.release()

cv2.destroyAllWindows()
