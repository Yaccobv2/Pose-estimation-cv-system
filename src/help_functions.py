"""
Different functions used in project
"""

import time


def get_fps(prev_time):
    """
    Calculate number of fps in the video
    :return number of fps
    """

    cap_time = time.time()
    fps_val = 1 / (cap_time - prev_time)
    prev_time = cap_time

    return fps_val, prev_time
