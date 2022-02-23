"""
Different functions used in project
"""
import time


def get_fps(prev_time: float):
    """
    Calculate number of fps in the video
        Args:
                prev_time: previous time

        return:
                fps_val: fps value
                prev_time: previous time value
    """

    cap_time = time.time()
    fps_val = 1 / (cap_time - prev_time)
    prev_time = cap_time

    return fps_val, prev_time
