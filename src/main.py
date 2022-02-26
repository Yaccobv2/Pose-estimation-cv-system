# pylint: disable=W0612,E1101,C0415,E0401,W0703,W1309

"""
Main App
"""
import time
import argparse
import cv2
from help_functions import get_fps
from barbell_detection_module import BarbellDetector
from camera_config import gstreamer_pipeline

PARSER = argparse.ArgumentParser()

PARSER.add_argument('-s', '--save', action='store_true', help='Save data and plots after recording')
PARSER.add_argument('-c', '--camera', action='store_true',
                    help='Use camera connected to JetsonNano')
PARSER.add_argument('-vp', '--video-path', action='extend',
                    type=str, nargs='*',
                    help='Set the video path to load')
PARSER.add_argument('-pd', '--pose-detector', action='store_true', help='Detect pose')
PARSER.add_argument('-a', '--angles', action='store_true',
                    help='Detect angles in joints')
PARSER.add_argument('-b', '--barbell-detector', action='store_true',
                    help='Detect center of the barbell')
PARSER.add_argument('-f', '--fps', action='store_true', help='Plot fps')
PARSER.add_argument('-sa', '--save-animation', action='extend',
                    type=str, nargs='+',
                    help='Save animation after recording and'
                         ' specify formats of files. Available: mp4')
PARSER.add_argument('-d', '--display', action='store_true', help='Display view while recording')
PARSER.add_argument('-o', '--output-file', help='Output file path')

ARGS = PARSER.parse_args()


def main() -> None:
    """
    Main function
    """

    output_file = ARGS.output_file or f"output.csv"

    if ARGS.pose_detector:
        from pose_detetion_module import PoseDetector

        pose_detector = PoseDetector(model_complexity=0)

    if ARGS.save_animation is not None:
        from plotter_3d import Plotter3D

        result_landmarks = []
        result_images = []
        formats = ARGS.save_animation

        plotter_3d = Plotter3D()
        plotter_3d.set_connections(pose_detector.get_connections())

    # choose video source
    if ARGS.video_path is not None:
        print("Path to the video: " + str(ARGS.video_path[0]))
        video = cv2.VideoCapture(str(ARGS.video_path[0]))
    else:
        if ARGS.camera:
            video = cv2.VideoCapture(gstreamer_pipeline(flip_method=0),
                                     cv2.CAP_GSTREAMER)
        else:
            video = cv2.VideoCapture(0)

    barbell_detector = BarbellDetector()
    barbell_detector.init_network()
    # camera image size
    image_dimensions = (None, None)
    network_input_frame_size = (416, 416)

    detection_results = []
    p_time = 0
    start_time = time.time()

    while True:
        ret, frame = video.read()

        # noinspection PyBroadException
        try:
            frame = cv2.resize(frame, (360, 640))
        except Exception as exception:
            print("Can't resize the frame: " + str(exception))

        # if the frame dimensions are empty, grab them
        if image_dimensions[0] is None or image_dimensions[1] is None:
            image_dimensions = frame.shape[:2]

        if ARGS.pose_detector:
            frame, pose_world_landmarks = pose_detector.find_pose(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if ARGS.save_animation:
                result_landmarks.append(pose_world_landmarks)
                result_images.append(frame_rgb)

        if ARGS.barbell_detector:
            # create input matrix from frame, apply transformations
            # and pass it to the first layer of ANN
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, network_input_frame_size,
                                         swapRB=True, crop=False)
            layer_outputs, runtime = barbell_detector.forward_pass(blob=blob)
            detections = barbell_detector.yolo_detect(layerOutputs=layer_outputs,
                                                      imageDimensions=image_dimensions)

            detection_time = time.time() - start_time
            iteration_results = []
            for detection in detections:

                if detection["label"] == "barbell":
                    iteration_results.append({"label": detection["label"],
                                              "results": [detection_time, detection["center"]]})
                    cv2.circle(frame, detection["center"], 5, (255, 0, 255), -1)
                    text = detection["label"] + ": " + str(round(detection["confidence"], 4))
                    cv2.putText(frame, text, (detection["x"], detection["y"] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                detection["color"], 2)

                if detection["label"] == "foot":
                    iteration_results.append({"label": detection["label"],
                                              "results": [detection_time, detection["center"]]})

                    cv2.line(frame, (int(detection["x"] + detection["w"] / 2), detection["y"]),
                             (int(detection["x"] + detection["w"] / 2),
                              detection["y"] + detection["h"]),
                             (255, 0, 255), 2)

                    text = detection["label"] + ": " + str(round(detection["confidence"], 4))
                    cv2.putText(frame, text, (detection["x"], detection["y"] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                detection["color"], 2)

            detection_results.append(iteration_results)
        # if ARGS.angles:
        #     lm_list = pose_detector.get_pixel_positions(img)
        #     angles = pose_detector.get_all_angles(img, lm_list, constants.JOINTS)
        #     result.append([time_curr-time_start, angles])

        if ARGS.fps:
            fps, p_time = get_fps(p_time)
            cv2.putText(frame, str(int(fps)), (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    if ARGS.pose_detector:
        if ARGS.save_animation:
            if formats:
                print(formats)
                plotter_3d.create_animation(result_landmarks, result_images)

    if ARGS.save:
        barbell_detector.make_plots(detection_results,
                                    image_dimensions[1], image_dimensions[0], show=True)


if __name__ == "__main__":
    main()
