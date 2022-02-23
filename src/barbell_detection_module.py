# pylint: disable=C0103,E1101,R1728,W0703
"""
Barbell detection module
"""

from typing import List, Any, Tuple
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2


class BarbellDetector:
    """
    Barbell detection module.
    """
    def __init__(self, CONFIDENCE_FILTER: float = 0.5, THRESHOLD: float = 0.3,
                 yolo_cfg_file: str = "../yolo/yolov4-tiny.cfg",
                 yolo_weights_file: str = "../yolo/yolov4-tiny_custom_best_v3.weights",
                 yoloNamesFile: str = "../yolo/yolov4.names"):

        # system parameters
        self.CONFIDENCE_FILTER = CONFIDENCE_FILTER
        self.THRESHOLD = THRESHOLD

        # files
        self.yolo_cfg_file = yolo_cfg_file
        self.yolo_weights_file = yolo_weights_file
        self.yolo_names_file = yoloNamesFile

        # network
        self.yolo = None
        self.outputlayers = None
        self.labels = None
        self.colors = None

    def init_network(self) -> None:
        """
        Initialize the yolo network.

        :return: None

        """
        # read neural network structure, weights and biases
        # noinspection PyBroadException
        try:
            self.yolo = cv2.dnn.readNetFromDarknet(self.yolo_cfg_file, self.yolo_weights_file)
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception as e:
            print("Can't load yolo config or weights: " + str(e))
        print(self.yolo_cfg_file)

        self.outputlayers = self.yolo.getUnconnectedOutLayersNames()

        # read labels
        # noinspection PyBroadException
        try:
            with open(self.yolo_names_file, 'r', encoding='UTF-8') as f:
                self.labels = f.read().splitlines()
        except Exception as e:
            print("Can't load yolo labels: " + str(e))

        # create rgb colors for every label
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3),
                                        dtype="uint8")

    def forward_pass(self, blob: List) -> tuple[Any, float]:
        """
        Perform forward pass.

        :param blob: input frame formatted with cv2.dnn.blobFromImage function

        :return: The output of the network and time it took to perform forward pass

        """

        self.yolo.setInput(blob)

        # make forward pass and calculate its time
        start = time.time()
        layerOutputs = self.yolo.forward(self.outputlayers)
        end = time.time()

        runtime = end - start

        return layerOutputs, runtime

    def yolo_detect(self, layerOutputs: Any, imageDimensions: Tuple):
        """
           Process output of the network.

           :param layerOutputs: output of yolo network from forward_pass function
           :param imageDimensions: tuple containing the size of the input frame

           :return: The list of dicts containing information about detected objects

       """

        outputs = []

        # initialize our lists of detected bounding boxes, confidences
        # and class IDs for every grabbed frame
        boxes = []
        confidences = []
        classIDs = []

        # use output of ANN
        for output in layerOutputs:
            for detection in output:

                # calculate highest score and get it`s confidence number
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]

                # if confidence is higher than selected value of
                # CONFIDENCE_FILTER create bounding box for every detection
                if confidence > self.CONFIDENCE_FILTER:
                    # print(detection[0:4])
                    # print("detection: ", detection, ", conf: ", confidence)
                    box = detection[0:4] * np.array(
                        [imageDimensions[1], imageDimensions[0],
                         imageDimensions[1], imageDimensions[0]])
                    (centerX, centerY, width, height) = box.astype("int")

                    # get left corner coordinates of bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(class_id)

            # apply non-maxima suppression to overlapping bounding boxes with low confidence
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_FILTER,
                                    self.THRESHOLD)

            # check if any bounding box exists
            if len(idxs) > 0:

                # save bounding boxes
                for i in idxs.flatten():
                    # print("box: :", boxes[i])
                    # get the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    detectedObject = {
                        "x": int(boxes[i][0]),
                        "y": int(boxes[i][1]),
                        "w": int(boxes[i][2]),
                        "h": int(boxes[i][3]),
                        "center": (int(x + w / 2), int(y + h / 2)),
                        "color": tuple([int(c) for c in self.colors[classIDs[i]]]),
                        "label": self.labels[classIDs[i]],
                        "confidence": confidences[i]
                    }

                    outputs.append(detectedObject)

        outputs = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in outputs)]

        return outputs

    @staticmethod
    def make_plots(detections_results, w, h, show=True, save=True):
        """
           Create output plots.

           :param detections_results: output detections from the video stream
           :param w: width of the frame
           :param h: height of the frame
           :param save: save flag

           :return: None

       """
        print("#########  BARBELL RESULTS  #########")
        print(detections_results)
        print("#####################################")

        white = np.zeros([h, w, 3], dtype=np.uint8)
        white.fill(0)

        for detections in detections_results:
            for detection in detections:
                print("detection: ", detection["results"])
                print("detection[1]: ", detection["results"][1])

                print()
                cv2.circle(white, detection["results"][1], 3, (115, 110, 60), -1)

        t_barbell = []
        x_barbell = []
        y_barbell = []
        t_foot = []
        x_foot = []
        y_foot = []
        for detections in detections_results:
            for detection in detections:
                if detection["label"] == "barbell":
                    t_barbell.append(detection["results"][0])
                    x_barbell.append(detection["results"][1][0])
                    y_barbell.append(detection["results"][1][1])

                if detection["label"] == "foot":
                    t_foot.append(detection["results"][0])
                    x_foot.append(detection["results"][1][0])
                    y_foot.append(detection["results"][1][1])

        # print("t: ", t)
        # print("len(t): ", len(t))
        # print("x: ", x)
        # print("len(x): ", len(x))
        # print("y: ", y)
        # print("len(y): ", len(y))

        fig, axs = plt.subplots(2)
        fig.suptitle('Pozycja sztangi')
        axs[0].scatter(t_barbell, x_barbell, c='b', marker='x', label='barbell')
        axs[0].scatter(t_foot, x_foot, c='r', marker='s', label='foot')
        plt.legend(loc='upper left')

        axs[1].scatter(t_barbell, y_barbell, c='b', marker='x', label='barbell')
        axs[1].scatter(t_foot, y_foot, c='r', marker='s', label='foot')
        plt.legend(loc='upper left')


        plt.gca().invert_yaxis()
        if save:
            plt.savefig('Result_plot.png')
            cv2.imwrite('Result.png', white)

        if show:
            cv2.imshow('Result', white)
            plt.show()
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()  # destroy all opened windows
