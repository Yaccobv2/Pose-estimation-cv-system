"""
Pose detection module created with mediapipe
"""
import math
import cv2
import mediapipe as mp



class PoseDetector:
    """
    Pose detection class.
    """

    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """Initializes PoseDetector object.

            Args:
              static_image_mode: Whether to treat the input images as a batch of static
                and possibly unrelated images, or a video stream. See details in
                https://solutions.mediapipe.dev/pose#static_image_mode.
              model_complexity: Complexity of the pose landmark model: 0, 1 or 2. See
                details in https://solutions.mediapipe.dev/pose#model_complexity.
              smooth_landmarks: Whether to filter landmarks across different input
                images to reduce jitter. See details in
                https://solutions.mediapipe.dev/pose#smooth_landmarks.
              enable_segmentation: Whether to predict segmentation mask. See details in
                https://solutions.mediapipe.dev/pose#enable_segmentation.
              smooth_segmentation: Whether to filter segmentation across different input
                images to reduce jitter. See details in
                https://solutions.mediapipe.dev/pose#smooth_segmentation.
              min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person
                detection to be considered successful. See details in
                https://solutions.mediapipe.dev/pose#min_detection_confidence.
              min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                pose landmarks to be considered tracked successfully. See details in
                https://solutions.mediapipe.dev/pose#min_tracking_confidence.
            """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode, self.model_complexity,
                                     self.smooth_landmarks, self.enable_segmentation,
                                     self.smooth_segmentation, self.min_detection_confidence,
                                     self.min_tracking_confidence)

        self.results = None

    def find_pose(self, img, draw=True):
        """Find joint detection

            Args:
                img: frame to process
                draw: Whether to draw detected points on given frame

            return:
                img: processed frame
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if draw:
            if self.results.pose_landmarks:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS)

        return img

    def get_pixel_positions(self, img, draw=True):
        """Find joint detection

            Args:
                img: frame to process

            return:
                img: processed frame
        """
        lm_list = []
        if self.results.pose_landmarks:
            for lm_id, l_m in enumerate(self.results.pose_landmarks.landmark):
                w_h, w_w, _w_c = img.shape
                p_x, p_y = int(l_m.x * w_w), int(l_m.y * w_h)
                lm_list.append([lm_id, p_x, p_y])

                if draw:
                    #text = str(p_x) + ' , ' + str(p_y)
                    #cv2.putText(img, text, (p_x, p_y), cv2.FONT_HERSHEY_PLAIN,
                    #            1, (115, 255, 127), 2)

                    cv2.putText(img, str(lm_id), (p_x, p_y+20), cv2.FONT_HERSHEY_PLAIN,
                                1, (0, 0, 255), 2)

                #print(lm_list)
        return lm_list

    def get_angle(self, img, lm_list, set_of_points):
        """Find joint detection
                Args:
                    img: frame to process
                    lm_list: list of all points found by neural network
                    set_of_points: set of 3 points that define joint

                return:
                    angle in degrees
        """
        coords = []
        for point in set_of_points:
            for position_x in lm_list:
                if position_x[0] == point:
                    coords.append(position_x)

        if len(coords) == 3:
            angle = self.measure_angle(coords)

            cv2.putText(img, str(int(angle)), (coords[1][1], coords[1][2] -35),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

            return angle, coords

        return None

    @staticmethod
    def measure_angle(points):
        """Find joint detection
                Args:
                    points: 3 points between which the angle will be measured

                return:
                    angle in degrees
        """
        a_vector = [points[1][2]-points[0][2], points[1][1]-points[0][1]]
        b_vector = [points[1][2]-points[2][2], points[1][1]-points[2][1]]
        a_length = math.sqrt(a_vector[0]**2 + a_vector[1]**2)
        b_length = math.sqrt(b_vector[0]**2 + b_vector[1]**2)
        if(a_length*b_length) != 0:
            return math.degrees(math.acos((a_vector[0]*b_vector[0] + a_vector[1]*b_vector[1])
                                          / (a_length*b_length)))

        return None

    def get_all_angles(self, img, lm_list, set_of_joints):
        """Find joint detection
                Args:
                    img: frame to process
                    lm_list: list of all points found by neural network
                    set_of_joints: list of all joints (defined by 3 points)
                                    where the angle will be measured

                return:
                    ret: angles and coordinates of all joints in set_of_joints
        """
        ret = []
        for joint in set_of_joints:
            temp_angle, temp_coords = self.get_angle(img, lm_list, joint)
            ret.append ([temp_angle, temp_coords])

        return ret
