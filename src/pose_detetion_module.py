"""
Pose detection module created with mediapipe
"""
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
                    text = str(p_x) + ' , ' + str(p_y)
                    cv2.putText(img, text, (p_x, p_y), cv2.FONT_HERSHEY_PLAIN,
                                1, (115, 255, 127), 2)
        return lm_list
