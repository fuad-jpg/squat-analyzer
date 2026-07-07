import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# Only the joints needed for side-view squat analysis.
LANDMARK_NAMES = {
    "shoulder": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    "hip": ("LEFT_HIP", "RIGHT_HIP"),
    "knee": ("LEFT_KNEE", "RIGHT_KNEE"),
    "ankle": ("LEFT_ANKLE", "RIGHT_ANKLE"),
    "heel": ("LEFT_HEEL", "RIGHT_HEEL"),
    "foot_index": ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"),
}


class PoseEstimator:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb):
        """Run pose detection on one RGB frame. Returns raw MediaPipe results."""
        return self._pose.process(frame_rgb)

    def close(self):
        self._pose.close()

    @staticmethod
    def get_landmark_points(results, frame_width, frame_height):
        """Convert normalized landmarks to pixel coords + visibility, indexed by name."""
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        points = {}
        for lm in mp_pose.PoseLandmark:
            point = landmarks[lm.value]
            points[lm.name] = {
                "x": point.x * frame_width,
                "y": point.y * frame_height,
                "visibility": point.visibility,
            }
        return points

    @staticmethod
    def pick_side(points):
        """Pick left or right side based on average visibility (in a side-view
        video, the far side is usually partially occluded)."""
        left_avg = np.mean([points[left]["visibility"] for left, _ in LANDMARK_NAMES.values()])
        right_avg = np.mean([points[right]["visibility"] for _, right in LANDMARK_NAMES.values()])
        return "LEFT" if left_avg >= right_avg else "RIGHT"

    @staticmethod
    def get_side_visibility(points, side):
        """Average MediaPipe visibility for the chosen side's tracked joints."""
        vis = [points[left if side == "LEFT" else right]["visibility"]
               for left, right in LANDMARK_NAMES.values()]
        return float(np.mean(vis))

    @staticmethod
    def get_side_joints(points, side):
        """Return {joint_name: (x, y)} for the chosen side."""
        joints = {}
        for joint_name, (left_name, right_name) in LANDMARK_NAMES.items():
            name = left_name if side == "LEFT" else right_name
            p = points[name]
            joints[joint_name] = (p["x"], p["y"])
        return joints
