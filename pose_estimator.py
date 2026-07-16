import os
import urllib.request

import mediapipe as mp
import numpy as np

#MediaPipe wrapper for pose estimation.
# MediaPipe removed the old mp.solutions.pose API in favor of the Tasks API
# (mediapipe>=0.10.x no longer ships mp.solutions at all). This wraps the
# PoseLandmarker task, downloading its model file on first use.
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_full.task")

# Only the joints needed for side-view squat analysis. "nose" has no
# left/right distinction (it's a midline point) -- it's paired with itself
# so it flows through the same per-side plumbing as everything else below.
LANDMARK_NAMES = {
    "shoulder": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    "hip": ("LEFT_HIP", "RIGHT_HIP"),
    "knee": ("LEFT_KNEE", "RIGHT_KNEE"),
    "ankle": ("LEFT_ANKLE", "RIGHT_ANKLE"),
    "heel": ("LEFT_HEEL", "RIGHT_HEEL"),
    "foot_index": ("LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"),
    "nose": ("NOSE", "NOSE"),
}

# Landmark indices for the 33-point BlazePose model (stable across MediaPipe's
# old Solutions API and the current Tasks API -- same underlying model).
_NAME_TO_INDEX = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
    "NOSE": 0,
}


def _ensure_model_downloaded():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print(f"Downloading pose landmark model to {_MODEL_PATH} (~9 MB, one-time)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


class PoseEstimator:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        _ensure_model_downloaded()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            # Detect a few people, not just one -- lets get_landmark_points
            # pick whichever is the actual lifter (largest in frame) instead
            # of MediaPipe silently locking onto someone in the background.
            num_poses=4,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def process(self, frame_rgb, timestamp_ms):
        """Run pose detection on one RGB frame. `timestamp_ms` must increase
        monotonically across calls (e.g. frame_index * 1000 / fps) -- the
        VIDEO running mode uses it for internal tracking. Returns the raw
        PoseLandmarkerResult."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

    def close(self):
        self._landmarker.close()

    @staticmethod
    def _main_subject(pose_landmarks_list):
        """When multiple people are detected, pick whichever one occupies the
        largest area in frame (bounding box over their own landmarks). The
        actual lifter is the main focal point and closest to the camera, so
        they read as bigger than anyone else caught in the background."""
        def bbox_area(landmarks):
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        return max(pose_landmarks_list, key=bbox_area)

    @staticmethod
    def get_landmark_points(results, frame_width, frame_height):
        """Convert normalized landmarks to pixel coords + visibility, indexed by name."""
        if not results.pose_landmarks:
            return None

        landmarks = PoseEstimator._main_subject(results.pose_landmarks)
        points = {}
        for name, index in _NAME_TO_INDEX.items():
            lm = landmarks[index]
            points[name] = {
                "x": lm.x * frame_width,
                "y": lm.y * frame_height,
                "visibility": lm.visibility,
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

    @staticmethod
    def get_side_joint_visibility(points, side):
        """Return {joint_name: visibility} for the chosen side.

        Unlike get_side_visibility (one number averaged across all 6 joints),
        this exposes each joint individually -- needed because a plate or
        prop covering just the hip or shoulder can pull that one landmark far
        off while the other 5 stay perfectly visible, keeping the average
        comfortably above threshold and hiding the problem.
        """
        vis = {}
        for joint_name, (left_name, right_name) in LANDMARK_NAMES.items():
            name = left_name if side == "LEFT" else right_name
            vis[joint_name] = points[name]["visibility"]
        return vis

    @staticmethod
    def apply_nose_fallback_for_shoulder(joints, visibility, min_visibility):
        """If the shoulder's own visibility is too low to trust but the
        nose's isn't, use the nose position in the shoulder's place for this
        frame. A barbell plate sits at bar height and can cover the shoulder
        while sitting well below the head -- the nose is usually still
        clearly visible when the shoulder isn't. Returns new dicts; doesn't
        mutate the inputs.
        """
        joints = dict(joints)
        visibility = dict(visibility)
        shoulder_vis = visibility.get("shoulder", 1.0)
        nose_vis = visibility.get("nose", 0.0)
        if shoulder_vis < min_visibility and nose_vis >= min_visibility:
            joints["shoulder"] = joints["nose"]
            visibility["shoulder"] = nose_vis
        return joints, visibility
