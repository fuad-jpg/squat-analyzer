from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np

import config
from angles import facing_direction, hip_angle, knee_angle, torso_lean_from_vertical
from feedback import RepReport, evaluate_rep
from metrics import FrameMetrics
from plate_detector import detect_plate, suppress_joints_inside_plate
from pose_estimator import PoseEstimator
from rep_counter import RepCounter
from smoothing import JointSmoother
from visibility_gate import JointVisibilityGate

JOINT_CONNECTIONS = [
    ("nose", "shoulder"),
    ("shoulder", "hip"),
    ("hip", "knee"),
    ("knee", "ankle"),
    ("ankle", "heel"),
    ("heel", "foot_index"),
    ("ankle", "foot_index"),
]


@dataclass
class AnalysisResult:
    fps: float
    frame_count: int
    duration_s: float
    reps: List[RepReport] = field(default_factory=list)
    frame_metrics: List[FrameMetrics] = field(default_factory=list)
    output_video_path: Optional[str] = None


# BGR colors, chosen for contrast against skin/gym backgrounds.
_SKELETON_COLOR = (255, 200, 30)     # vivid azure
_JOINT_FILL_COLOR = (40, 170, 255)   # warm orange
_JOINT_RING_COLOR = (255, 255, 255)
_REP_TEXT_COLOR = (60, 230, 255)     # gold
_ANGLE_TEXT_COLOR = (255, 255, 255)
_TEXT_OUTLINE_COLOR = (0, 0, 0)


def _put_text_with_outline(frame, text, org, color, scale=0.6, thickness=2):
    # Draw a thick black copy first, then the real text on top, so labels
    # stay readable against any background color behind them.
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                _TEXT_OUTLINE_COLOR, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _draw_overlay(frame, joints, fm: FrameMetrics, rep_count: int):
    for a, b in JOINT_CONNECTIONS:
        pa = tuple(int(v) for v in joints[a])
        pb = tuple(int(v) for v in joints[b])
        cv2.line(frame, pa, pb, _SKELETON_COLOR, 3, cv2.LINE_AA)
    for x, y in joints.values():
        center = (int(x), int(y))
        cv2.circle(frame, center, 7, _JOINT_RING_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 5, _JOINT_FILL_COLOR, -1, cv2.LINE_AA)

    knee_pt = tuple(int(v) for v in joints["knee"])
    hip_pt = tuple(int(v) for v in joints["hip"])
    _put_text_with_outline(frame, f"Knee {fm.knee_angle:.0f}",
                            (knee_pt[0] + 12, knee_pt[1]), _ANGLE_TEXT_COLOR)
    _put_text_with_outline(frame, f"Hip {fm.hip_angle:.0f}",
                            (hip_pt[0] + 12, hip_pt[1]), _ANGLE_TEXT_COLOR)

    # Semi-transparent badge behind the rep counter so it stays legible.
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (170, 56), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    _put_text_with_outline(frame, f"Reps: {rep_count}", (26, 44),
                            _REP_TEXT_COLOR, scale=0.9, thickness=2)
    return frame


def analyze_video(
    input_path: str,
    output_path: Optional[str] = None,
    min_visibility: float = config.MIN_VISIBILITY,
    progress_callback=None,
) -> AnalysisResult:
    """Run the full pipeline on a side-view squat video: pose tracking, angle
    math, rep counting, and per-rep form feedback. Optionally writes an
    annotated copy of the video with a skeleton overlay and rep counter
    (form feedback itself is returned as text, not drawn on the video).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    # Phone videos filmed in portrait are usually stored as landscape pixels
    # with a rotation flag telling players to display them rotated. Without
    # this, OpenCV ignores that flag and hands back raw sideways frames.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        # OpenCV's own VideoWriter can't reliably encode H.264 on most
        # platforms, and browsers (including Streamlit's st.video) won't
        # play back OpenCV's usual "mp4v" codec. imageio + its bundled
        # ffmpeg binary writes real H.264, so the output actually plays.
        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=8, macro_block_size=None
        )

    estimator = PoseEstimator()
    counter = RepCounter()
    smoother = JointSmoother()
    visibility_gate = JointVisibilityGate(min_visibility)

    side = None
    frame_metrics: List[FrameMetrics] = []
    reps: List[RepReport] = []
    frame_index = 0
    # Carried across frames so the overlay keeps showing the last known pose
    # instead of flickering off on any single low-confidence/undetected frame.
    last_joints = None
    last_fm: Optional[FrameMetrics] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Read the actual frame's own dimensions rather than trusting
            # cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT): those reflect the raw
            # stream, not what ORIENTATION_AUTO rotates frames into.
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * (1000.0 / fps))
            results = estimator.process(rgb, timestamp_ms)
            points = PoseEstimator.get_landmark_points(results, width, height)

            if points is not None:
                if side is None:
                    side = PoseEstimator.pick_side(points)

                if PoseEstimator.get_side_visibility(points, side) >= min_visibility:
                    raw_joints = PoseEstimator.get_side_joints(points, side)
                    joint_visibility = PoseEstimator.get_side_joint_visibility(points, side)

                    # MediaPipe can report a landmark it's covered by a plate
                    # as confidently visible anyway -- this catches that with
                    # an independent geometric check instead of trusting its
                    # own confidence score.
                    plate = detect_plate(frame)
                    joint_visibility = suppress_joints_inside_plate(
                        raw_joints, joint_visibility, plate
                    )

                    raw_joints, joint_visibility = PoseEstimator.apply_nose_fallback_for_shoulder(
                        raw_joints, joint_visibility, min_visibility
                    )
                    trusted_joints = visibility_gate.filter(raw_joints, joint_visibility)
                    joints = smoother.smooth(trusted_joints)

                    k_angle = knee_angle(joints["hip"], joints["knee"], joints["ankle"])
                    h_angle = hip_angle(joints["shoulder"], joints["hip"], joints["knee"])
                    lean = torso_lean_from_vertical(joints["shoulder"], joints["hip"])
                    facing = facing_direction(joints["heel"], joints["foot_index"])

                    thigh_length = float(np.linalg.norm(
                        np.array(joints["hip"]) - np.array(joints["knee"])
                    )) + 1e-6
                    knee_over_toe_ratio = (
                        (joints["knee"][0] - joints["foot_index"][0]) * facing / thigh_length
                    )

                    fm = FrameMetrics(
                        frame_index=frame_index,
                        time_s=frame_index / fps,
                        knee_angle=k_angle,
                        hip_angle=h_angle,
                        torso_lean=lean,
                        knee_over_toe_ratio=knee_over_toe_ratio,
                        heel_y=joints["heel"][1],
                        thigh_length=thigh_length,
                    )
                    frame_metrics.append(fm)

                    closed = counter.update(fm)
                    if closed is not None:
                        reps.append(evaluate_rep(len(reps) + 1, closed.frames))

                    last_joints, last_fm = joints, fm

            if writer is not None:
                if last_joints is not None:
                    frame = _draw_overlay(frame, last_joints, last_fm, len(reps))
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_index += 1
            if progress_callback and frame_count:
                progress_callback(frame_index / frame_count)

        closed = counter.finalize()
        if closed is not None:
            reps.append(evaluate_rep(len(reps) + 1, closed.frames))

    finally:
        cap.release()
        if writer is not None:
            writer.close()
        estimator.close()

    return AnalysisResult(
        fps=fps,
        frame_count=frame_count,
        duration_s=(frame_count / fps) if fps else 0.0,
        reps=reps,
        frame_metrics=frame_metrics,
        output_video_path=output_path,
    )
