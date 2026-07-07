from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

import config
from angles import facing_direction, hip_angle, knee_angle, torso_lean_from_vertical
from feedback import RepReport, evaluate_rep
from metrics import FrameMetrics
from pose_estimator import PoseEstimator
from rep_counter import RepCounter

JOINT_CONNECTIONS = [
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


def _draw_overlay(frame, joints, fm: FrameMetrics, rep_count: int, last_feedback: List[str]):
    for a, b in JOINT_CONNECTIONS:
        pa = tuple(int(v) for v in joints[a])
        pb = tuple(int(v) for v in joints[b])
        cv2.line(frame, pa, pb, (0, 255, 0), 2)
    for x, y in joints.values():
        cv2.circle(frame, (int(x), int(y)), 5, (0, 140, 255), -1)

    knee_pt = tuple(int(v) for v in joints["knee"])
    hip_pt = tuple(int(v) for v in joints["hip"])
    cv2.putText(frame, f"knee {fm.knee_angle:.0f}", (knee_pt[0] + 10, knee_pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"hip {fm.hip_angle:.0f}", (hip_pt[0] + 10, hip_pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {rep_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if last_feedback:
        y0 = frame.shape[0] - 20 * len(last_feedback) - 10
        for i, line in enumerate(last_feedback):
            cv2.putText(frame, line[:90], (20, y0 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


def analyze_video(
    input_path: str,
    output_path: Optional[str] = None,
    min_visibility: float = config.MIN_VISIBILITY,
    progress_callback=None,
) -> AnalysisResult:
    """Run the full pipeline on a side-view squat video: pose tracking, angle
    math, rep counting, and per-rep form feedback. Optionally writes an
    annotated copy of the video with skeleton overlay and live feedback.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    estimator = PoseEstimator()
    counter = RepCounter()

    side = None
    standing_heel_y = None
    frame_metrics: List[FrameMetrics] = []
    reps: List[RepReport] = []
    last_feedback: List[str] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = estimator.process(rgb)
            points = PoseEstimator.get_landmark_points(results, width, height)

            if points is not None:
                if side is None:
                    side = PoseEstimator.pick_side(points)

                if PoseEstimator.get_side_visibility(points, side) >= min_visibility:
                    joints = PoseEstimator.get_side_joints(points, side)

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

                    if k_angle >= config.STANDING_KNEE_ANGLE:
                        standing_heel_y = (
                            joints["heel"][1] if standing_heel_y is None
                            else max(standing_heel_y, joints["heel"][1])
                        )
                    if standing_heel_y is None:
                        standing_heel_y = joints["heel"][1]
                    heel_rise_ratio = max(
                        0.0, (standing_heel_y - joints["heel"][1]) / thigh_length
                    )

                    fm = FrameMetrics(
                        frame_index=frame_index,
                        time_s=frame_index / fps,
                        knee_angle=k_angle,
                        hip_angle=h_angle,
                        torso_lean=lean,
                        knee_over_toe_ratio=knee_over_toe_ratio,
                        heel_rise_ratio=heel_rise_ratio,
                    )
                    frame_metrics.append(fm)

                    closed = counter.update(fm)
                    if closed is not None:
                        report = evaluate_rep(len(reps) + 1, closed.frames)
                        reps.append(report)
                        last_feedback = report.feedback

                    if writer is not None:
                        frame = _draw_overlay(frame, joints, fm, len(reps), last_feedback)

            if writer is not None:
                writer.write(frame)

            frame_index += 1
            if progress_callback and frame_count:
                progress_callback(frame_index / frame_count)

        closed = counter.finalize()
        if closed is not None:
            reps.append(evaluate_rep(len(reps) + 1, closed.frames))

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        estimator.close()

    return AnalysisResult(
        fps=fps,
        frame_count=frame_count,
        duration_s=(frame_count / fps) if fps else 0.0,
        reps=reps,
        frame_metrics=frame_metrics,
        output_video_path=output_path,
    )
