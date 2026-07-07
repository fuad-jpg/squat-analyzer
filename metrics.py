from dataclasses import dataclass


@dataclass
class FrameMetrics:
    """Angles and ratios computed from one video frame's joint positions."""

    frame_index: int
    time_s: float
    knee_angle: float
    hip_angle: float
    torso_lean: float
    # Ratios below are normalized by thigh length (hip-to-knee pixel
    # distance) so they hold up across camera distances/resolutions.
    knee_over_toe_ratio: float
    heel_rise_ratio: float