from dataclasses import dataclass


@dataclass
class FrameMetrics:
    """Angles and ratios computed from one video frame's joint positions."""

    frame_index: int
    time_s: float
    knee_angle: float
    hip_angle: float
    torso_lean: float
    # Normalized by thigh length (hip-to-knee pixel distance) so it holds up
    # across camera distances/resolutions.
    knee_over_toe_ratio: float
    # Raw heel position + the same normalizing ruler, kept per-frame rather
    # than pre-reduced to a ratio: heel rise needs a "heel on the ground"
    # reference, and that has to be established locally per rep (see
    # feedback.evaluate_rep), not once globally for the whole video.
    heel_y: float
    thigh_length: float