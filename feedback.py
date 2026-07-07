from dataclasses import dataclass, field
from typing import List

import config


@dataclass
class RepReport:
    rep_number: int
    start_frame: int
    end_frame: int
    bottom_frame: int
    min_knee_angle: float
    max_torso_lean: float
    max_knee_over_toe_ratio: float
    max_heel_rise_ratio: float
    feedback: List[str] = field(default_factory=list)

    @property
    def is_good_form(self) -> bool:
        return self.feedback == ["Good rep: solid depth, upright torso, and stable feet."]


def evaluate_rep(rep_number: int, frames) -> RepReport:
    """Apply simple, well-agreed-upon form rules to one completed rep's frames.

    Deliberately does NOT flag "knee past toe" on its own -- some forward
    knee travel is normal and often necessary (especially high-bar/front
    squats, or taller lifters). It's only mentioned as context when heels
    are also lifting, which is the actual sign of weight shifting forward.
    """
    bottom = min(frames, key=lambda f: f.knee_angle)
    max_lean = max(f.torso_lean for f in frames)
    max_over_toe = max(f.knee_over_toe_ratio for f in frames)
    max_heel_rise = max(f.heel_rise_ratio for f in frames)

    fb = []

    if bottom.knee_angle > config.PARALLEL_KNEE_ANGLE + config.SHALLOW_KNEE_ANGLE_MARGIN:
        fb.append(
            f"Didn't quite hit depth: knee angle only reached {bottom.knee_angle:.0f}°. "
            f"Aim for thighs at least parallel to the floor (roughly "
            f"{config.PARALLEL_KNEE_ANGLE:.0f}° or less)."
        )

    if max_lean > config.EXCESSIVE_TORSO_LEAN_DEG:
        fb.append(
            f"Torso leans forward about {max_lean:.0f}° from vertical near the bottom. "
            "Brace your core and try to keep your chest up through the descent."
        )

    if max_heel_rise > config.HEEL_RISE_RATIO_LIMIT:
        fb.append(
            "Heels lift off the floor near the bottom, shifting weight onto your toes "
            f"(knees also travel {max_over_toe:.1f}x a thigh-length past your toes at the "
            "deepest point). Keep weight through the whole foot, or check ankle mobility."
        )

    if not fb:
        fb.append("Good rep: solid depth, upright torso, and stable feet.")

    return RepReport(
        rep_number=rep_number,
        start_frame=frames[0].frame_index,
        end_frame=frames[-1].frame_index,
        bottom_frame=bottom.frame_index,
        min_knee_angle=bottom.knee_angle,
        max_torso_lean=max_lean,
        max_knee_over_toe_ratio=max_over_toe,
        max_heel_rise_ratio=max_heel_rise,
        feedback=fb,
    )