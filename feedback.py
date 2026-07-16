from dataclasses import dataclass, field
from typing import List

import config


def _max_sustained(values, window=3):
    """Max value that holds for at least `window` consecutive samples.

    Filters out a single- or two-frame spike (e.g. MediaPipe momentarily
    misjudging the heel's position where a shoe blends into the floor)
    while still catching a genuine change: a real heel lift typically
    holds for several consecutive frames, not just one.
    """
    values = list(values)
    if len(values) < window:
        return min(values) if values else 0.0
    return max(min(values[i:i + window]) for i in range(len(values) - window + 1))


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
    is_good_form: bool
    feedback: List[str] = field(default_factory=list)


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

    # "Heel on the ground" has to be a local reference, not one baseline
    # shared across the whole video: even a small shift in where the lifter
    # is standing between reps (re-racking, adjusting stance) moves the
    # absolute pixel position of "ground" without the heel actually lifting.
    # frames[0] is right as this specific rep starts descending -- the knee
    # has only just crossed the descent trigger, so the heel is still flat.
    standing_heel_y = frames[0].heel_y
    heel_rise_ratios = [
        max(0.0, (standing_heel_y - f.heel_y) / f.thigh_length) for f in frames
    ]
    max_heel_rise = _max_sustained(heel_rise_ratios)

    fb = []

    if bottom.knee_angle > config.PARALLEL_KNEE_ANGLE + config.SHALLOW_KNEE_ANGLE_MARGIN:
        fb.append(
            f"**Depth: didn't quite get low enough.** Your knee angle only reached "
            f"{bottom.knee_angle:.0f}° at the bottom of the rep. Knee angle is the angle "
            f"your leg bends to at the knee -- 180° is standing fully straight, and smaller "
            f"numbers mean a deeper bend. A common depth target is thighs parallel to the "
            f"floor, which works out to roughly {config.PARALLEL_KNEE_ANGLE:.0f}° or less. "
            f"To get there, try sitting your hips further down and back, as if sitting into "
            f"a low chair, until the crease of your hip drops to about the same height as "
            f"the top of your knee."
        )

    if max_lean > config.EXCESSIVE_TORSO_LEAN_DEG:
        fb.append(
            f"**Torso: leaning too far forward.** Near the bottom of the rep your torso was "
            f"about {max_lean:.0f}° away from perfectly upright (0° would be standing bolt "
            f"straight). Some forward lean is normal and even necessary in a squat, but "
            f"past a certain point ({config.EXCESSIVE_TORSO_LEAN_DEG:.0f}°+) too much of the "
            f"weight shifts off your legs and onto your lower back, which is both less "
            f"efficient and puts more strain on your spine. Try taking a bigger breath and "
            f"tightening your core ('bracing') before you descend, and check whether your "
            f"hips or ankles are tight -- limited mobility there often forces extra forward lean."
        )

    if max_heel_rise > config.HEEL_RISE_RATIO_LIMIT:
        fb.append(
            f"**Feet: heels lifting off the ground.** Near the bottom of the rep your heels "
            f"rose up off the floor (your knees also drifted about {max_over_toe:.1f}x a "
            f"thigh-length past your toes at the same moment). Heels lifting means your "
            f"weight rolled forward onto the balls of your feet instead of staying spread "
            f"across your whole foot -- that makes you less stable and can put extra strain "
            f"on your knees. It usually comes down to limited ankle flexibility. Two common "
            f"fixes: squat in shoes with a small raised heel (or put a thin plate under each "
            f"heel), or spend time on ankle-mobility stretches so you can sit deep without "
            f"your heels needing to come up."
        )

    is_good_form = not fb
    if is_good_form:
        fb.append(
            "**Good rep.** You hit solid depth (knees bent well past parallel), stayed "
            "upright without excessive forward lean, and kept your feet flat and stable "
            "the whole way down and up. This is the form to repeat."
        )

    return RepReport(
        rep_number=rep_number,
        start_frame=frames[0].frame_index,
        end_frame=frames[-1].frame_index,
        bottom_frame=bottom.frame_index,
        min_knee_angle=bottom.knee_angle,
        max_torso_lean=max_lean,
        max_knee_over_toe_ratio=max_over_toe,
        max_heel_rise_ratio=max_heel_rise,
        is_good_form=is_good_form,
        feedback=fb,
    )
