from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    DESCENT_TRIGGER_KNEE_ANGLE,
    MIN_KNEE_DROP_FOR_VALID_REP,
    MIN_STANDING_DURATION_S,
    STANDING_KNEE_ANGLE,
)


@dataclass
class RepFrames:
    """Raw per-frame metrics accumulated for a single rep, start to finish."""

    start_frame: int
    frames: list = field(default_factory=list)


class RepCounter:
    """Hysteresis state machine over knee angle:

        STANDING --(knee angle drops below DESCENT_TRIGGER)--> IN_REP
        IN_REP --(knee angle rises back near the standing baseline)--> STANDING (rep closed)

    A rep is only kept if the knee angle dropped far enough below the
    standing baseline (MIN_KNEE_DROP_FOR_VALID_REP) to rule out noise or a
    small weight shift being mistaken for a squat, AND the lifter was
    genuinely standing still for at least MIN_STANDING_DURATION_S right
    before the descent started (rules out a video opening mid-motion, or a
    quick bend while unracking, ever being treated as a rep attempt at all).
    """

    def __init__(self, fps: float = 30.0):
        self.state = "STANDING"
        self._current: Optional[RepFrames] = None
        self._standing_knee_angle = STANDING_KNEE_ANGLE
        self.completed_reps: List[RepFrames] = []
        self._consecutive_standing_frames = 0
        self._min_standing_frames = max(1, round(fps * MIN_STANDING_DURATION_S))

    def update(self, frame_metrics) -> Optional[RepFrames]:
        angle = frame_metrics.knee_angle
        closed_rep = None

        if self.state == "STANDING":
            # Captured before this frame updates the streak: the descent
            # trigger below needs to know how long the lifter was ALREADY
            # standing before this frame, not whether this frame itself
            # (which may be the very frame the knee drops on) still counts.
            standing_streak_before_this_frame = self._consecutive_standing_frames

            if angle >= STANDING_KNEE_ANGLE:
                self._standing_knee_angle = max(self._standing_knee_angle, angle)
                self._consecutive_standing_frames += 1
            # Otherwise leave the streak as-is (don't reset it): a real
            # descent passes gradually through the gap between
            # DESCENT_TRIGGER_KNEE_ANGLE and STANDING_KNEE_ANGLE over many
            # frames, and resetting here would always zero the streak out
            # before the entry check below ever sees it -- which used to
            # make it impossible for ANY descent to pass this check,
            # regardless of MIN_STANDING_DURATION_S. It only needs to reset
            # when a rep actually starts (below) or closes.

            if (angle < DESCENT_TRIGGER_KNEE_ANGLE
                    and standing_streak_before_this_frame >= self._min_standing_frames):
                self.state = "IN_REP"
                self._current = RepFrames(start_frame=frame_metrics.frame_index)
                self._current.frames.append(frame_metrics)
                self._consecutive_standing_frames = 0

        elif self.state == "IN_REP":
            self._current.frames.append(frame_metrics)
            if angle >= self._standing_knee_angle - 5:
                min_angle = min(f.knee_angle for f in self._current.frames)
                if self._standing_knee_angle - min_angle >= MIN_KNEE_DROP_FOR_VALID_REP:
                    self.completed_reps.append(self._current)
                    closed_rep = self._current
                self._current = None
                self.state = "STANDING"

        return closed_rep

    def finalize(self) -> Optional[RepFrames]:
        """Call once after the last frame, in case the video ends mid-rep."""
        if self.state == "IN_REP" and self._current and len(self._current.frames) > 1:
            min_angle = min(f.knee_angle for f in self._current.frames)
            if self._standing_knee_angle - min_angle >= MIN_KNEE_DROP_FOR_VALID_REP:
                self.completed_reps.append(self._current)
                return self._current
        return None