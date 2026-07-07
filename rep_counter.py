from dataclasses import dataclass, field
from typing import List, Optional

from config import (
    DESCENT_TRIGGER_KNEE_ANGLE,
    MIN_KNEE_DROP_FOR_VALID_REP,
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
    small weight shift being mistaken for a squat.
    """

    def __init__(self):
        self.state = "STANDING"
        self._current: Optional[RepFrames] = None
        self._standing_knee_angle = STANDING_KNEE_ANGLE
        self.completed_reps: List[RepFrames] = []

    def update(self, frame_metrics) -> Optional[RepFrames]:
        angle = frame_metrics.knee_angle
        closed_rep = None

        if self.state == "STANDING":
            if angle >= STANDING_KNEE_ANGLE:
                self._standing_knee_angle = max(self._standing_knee_angle, angle)
            if angle < DESCENT_TRIGGER_KNEE_ANGLE:
                self.state = "IN_REP"
                self._current = RepFrames(start_frame=frame_metrics.frame_index)
                self._current.frames.append(frame_metrics)

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