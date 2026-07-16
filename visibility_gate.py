class JointVisibilityGate:
    """Holds each joint at its last trusted position until its own MediaPipe
    visibility score clears the threshold again.

    A plate or other prop can occlude just one joint (commonly the hip or
    shoulder, in a side-view squat) while every other joint stays perfectly
    visible. A single per-frame gate averaged across all joints lets that one
    bad joint hide behind the other good ones instead of being caught --
    this checks each joint against its own score instead.
    """

    def __init__(self, min_visibility: float):
        self._min_visibility = min_visibility
        self._last_good: dict = {}

    def filter(self, joints: dict, visibility: dict) -> dict:
        trusted = {}
        for name, pos in joints.items():
            if visibility[name] >= self._min_visibility or name not in self._last_good:
                trusted[name] = pos
                self._last_good[name] = pos
            else:
                trusted[name] = self._last_good[name]
        return trusted
