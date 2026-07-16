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
            if visibility[name] >= self._min_visibility:
                trusted[name] = pos
                self._last_good[name] = pos
            elif name in self._last_good:
                trusted[name] = self._last_good[name]
            else:
                # Nothing trustworthy recorded yet (e.g. occluded since the
                # very first frame) -- pass the current reading through
                # since there's no better option, but don't cache it as
                # "good". Caching it would anchor later frames to a bad
                # position and cause a jump/snap once a real good reading
                # finally arrives and overwrites it.
                trusted[name] = pos
        return trusted
