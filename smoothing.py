import statistics
from collections import defaultdict, deque


class JointSmoother:
    """Rolling median filter over each joint's (x, y) position.

    MediaPipe can report high confidence for a landmark while still jittering
    its pixel position by several pixels frame-to-frame -- especially in low
    contrast footage (e.g. black shoes on a black lifting mat), where there's
    no clean edge for the model to anchor on. A median over a short window
    rejects that kind of single-frame position spike without needing to
    re-detect anything or discard the frame; a true fast movement still comes
    through since it persists across the window instead of being a one-off.
    """

    def __init__(self, window: int = 5):
        self._history = defaultdict(lambda: deque(maxlen=window))

    def smooth(self, joints: dict) -> dict:
        smoothed = {}
        for name, (x, y) in joints.items():
            hist = self._history[name]
            hist.append((x, y))
            smoothed[name] = (
                statistics.median(p[0] for p in hist),
                statistics.median(p[1] for p in hist),
            )
        return smoothed
