import statistics
from collections import defaultdict, deque


class JointSmoother:
    """Rolling median filter over each joint's (x, y) position.

    MediaPipe can report high confidence for a landmark while still jittering
    its pixel position by several pixels frame-to-frame -- especially in low
    contrast footage (e.g. black shoes on a black lifting mat, or a shadowed
    hip near a plate), where there's no clean edge for the model to anchor
    on. A median over a window rejects that kind of position spike without
    needing to re-detect anything or discard the frame; a true fast movement
    still comes through since it persists across the window instead of being
    a one-off.

    Tested against real footage with a sustained hip/heel misdetection (up
    to 9 consecutive frames, using the pose_landmarker_heavy model): window
    had to reach 15 to fully outvote that bad stretch -- 5, 9, and 11 all
    still left some anomalies. This default is intentionally lower than
    that fully-clean value (trading some of that correction back for less
    lag), since a wider window also delays how quickly a genuine fast
    movement is fully reflected (window=15 is ~0.5s of lag at 30fps).
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
