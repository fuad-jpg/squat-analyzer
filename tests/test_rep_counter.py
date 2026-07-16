from dataclasses import dataclass

import config
from rep_counter import RepCounter


@dataclass
class FakeFrame:
    frame_index: int
    knee_angle: float


def _feed(counter, angles):
    closed = []
    for i, angle in enumerate(angles):
        result = counter.update(FakeFrame(frame_index=i, knee_angle=angle))
        if result is not None:
            closed.append(result)
    final = counter.finalize()
    if final is not None:
        closed.append(final)
    return closed


# Built from config's own thresholds (rather than hardcoded angles) so these
# tests stay meaningful if the thresholds get retuned later. _BOTTOM has to
# clear two separate bars: it must cross the descent trigger (to start being
# tracked as a rep attempt at all) AND drop far enough below _STANDING to
# satisfy MIN_KNEE_DROP_FOR_VALID_REP (to actually count once it's over) --
# so take whichever constraint is currently stricter, with margin to spare.
_STANDING = config.STANDING_KNEE_ANGLE + 10
_BOTTOM = min(
    config.DESCENT_TRIGGER_KNEE_ANGLE - 20,
    _STANDING - config.MIN_KNEE_DROP_FOR_VALID_REP - 20,
)


def test_counts_one_full_rep():
    angles = [_STANDING, _STANDING, _STANDING - 10, _BOTTOM + 20, _BOTTOM,
              _BOTTOM + 5, _BOTTOM + 20, _STANDING - 5, _STANDING, _STANDING]
    reps = _feed(RepCounter(), angles)
    assert len(reps) == 1
    assert min(f.knee_angle for f in reps[0].frames) == _BOTTOM


def test_ignores_bend_that_never_crosses_the_descent_trigger():
    # Dips, but stays above DESCENT_TRIGGER_KNEE_ANGLE, so it's never even
    # considered the start of a squat attempt.
    shallow = config.DESCENT_TRIGGER_KNEE_ANGLE + 10
    angles = [_STANDING, _STANDING, shallow, shallow, _STANDING, _STANDING]
    reps = _feed(RepCounter(), angles)
    assert len(reps) == 0


def test_counts_multiple_reps():
    one_rep = [_STANDING, _BOTTOM + 20, _BOTTOM, _BOTTOM + 20, _STANDING]
    angles = one_rep + one_rep[1:]
    reps = _feed(RepCounter(), angles)
    assert len(reps) == 2
