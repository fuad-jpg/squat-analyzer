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


def _ramp(start, end, steps):
    """Linearly interpolate from start to end over `steps` frames.

    Mimics how a real knee angle actually moves continuously through
    intermediate values, rather than jumping straight from one extreme to
    another in a single frame the way a hand-written test fixture might.
    That distinction matters: a real descent always passes gradually through
    the gap between DESCENT_TRIGGER_KNEE_ANGLE and STANDING_KNEE_ANGLE, and
    a bug that only manifests while passing through that gap (there was
    exactly one) stays invisible if every test fixture skips straight over
    it with an instantaneous jump.
    """
    if steps <= 1:
        return [end]
    return [start + (end - start) * i / (steps - 1) for i in range(steps)]


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


def _one_rep(hold_top=5, hold_bottom=3, ramp_steps=8):
    """A realistic-shaped single rep: hold standing, ramp gradually down
    through the descent trigger, hold briefly at the bottom, ramp back up."""
    return (
        [_STANDING] * hold_top
        + _ramp(_STANDING, _BOTTOM, ramp_steps)
        + [_BOTTOM] * hold_bottom
        + _ramp(_BOTTOM, _STANDING, ramp_steps)
    )


def _counter():
    # fps=1 makes the "must stand for MIN_STANDING_DURATION_S before
    # descending" precondition require just 1 frame (max(1, round(1*0.3))),
    # effectively a no-op -- these tests are about other behavior and
    # already include a several standing frames as a matter of course, not
    # specifically testing the duration requirement itself (see
    # test_requires_a_minimum_standing_duration_before_a_descent_counts).
    return RepCounter(fps=1)


def test_counts_one_full_rep():
    reps = _feed(_counter(), _one_rep())
    assert len(reps) == 1
    assert min(f.knee_angle for f in reps[0].frames) == _BOTTOM


def test_ignores_bend_that_never_crosses_the_descent_trigger():
    # Dips, but stays above DESCENT_TRIGGER_KNEE_ANGLE, so it's never even
    # considered the start of a squat attempt.
    shallow = config.DESCENT_TRIGGER_KNEE_ANGLE + 10
    angles = (
        [_STANDING] * 5
        + _ramp(_STANDING, shallow, 6)
        + [shallow] * 3
        + _ramp(shallow, _STANDING, 6)
    )
    reps = _feed(_counter(), angles)
    assert len(reps) == 0


def test_counts_multiple_reps():
    # Two full reps back to back, each with its own standing hold -- the
    # standing-duration requirement means back-to-back reps need an actual
    # standing stretch between them, not just the single frame that closes
    # one rep re-used as if it were also the next rep's pre-descent stand.
    angles = _one_rep() + _one_rep()
    reps = _feed(_counter(), angles)
    assert len(reps) == 2


def test_requires_a_minimum_standing_duration_before_a_descent_counts():
    # At a realistic fps, a descent that starts before the lifter has stood
    # still for MIN_STANDING_DURATION_S shouldn't count as a rep attempt at
    # all -- e.g. the video opening mid-motion, or a quick knee bend while
    # unracking that happens to cross the descent trigger.
    fps = 30
    min_frames = round(fps * config.MIN_STANDING_DURATION_S)

    # Only briefly at standing angle before descending: too short.
    angles = [_STANDING] * (min_frames - 2) + _one_rep(hold_top=0)
    reps = _feed(RepCounter(fps=fps), angles)
    assert len(reps) == 0

    # Same shape, but standing for long enough first: counts normally.
    angles = [_STANDING] * (min_frames + 2) + _one_rep(hold_top=0)
    reps = _feed(RepCounter(fps=fps), angles)
    assert len(reps) == 1
