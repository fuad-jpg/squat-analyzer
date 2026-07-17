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


def _counter():
    # fps=1 makes the new "must stand for MIN_STANDING_DURATION_S before
    # descending" precondition require just 1 frame (max(1, round(1*0.3))),
    # effectively a no-op -- these tests are about other behavior and
    # already include a couple of standing frames as a matter of course,
    # not specifically testing the duration requirement itself (see
    # test_requires_a_minimum_standing_duration_before_a_descent_counts).
    return RepCounter(fps=1)


def test_counts_one_full_rep():
    angles = [_STANDING, _STANDING, _STANDING - 10, _BOTTOM + 20, _BOTTOM,
              _BOTTOM + 5, _BOTTOM + 20, _STANDING - 5, _STANDING, _STANDING]
    reps = _feed(_counter(), angles)
    assert len(reps) == 1
    assert min(f.knee_angle for f in reps[0].frames) == _BOTTOM


def test_ignores_bend_that_never_crosses_the_descent_trigger():
    # Dips, but stays above DESCENT_TRIGGER_KNEE_ANGLE, so it's never even
    # considered the start of a squat attempt.
    shallow = config.DESCENT_TRIGGER_KNEE_ANGLE + 10
    angles = [_STANDING, _STANDING, shallow, shallow, _STANDING, _STANDING]
    reps = _feed(_counter(), angles)
    assert len(reps) == 0


def test_counts_multiple_reps():
    # Note: two full copies, not one_rep + one_rep[1:] -- the new standing-
    # duration requirement means back-to-back reps need an actual standing
    # frame between them, not just the single frame that closes the first
    # rep re-used as if it were also the second rep's pre-descent stand.
    one_rep = [_STANDING, _BOTTOM + 20, _BOTTOM, _BOTTOM + 20, _STANDING]
    angles = one_rep + one_rep
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
    angles = ([_STANDING] * (min_frames - 2)
              + [_BOTTOM + 20, _BOTTOM, _BOTTOM + 20, _STANDING, _STANDING])
    reps = _feed(RepCounter(fps=fps), angles)
    assert len(reps) == 0

    # Same shape, but standing for long enough first: counts normally.
    angles = ([_STANDING] * (min_frames + 2)
              + [_BOTTOM + 20, _BOTTOM, _BOTTOM + 20, _STANDING, _STANDING])
    reps = _feed(RepCounter(fps=fps), angles)
    assert len(reps) == 1
