from dataclasses import dataclass

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


def test_counts_one_full_rep():
    angles = [170, 170, 160, 130, 95, 90, 100, 140, 165, 170, 170]
    reps = _feed(RepCounter(), angles)
    assert len(reps) == 1
    assert min(f.knee_angle for f in reps[0].frames) == 90


def test_ignores_small_knee_bend_as_noise():
    # Dips below the standing baseline but not far enough to be a real squat.
    angles = [170, 170, 148, 148, 170, 170]
    reps = _feed(RepCounter(), angles)
    assert len(reps) == 0


def test_counts_multiple_reps():
    one_rep = [170, 130, 90, 130, 170]
    angles = one_rep + one_rep[1:]
    reps = _feed(RepCounter(), angles)
    assert len(reps) == 2
