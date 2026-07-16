from feedback import evaluate_rep
from metrics import FrameMetrics


def _frame(frame_index, knee_angle, heel_y, thigh_length=100.0):
    return FrameMetrics(
        frame_index=frame_index,
        time_s=frame_index / 30.0,
        knee_angle=knee_angle,
        hip_angle=90.0,
        torso_lean=10.0,
        knee_over_toe_ratio=0.1,
        heel_y=heel_y,
        thigh_length=thigh_length,
    )


def test_heel_rise_uses_this_reps_own_starting_position():
    # heel_y never changes within the rep -- it shouldn't matter that this
    # "ground level" (500) is a completely different number than some other
    # reference point elsewhere in the video.
    frames = [
        _frame(0, 130, heel_y=500.0),
        _frame(1, 100, heel_y=500.0),
        _frame(2, 90, heel_y=500.0),
        _frame(3, 130, heel_y=500.0),
    ]
    report = evaluate_rep(1, frames)
    assert report.max_heel_rise_ratio == 0.0


def test_heel_rise_detects_a_real_lift_within_the_rep():
    frames = [
        _frame(0, 130, heel_y=500.0),
        _frame(1, 100, heel_y=480.0),  # smaller y = higher on screen = heel lifting
        _frame(2, 90, heel_y=470.0),
        _frame(3, 130, heel_y=500.0),
    ]
    report = evaluate_rep(1, frames)
    assert report.max_heel_rise_ratio > 0.0


def test_stance_shift_between_reps_does_not_cause_a_false_heel_rise():
    # This is the exact bug scenario: the lifter's absolute position in
    # frame differs between two reps (re-racking, adjusting stance), so
    # "heel on the ground" sits at a different pixel row each time -- but
    # the heel itself never lifts within either rep.
    rep1_frames = [_frame(i, 160 - i * 20, heel_y=500.0) for i in range(4)]
    rep2_frames = [_frame(i, 160 - i * 20, heel_y=440.0) for i in range(4)]

    rep1 = evaluate_rep(1, rep1_frames)
    rep2 = evaluate_rep(2, rep2_frames)

    assert rep1.max_heel_rise_ratio == 0.0
    assert rep2.max_heel_rise_ratio == 0.0
