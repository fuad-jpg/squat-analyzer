from visibility_gate import JointVisibilityGate


def test_holds_a_single_occluded_joint_at_its_last_good_position():
    # This is the exact bug scenario: a big plate covers the hip for several
    # consecutive frames (not just one) while every other joint -- and the
    # frame's overall average visibility, which video_processor.py checks
    # separately before this gate even runs -- stays fine.
    gate = JointVisibilityGate(min_visibility=0.5)

    good_frame = {"hip": (100.0, 200.0), "knee": (110.0, 300.0)}
    good_vis = {"hip": 0.9, "knee": 0.9}
    trusted = gate.filter(good_frame, good_vis)
    assert trusted == good_frame

    # Hip gets occluded by the plate for a few frames; knee stays fine.
    occluded_frame = {"hip": (400.0, 50.0), "knee": (112.0, 305.0)}
    occluded_vis = {"hip": 0.2, "knee": 0.9}
    for _ in range(4):
        trusted = gate.filter(occluded_frame, occluded_vis)
        assert trusted["hip"] == (100.0, 200.0)  # held at last good position
        assert trusted["knee"] == (112.0, 305.0)  # unaffected, updates normally

    # Once visibility recovers, the joint starts updating again.
    recovered_frame = {"hip": (105.0, 202.0), "knee": (113.0, 306.0)}
    recovered_vis = {"hip": 0.8, "knee": 0.9}
    trusted = gate.filter(recovered_frame, recovered_vis)
    assert trusted == recovered_frame


def test_trusts_a_joint_seen_for_the_first_time_regardless_of_visibility():
    # No prior "last good" position exists yet, so there's nothing to hold --
    # the first reading has to be trusted even if it's low-confidence.
    gate = JointVisibilityGate(min_visibility=0.5)
    frame = {"hip": (100.0, 200.0)}
    trusted = gate.filter(frame, {"hip": 0.1})
    assert trusted == frame
