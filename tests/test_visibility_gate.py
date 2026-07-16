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


def test_passes_through_but_does_not_cache_a_bad_reading_with_no_prior_good_data():
    # No prior "last good" position exists yet, so there's nothing to hold --
    # the current reading has to be used even if it's low-confidence, since
    # there's no better option. But it must NOT get cached as trustworthy:
    # doing that anchors later frames to a bad position and then "snaps" to
    # the real one once a genuinely good reading finally arrives -- this was
    # a real bug (a plate covering the hip during unracking, right at the
    # start of a video, produced a jump big enough to spuriously trip rep
    # detection).
    gate = JointVisibilityGate(min_visibility=0.5)

    bad_frame = {"hip": (400.0, 50.0)}
    for _ in range(5):
        trusted = gate.filter(bad_frame, {"hip": 0.1})
        assert trusted == bad_frame  # passed through every time, never held

    # A genuinely good reading arrives -- it should be adopted immediately,
    # not compared against some stale cached "last good" from the bad data.
    good_frame = {"hip": (105.0, 202.0)}
    trusted = gate.filter(good_frame, {"hip": 0.9})
    assert trusted == good_frame

    # And it's NOW cached: a subsequent bad frame holds at this good value.
    trusted = gate.filter(bad_frame, {"hip": 0.1})
    assert trusted == good_frame
