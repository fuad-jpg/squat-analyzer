from plate_detector import PlateHipEstimator


def test_learns_offset_and_estimates_hip_while_occluded():
    estimator = PlateHipEstimator()

    # Plate and hip both visible: learn the (dx, dy) relationship.
    plate = (200.0, 100.0, 60.0)
    joints = {"hip": (210.0, 250.0)}
    visibility = {"hip": 0.9}
    new_joints, new_vis = estimator.update_and_estimate(joints, visibility, plate)
    assert new_joints == joints  # untouched while hip is trusted
    assert new_vis == visibility

    # Hip becomes occluded (suppressed to 0 upstream), but the plate has
    # since moved -- the estimate should track that movement, not freeze.
    moved_plate = (220.0, 130.0, 60.0)
    occluded_joints = {"hip": (999.0, 999.0)}  # whatever MediaPipe reported, ignored
    occluded_vis = {"hip": 0.0}
    new_joints, new_vis = estimator.update_and_estimate(occluded_joints, occluded_vis, moved_plate)

    expected = (220.0 + (210.0 - 200.0), 130.0 + (250.0 - 100.0))
    assert new_joints["hip"] == expected
    assert new_vis["hip"] == 1.0


def test_does_nothing_without_a_learned_offset_yet():
    estimator = PlateHipEstimator()
    plate = (200.0, 100.0, 60.0)
    joints = {"hip": (999.0, 999.0)}
    visibility = {"hip": 0.0}

    new_joints, new_vis = estimator.update_and_estimate(joints, visibility, plate)

    # Nothing learned yet, so pass through unchanged rather than guess.
    assert new_joints == joints
    assert new_vis == visibility


def test_is_a_no_op_when_no_plate_detected():
    estimator = PlateHipEstimator()
    joints = {"hip": (210.0, 250.0)}
    visibility = {"hip": 0.9}
    new_joints, new_vis = estimator.update_and_estimate(joints, visibility, None)
    assert new_joints == joints
    assert new_vis == visibility
