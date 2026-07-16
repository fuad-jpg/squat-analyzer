from pose_estimator import PoseEstimator


def test_ear_fallback_kicks_in_when_shoulder_visibility_is_low():
    joints = {"shoulder": (400.0, 50.0), "ear": (105.0, 40.0), "hip": (100.0, 200.0)}
    visibility = {"shoulder": 0.1, "ear": 0.9, "hip": 0.9}

    new_joints, new_visibility = PoseEstimator.apply_ear_fallback_for_shoulder(
        joints, visibility, min_visibility=0.5
    )

    assert new_joints["shoulder"] == joints["ear"]
    assert new_visibility["shoulder"] == visibility["ear"]
    # Unrelated joints pass through untouched.
    assert new_joints["hip"] == joints["hip"]


def test_ear_fallback_does_nothing_when_shoulder_is_already_trustworthy():
    joints = {"shoulder": (110.0, 55.0), "ear": (105.0, 40.0), "hip": (100.0, 200.0)}
    visibility = {"shoulder": 0.9, "ear": 0.9, "hip": 0.9}

    new_joints, new_visibility = PoseEstimator.apply_ear_fallback_for_shoulder(
        joints, visibility, min_visibility=0.5
    )

    assert new_joints["shoulder"] == joints["shoulder"]
    assert new_visibility["shoulder"] == visibility["shoulder"]


def test_ear_fallback_skipped_when_ear_is_also_unreliable():
    # Nothing trustworthy to fall back to -- leave the (bad) shoulder as-is
    # rather than substituting an equally-unreliable ear reading.
    joints = {"shoulder": (400.0, 50.0), "ear": (410.0, 45.0), "hip": (100.0, 200.0)}
    visibility = {"shoulder": 0.1, "ear": 0.2, "hip": 0.9}

    new_joints, new_visibility = PoseEstimator.apply_ear_fallback_for_shoulder(
        joints, visibility, min_visibility=0.5
    )

    assert new_joints["shoulder"] == joints["shoulder"]
    assert new_visibility["shoulder"] == visibility["shoulder"]
