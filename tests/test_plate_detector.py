import numpy as np

from plate_detector import detect_plate, suppress_joints_inside_plate


def test_detects_a_clearly_drawn_circle():
    import cv2

    frame = np.full((400, 400, 3), 255, dtype=np.uint8)  # white background
    true_center = (200, 180)
    true_radius = 90
    cv2.circle(frame, true_center, true_radius, (20, 20, 20), thickness=-1)

    result = detect_plate(frame)

    assert result is not None
    cx, cy, r = result
    assert abs(cx - true_center[0]) < 15
    assert abs(cy - true_center[1]) < 15
    assert abs(r - true_radius) < 20


def test_returns_none_for_a_blank_frame():
    frame = np.full((400, 400, 3), 128, dtype=np.uint8)
    assert detect_plate(frame) is None


def test_suppress_zeroes_visibility_for_a_joint_inside_the_plate():
    plate = (200.0, 200.0, 80.0)
    joints = {
        "hip": (210.0, 190.0),  # inside the circle
        "ankle": (10.0, 390.0),  # far outside
        "ear": (205.0, 195.0),  # inside, but exempt
    }
    visibility = {"hip": 0.9, "ankle": 0.9, "ear": 0.9}

    result = suppress_joints_inside_plate(joints, visibility, plate)

    assert result["hip"] == 0.0
    assert result["ankle"] == 0.9
    assert result["ear"] == 0.9


def test_suppress_is_a_no_op_when_no_plate_detected():
    joints = {"hip": (210.0, 190.0)}
    visibility = {"hip": 0.9}
    assert suppress_joints_inside_plate(joints, visibility, None) == visibility
