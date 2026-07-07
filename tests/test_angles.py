import math

from angles import calculate_angle, facing_direction, knee_angle, torso_lean_from_vertical


def test_calculate_angle_right_angle():
    a, b, c = (0, 0), (0, 1), (1, 1)
    assert math.isclose(calculate_angle(a, b, c), 90.0, abs_tol=1e-6)


def test_calculate_angle_straight_line():
    a, b, c = (0, 0), (1, 0), (2, 0)
    assert math.isclose(calculate_angle(a, b, c), 180.0, abs_tol=1e-6)


def test_knee_angle_fully_extended():
    hip, knee, ankle = (0, 0), (0, 10), (0, 20)
    assert math.isclose(knee_angle(hip, knee, ankle), 180.0, abs_tol=1e-6)


def test_torso_lean_upright_is_zero():
    shoulder, hip = (5, 0), (5, 10)
    assert math.isclose(torso_lean_from_vertical(shoulder, hip), 0.0, abs_tol=1e-6)


def test_facing_direction():
    assert facing_direction(heel=(0, 0), toe=(5, 0)) == 1.0
    assert facing_direction(heel=(5, 0), toe=(0, 0)) == -1.0
