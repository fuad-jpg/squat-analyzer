import numpy as np


def calculate_angle(a, b, c):
    """Angle at point b formed by rays b->a and b->c, in degrees (0-180)."""
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    cosine = np.dot(ba, bc) / denom if denom > 1e-9 else 0.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def knee_angle(hip, knee, ankle):
    return calculate_angle(hip, knee, ankle)


def hip_angle(shoulder, hip, knee):
    return calculate_angle(shoulder, hip, knee)


def torso_lean_from_vertical(shoulder, hip):
    """0 deg = perfectly upright torso, larger = more forward lean."""
    vertical_reference = (hip[0], hip[1] - 100)
    return calculate_angle(vertical_reference, hip, shoulder)


def facing_direction(heel, toe):
    """+1 if the person faces toward +x (toe right of heel), else -1."""
    return 1.0 if toe[0] >= heel[0] else -1.0
