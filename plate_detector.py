import math

import cv2
import numpy as np


def detect_plate(frame_bgr):
    """Detect the most prominent large dark circular object in a frame (a
    barbell plate) via a Hough circle transform. Returns (cx, cy, r) in
    pixel coordinates, or None if nothing plausible is found.

    This is a heuristic, not a guarantee: circle detection works off edge
    contrast, which can be weak wherever the plate happens to blend into a
    dark background (the same lighting problem that causes MediaPipe's own
    landmark errors). It only needs a good arc of contrast to fit a circle,
    not the full circumference, so it tends to work when part of the plate
    is against a lighter background even if another part isn't.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    h = gray.shape[0]

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=h / 3,
        param1=100,
        param2=40,
        minRadius=int(h * 0.08),
        maxRadius=int(h * 0.35),
    )
    if circles is None:
        return None

    # Prefer the largest candidate: a barbell plate close enough to read
    # form from should be one of the biggest circular things in frame.
    cx, cy, r = max(np.round(circles[0]).astype(float), key=lambda c: c[2])
    return float(cx), float(cy), float(r)


def suppress_joints_inside_plate(joints, visibility, plate, exempt=("nose",)):
    """Zero out the visibility of any joint whose reported position falls
    inside the detected plate's circle, `exempt` joints aside. MediaPipe can
    report a confidently-wrong position for a landmark the plate is
    covering; a geometric check like this catches that even when MediaPipe's
    own confidence score doesn't reflect it.
    """
    if plate is None:
        return dict(visibility)

    cx, cy, r = plate
    visibility = dict(visibility)
    for name, (x, y) in joints.items():
        if name in exempt:
            continue
        if math.hypot(x - cx, y - cy) <= r:
            visibility[name] = 0.0
    return visibility
