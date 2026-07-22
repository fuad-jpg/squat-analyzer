"""Tunable thresholds for squat rep detection and form feedback.

These are reasonable starting points based on common strength-coaching cues
for a side-view barbell back squat, not a substitute for a real coach.
See README.md ("Design notes") for the reasoning behind each one.
"""

# Minimum average landmark visibility (MediaPipe confidence, 0-1) required
# before a frame's joint angles are trusted.
MIN_VISIBILITY = 0.5

# --- Rep detection ---------------------------------------------------------
# Knee angle in degrees; 180 = leg fully extended.
STANDING_KNEE_ANGLE = 160.0
DESCENT_TRIGGER_KNEE_ANGLE = 135.0
# A rep only counts if the knee angle dropped at least this far below the
# standing baseline, so weight shifts / knee wobbles while standing aren't
# mistaken for a rep. With STANDING_KNEE_ANGLE=160, this requires the knee
# to reach ~105 deg before a rep is valid.
MIN_KNEE_DROP_FOR_VALID_REP = 55.0
# The knee angle must hold at/above STANDING_KNEE_ANGLE for at least this
# long before a subsequent descent is allowed to start a valid rep attempt.
# Rules out a video that opens mid-motion, or a quick knee bend (e.g. while
# unracking) that happens to cross the descent trigger without the lifter
# ever having actually stood still first.
MIN_STANDING_DURATION_S = 0.3

# --- Depth -------------------------------------------------------------
# Approximate knee angle when thighs are parallel to the floor.
PARALLEL_KNEE_ANGLE = 100.0
SHALLOW_KNEE_ANGLE_MARGIN = 10.0

# --- Torso lean ----------------------------------------------------------
# Loosened from 45: a big plate can visually overlap the shoulder landmark
# in a side view, occasionally pulling MediaPipe's shoulder position off and
# inflating this reading. Tune further based on your own footage.
EXCESSIVE_TORSO_LEAN_DEG = 55.0

# --- Weight shifting forward / heel rise ----------------------------------
# Normalized by thigh length (hip-to-knee pixel distance) so it holds up
# across different camera distances and video resolutions.
HEEL_RISE_RATIO_LIMIT = 0.2