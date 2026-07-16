"""Generates the "how to film your squat" diagram shown on the app's main
screen, before a video is uploaded. Drawn with PIL rather than shipped as a
static image so it has no binary asset to keep in sync with code changes.

Colors are tuned for Streamlit's built-in dark theme (.streamlit/config.toml
sets base="dark", background #0E1117) -- light, high-contrast lines and
text so nothing reads as thin, faint, or hard to see.
"""
import math

from PIL import Image, ImageDraw, ImageFont

_INK = (250, 250, 250, 255)      # matches theme textColor
_SUBTLE = (176, 182, 194, 255)   # light gray-blue, still clearly legible on navy
_FAINT = (176, 182, 194, 110)
_SAGE = (129, 199, 165, 255)     # bright-enough green to read on dark navy
_RUST = (235, 140, 116, 255)


def _font(size, bold=False):
    try:
        path = r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _dashed_line(draw, p1, p2, color, width=2, dash=7, gap=7):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    n = int(dist // (dash + gap)) + 1
    for i in range(n):
        start = i * (dash + gap)
        end = min(start + dash, dist)
        draw.line([(x1 + dx * start, y1 + dy * start), (x1 + dx * end, y1 + dy * end)],
                  fill=color, width=width)


def _camera_icon(draw, cx, cy, color, scale=1.0):
    w, h = 32 * scale, 21 * scale
    draw.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=5, outline=color, width=3)
    draw.rectangle([cx - w / 2 + 4, cy - h / 2 - 6, cx - w / 2 + 12, cy - h / 2 + 1], outline=color, width=3)
    r = 6.5 * scale
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)


def build_guide_image() -> Image.Image:
    """Returns a two-panel RGBA diagram (transparent background, so it blends
    into the app's dark page background): (1) bird's-eye camera placement,
    showing the camera should be at 90 deg to the lifter, not at an angle,
    and (2) a framing guide showing a full-body squat-bottom silhouette
    inside a video frame with headroom/footroom margins.

    Sized to be shown at (roughly) its own native resolution -- see app.py,
    which intentionally does NOT stretch this to fill the container. Earlier
    attempts either scaled the canvas up to fight a "stretch to full width"
    display (ballooning it on a wide layout) or scaled it down while still
    being stretched (shrinking the text). Fixing the display mode instead of
    the canvas size is what actually keeps this legible without being huge.
    """
    w, h = 1000, 480
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    f_eyebrow = _font(14, bold=True)
    f_label = _font(16)
    f_label_b = _font(16, bold=True)
    f_small = _font(13)

    lm, rm, tm = 55, 55, 40
    gutter_center = w // 2

    d.text((lm, tm), "CAMERA ANGLE", font=f_eyebrow, fill=_SUBTLE)
    d.text((gutter_center + 55, tm), "FRAMING", font=f_eyebrow, fill=_SUBTLE)

    # ---------- LEFT: bird's-eye camera placement ----------
    left_cx = lm + 175
    lifter_y = 250

    d.line([(left_cx, lifter_y), (left_cx + 58, lifter_y)], fill=_SUBTLE, width=3)
    d.polygon([(left_cx + 58, lifter_y - 6), (left_cx + 70, lifter_y), (left_cx + 58, lifter_y + 6)], fill=_SUBTLE)
    d.text((left_cx + 14, lifter_y + 13), "facing", font=f_small, fill=_SUBTLE)

    d.ellipse([left_cx - 11, lifter_y - 11, left_cx + 11, lifter_y + 11], fill=_INK)

    _dashed_line(d, (left_cx, 105), (left_cx, 385), _FAINT, width=2, dash=5, gap=6)

    good_y = 350
    _dashed_line(d, (left_cx, good_y - 14), (left_cx, lifter_y + 13), _SAGE, width=2, dash=5, gap=5)
    _camera_icon(d, left_cx, good_y, _SAGE, scale=1.15)
    d.text((left_cx, good_y + 33), "Straight-on side view", font=f_label_b, fill=_INK, anchor="mm")
    d.text((left_cx, good_y + 53), "camera at 90° to the lifter", font=f_small, fill=_SUBTLE, anchor="mm")

    bad_x, bad_y = left_cx + 140, 135
    _dashed_line(d, (bad_x - 10, bad_y + 11), (left_cx + 11, lifter_y - 11), _FAINT, width=2, dash=5, gap=5)
    _camera_icon(d, bad_x, bad_y, _RUST, scale=1.0)
    d.text((bad_x, bad_y + 28), "Angled / 3-4 view", font=f_label, fill=_SUBTLE, anchor="mm")
    d.text((bad_x, bad_y + 46), "(avoid)", font=f_small, fill=_SUBTLE, anchor="mm")

    # ---------- RIGHT: framing guide ----------
    fx0 = gutter_center + 55
    fx1 = w - rm - 80
    fy0 = tm + 50
    fy1 = h - 65

    ground_y = fy1 - 16
    d.line([(fx0, ground_y), (fx1, ground_y)], fill=_SUBTLE, width=2)
    d.rectangle([fx0, fy0, fx1, fy1], outline=_FAINT, width=2)

    cx = (fx0 + fx1) // 2

    # Squat-bottom stick figure, facing right: hips sit back with the thigh
    # roughly horizontal (parallel depth), knee forward of the ankle, torso
    # leaning forward to keep the bar path over mid-foot.
    ankle = (cx, ground_y)
    knee = (cx + 27, ground_y - 76)
    hip = (knee[0] - 66, knee[1] - 10)
    shoulder = (hip[0] + 66, hip[1] - 80)
    head_r = 14
    head = (shoulder[0] - 2, shoulder[1] - head_r - 7)

    d.ellipse([head[0] - head_r, head[1] - head_r, head[0] + head_r, head[1] + head_r], outline=_INK, width=3)
    d.line([(shoulder[0] - 25, shoulder[1] - 3), (shoulder[0] + 21, shoulder[1] - 3)], fill=_SAGE, width=5)
    d.line([shoulder, hip], fill=_INK, width=3)
    d.line([hip, knee], fill=_INK, width=3)
    d.line([knee, ankle], fill=_INK, width=3)
    d.line([(ankle[0] - 17, ankle[1]), (ankle[0] + 23, ankle[1])], fill=_INK, width=3)

    _dashed_line(d, (fx0, fy0 - 13), (fx1, fy0 - 13), _FAINT, width=2, dash=5, gap=6)
    d.text((cx, fy0 - 28), "a little headroom above", font=f_small, fill=_SUBTLE, anchor="mm")
    d.text((cx, ground_y + 30), "feet and floor fully visible", font=f_small, fill=_SUBTLE, anchor="mm")

    d.line([(fx1 + 12, hip[1]), (fx1 + 25, hip[1])], fill=_SUBTLE, width=2)
    d.text((fx1 + 32, hip[1]), "camera ~\nhip height", font=f_small, fill=_SUBTLE, anchor="lm")

    return img
