from smoothing import JointSmoother


def test_rejects_single_frame_outlier():
    smoother = JointSmoother(window=5)
    steady = (100.0, 200.0)

    for _ in range(5):
        smoothed = smoother.smooth({"heel": steady})
    assert smoothed["heel"] == steady

    # One bad-contrast frame spikes the raw reading far from where the heel
    # actually is -- the median should absorb it instead of reporting it.
    spiked = smoother.smooth({"heel": (100.0, 260.0)})
    assert spiked["heel"] == steady

    # A real, sustained move should still come through once it persists.
    moved = (100.0, 260.0)
    for _ in range(5):
        smoothed = smoother.smooth({"heel": moved})
    assert smoothed["heel"] == moved


def test_tracks_multiple_joints_independently():
    smoother = JointSmoother(window=3)
    for _ in range(3):
        out = smoother.smooth({"heel": (10.0, 20.0), "knee": (50.0, 60.0)})
    assert out == {"heel": (10.0, 20.0), "knee": (50.0, 60.0)}
