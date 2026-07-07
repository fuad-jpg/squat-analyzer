import os
import tempfile

import streamlit as st

from video_processor import analyze_video

st.set_page_config(page_title="Squat Form Analyzer", page_icon="🏋️", layout="centered")

st.title("Barbell Squat Form Analyzer")
st.write(
    "Upload a **side-view** video of a barbell back squat. The app tracks your "
    "joints frame by frame, counts reps, and flags depth, torso lean, and "
    "weight-shift/heel-rise issues."
)

uploaded = st.file_uploader("Upload squat video", type=["mp4", "mov", "avi", "m4v"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, uploaded.name)
        with open(input_path, "wb") as f:
            f.write(uploaded.read())

        output_path = os.path.join(tmp_dir, "annotated.mp4")
        progress_bar = st.progress(0.0, text="Analyzing video...")

        result = analyze_video(
            input_path,
            output_path=output_path,
            progress_callback=lambda frac: progress_bar.progress(min(frac, 1.0), text="Analyzing video..."),
        )
        progress_bar.empty()

        st.video(output_path)
        st.subheader(f"{len(result.reps)} rep(s) detected")

        if not result.reps:
            st.warning(
                "No reps detected. Make sure the full body is visible from the side, "
                "the lighting is decent, and the squat is filmed with a clear knee bend."
            )

        for rep in result.reps:
            with st.expander(f"Rep {rep.rep_number} — min knee angle {rep.min_knee_angle:.0f}°", expanded=True):
                for line in rep.feedback:
                    st.write(f"- {line}")

        knee_angles = [fm.knee_angle for fm in result.frame_metrics]
        if knee_angles:
            st.subheader("Knee angle over time")
            st.line_chart(knee_angles)

st.divider()
st.caption(
    "Phase 1 of a larger project analyzing gym exercise form (squat, bench, and more). "
    "This Python/OpenCV/MediaPipe pipeline is the analysis engine behind a future full web app."
)
