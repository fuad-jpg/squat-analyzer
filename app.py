import os
import tempfile

import altair as alt
import pandas as pd
import streamlit as st

from filming_guide import build_guide_image
from video_processor import analyze_video

st.set_page_config(page_title="Squat Form Analyzer", page_icon="🏋️", layout="centered")

st.title("Barbell Squat Form Analyzer")
st.write(
    "Upload a **side-view** video of a barbell back squat. The app tracks your "
    "joints frame by frame, counts reps, and flags depth, torso lean, and "
    "weight-shift/heel-rise issues."
)

@st.dialog("How to film your video", width="large")
def show_filming_guide():
    st.image(build_guide_image(), width="content")


if st.button("📹 How should I film my video?"):
    show_filming_guide()

with st.expander("New to this? What do these terms mean?"):
    st.markdown(
        """
- **Knee angle** —  The degree of bend at your knee joint. A standing leg is `180°`.
 Reaching "parallel" (thighs level with the floor) occurs around `90°`. 
 Going below `90°` is ideal for full muscle engagement, provided you can maintain
 flat feet and a straight back.
- **Torso lean** — how far your upper body tips forward, measured from
  perfectly upright (`0°`). Some forward lean is normal in a squat; a lot of
  it usually means the weight has shifted off your legs and onto your lower back.
- **Heel rise** — whether your heels lift off the floor. Your feet should stay
  flat and planted the whole rep; heels coming up is a sign of weight
  shifting forward onto your toes, often from tight ankles.
- **Rep** — one full squat: down and back up. The app counts a rep only once
  your knee bends deep enough that it's clearly an intentional squat, not
  just a wobble while standing.
        """
    )

st.write("")
uploaded = st.file_uploader("Upload squat video", type=["mp4", "mov", "avi", "m4v"])

if uploaded is not None:
    # Streamlit reruns this whole script on every interaction -- clicking the
    # filming-guide button, dismissing its dialog, anything -- and the file
    # uploader keeps returning the same file across those reruns. Without
    # this check, re-analysis (and the temp-file write) would fire every
    # single time, not just on an actual new upload.
    if st.session_state.get("analyzed_file_id") != uploaded.file_id:
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

            with open(output_path, "rb") as f:
                video_bytes = f.read()

        st.session_state.analyzed_file_id = uploaded.file_id
        st.session_state.analysis_result = result
        st.session_state.annotated_video_bytes = video_bytes

    result = st.session_state.analysis_result
    video_bytes = st.session_state.annotated_video_bytes

    st.video(video_bytes)
    st.subheader(f"{len(result.reps)} rep(s) detected")

    if not result.reps:
        st.warning(
            "No reps detected. Make sure the full body is visible from the side, "
            "the lighting is decent, and the squat is filmed with a clear knee bend."
        )

    for rep in result.reps:
        icon = "✅" if rep.is_good_form else "⚠️"
        with st.expander(
            f"{icon} Rep {rep.rep_number} — min knee angle {rep.min_knee_angle:.0f}°",
            expanded=True,
        ):
            for line in rep.feedback:
                st.write(f"- {line}")

    if result.frame_metrics:
        chart_df = pd.DataFrame({
            "Time (s)": [fm.time_s for fm in result.frame_metrics],
            "Knee angle (deg)": [fm.knee_angle for fm in result.frame_metrics],
        })
        line = alt.Chart(chart_df).mark_line().encode(
            x=alt.X("Time (s)", title="Time (s)"),
            y=alt.Y("Knee angle (deg)", title="Knee angle (deg)"),
        )
        chart = line

        # Mark the bottom of each counted rep with a circle, so it's easy to
        # see exactly where in the video each rep the app detected lines up.
        by_frame = {fm.frame_index: fm for fm in result.frame_metrics}
        rep_points = [
            {
                "Time (s)": by_frame[rep.bottom_frame].time_s,
                "Knee angle (deg)": by_frame[rep.bottom_frame].knee_angle,
                "Rep": f"Rep {rep.rep_number}",
            }
            for rep in result.reps
            if rep.bottom_frame in by_frame
        ]
        if rep_points:
            markers = alt.Chart(pd.DataFrame(rep_points)).mark_circle(
                size=140, color="#FFD54A", stroke="#8A6D00", strokeWidth=1.5,
            ).encode(
                x="Time (s)",
                y="Knee angle (deg)",
                tooltip=["Rep", "Time (s)", "Knee angle (deg)"],
            )
            chart = line + markers

        st.subheader("Knee angle over time")
        st.altair_chart(chart, width="stretch")

st.divider()
st.caption(
    "Phase 1 of a larger project analyzing gym exercise form (squat, bench, and more). "
    "This Python/OpenCV/MediaPipe pipeline is the analysis engine behind a future full web app."
)
