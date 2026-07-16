import argparse
import json
import sys

from video_processor import analyze_video


def main():
    parser = argparse.ArgumentParser(
        description="Analyze barbell squat form from a side-view video."
    )
    parser.add_argument("video", help="Path to input video file (mp4, mov, avi, ...)")
    parser.add_argument("-o", "--output", default=None,
                         help="Path to save the annotated output video (default: <video>_annotated.mp4)")
    parser.add_argument("--no-video", action="store_true",
                         help="Skip writing an annotated video; just print the report.")
    parser.add_argument("--json", default=None, help="Optional path to dump results as JSON.")
    args = parser.parse_args()

    output_path = None if args.no_video else (args.output or _default_output_path(args.video))

    print(f"Analyzing {args.video} ...")
    result = analyze_video(args.video, output_path=output_path)

    print(f"\n{len(result.reps)} rep(s) detected over {result.duration_s:.1f}s at {result.fps:.1f} fps\n")
    for rep in result.reps:
        print(f"Rep {rep.rep_number}: min knee angle {rep.min_knee_angle:.0f}°, "
              f"max torso lean {rep.max_torso_lean:.0f}°")
        for line in rep.feedback:
            print(f"  - {line.replace('**', '')}")
        print()

    if not result.reps:
        print("No reps detected. Make sure the full body is visible from the side "
              "and the squat has a clear knee bend.")

    if result.output_video_path:
        print(f"Annotated video saved to {result.output_video_path}")

    if args.json:
        _dump_json(result, args.json)
        print(f"Results saved to {args.json}")


def _default_output_path(video_path: str) -> str:
    stem, dot, ext = video_path.rpartition(".")
    stem = stem if dot else video_path
    return f"{stem}_annotated.mp4"


def _dump_json(result, path: str):
    data = {
        "fps": result.fps,
        "frame_count": result.frame_count,
        "duration_s": result.duration_s,
        "reps": [
            {
                "rep_number": r.rep_number,
                "min_knee_angle": r.min_knee_angle,
                "max_torso_lean": r.max_torso_lean,
                "max_heel_rise_ratio": r.max_heel_rise_ratio,
                "feedback": r.feedback,
            }
            for r in result.reps
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
