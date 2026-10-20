#!/usr/bin/env python3
"""
run_pipeline.py - the whole reference-style pipeline with one command.

    input/movie.mp4
      transcribe.py     -> audio/, transcript/                   (skipped if the transcript already exists)
      locate_scene.py   -> analysis/<stem>_anchors.json          (finds where the recap should STOP, and optionally the
                                                                  opening shot; needs a Gemini key; cached)
      analyze_movie.py  -> analysis/, script/                    (reuses saved plot analysis; Groq or Gemini)
      verify_clips.py   -> optional footage-vs-narration check   (--check-footage; Gemini vision)
      voiceover.py      -> narration/<stem>/narration.wav + timeline.json   (one continuous take)
      cut_shots.py      -> clips/<stem>/                         (hard-cut, frame-exact, full-screen shots + end card)
      render_final.py   -> output/<stem>_recap.mp4
      verify_video.py   -> output/<stem>_verify/                 (images + report)

Usage (from the project root):
    python3 src/run_pipeline.py --reference reference/ref.mp4      # copy a reference video exactly (recommended)
    python3 src/run_pipeline.py
    python3 src/run_pipeline.py --check-footage --vision
    python3 src/run_pipeline.py --from-step voiceover          # resume at a later step
    python3 src/run_pipeline.py --stop-after analyze           # read the script before voicing it
    python3 src/run_pipeline.py --cliffhanger 00:21:30         # you choose where the recap stops
"""

import argparse
import subprocess
import sys
from pathlib import Path

from common import ROOT, d, list_movies, load_brief, load_json

SRC = Path(__file__).resolve().parent
STEPS = ["transcribe", "locate", "analyze", "footage", "voiceover", "shots", "render", "verify"]
CLONE_STEPS = ["clone", "voiceover", "shots", "render", "verify"]


def run_step(script, *args, allow_fail=False):
    print(f"\n=== {script} {' '.join(args)} ===")
    rc = subprocess.run([sys.executable, str(SRC / script), *args], cwd=ROOT).returncode
    if rc != 0 and not allow_fail:
        sys.exit(f"\nStopped: {script} failed (see above). Fix it and re-run; finished steps are skipped or reused.")
    return rc


def run_clone_flow(args):
    """Copy a reference video: same shots, same cut points, same narration, same ending, plus the call to action."""
    order = CLONE_STEPS
    first = args.from_step if args.from_step in order else "clone"
    todo = order[order.index(first):]
    if args.stop_after and args.stop_after in order:
        todo = todo[:todo.index(args.stop_after) + 1]
    if "clone" in todo:
        run_step("clone_reference.py", "--reference", args.reference, "--whisper-model", args.whisper_model,
                 *(["--title", args.title] if args.title else []), *(["--brief", args.brief] if args.brief else []))
        print("\nOpen analysis/*_matches.jpg now: top row = the reference shot, bottom row = the matched movie frame. "
              "Red numbers mark doubtful matches.")
    if "voiceover" in todo:
        run_step("voiceover.py", "--engine", args.engine, *(["--voice", args.voice] if args.voice else []),
                 *(["--style", args.style] if args.style else []))
    if "shots" in todo:
        run_step("cut_shots.py", "--resolution", args.resolution, *(["--poster", args.poster] if args.poster else []))
    if "render" in todo:
        run_step("render_final.py", *(["--music", args.music] if args.music else []))
    if "verify" in todo:
        rc = run_step("verify_video.py", allow_fail=True)
        print("\nPipeline finished. Compare output/<name>_recap.mp4 with your reference, and read the picture sheet."
              + ("" if rc == 0 else "\nverify_video.py reported FAIL lines. Read them before publishing."))


def main():
    ap = argparse.ArgumentParser(description="Run the full movie-recap pipeline.")
    ap.add_argument("--title", help="Movie title (default: assets/<stem>_brief.json, else the file name)")
    ap.add_argument("--brief", help="Brief JSON (default: assets/<stem>_brief.json)")
    ap.add_argument("--reference", help="A reference video to copy shot for shot (same footage, cuts, narration, ending). "
                                         "Keep it OUTSIDE input/ (e.g. reference/ref.mp4)")
    ap.add_argument("--from-step", choices=sorted(set(STEPS + CLONE_STEPS), key=(STEPS + CLONE_STEPS).index), default=None)
    ap.add_argument("--stop-after", choices=sorted(set(STEPS + CLONE_STEPS), key=(STEPS + CLONE_STEPS).index))
    ap.add_argument("--provider", choices=["gemini", "groq"], help="LLM for the script (default: from your API keys)")
    ap.add_argument("--model", help="Model for the script step")
    ap.add_argument("--cliffhanger", help="Stop the recap at this HH:MM:SS (skips the automatic search)")
    ap.add_argument("--opening-lock", action="store_true", help="Also find and lock a specific opening shot (brief: opening)")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--language")
    ap.add_argument("--engine", choices=["auto", "gemini", "edge"], default="edge",
                    help="Voice engine (default edge: the same free voice as your previous video)")
    ap.add_argument("--voice")
    ap.add_argument("--style", default="", help="Short delivery note for the Gemini voice")
    ap.add_argument("--target", type=float, default=116.0, help="Total video length in seconds (default 116 = 1:56)")
    ap.add_argument("--resolution", default="1920x1080", help="Output size (default full-screen widescreen)")
    ap.add_argument("--poster", help="Thumbnail for the end card (default: search assets/)")
    ap.add_argument("--music", help="Music bed file, or 'none' (default: a music file in assets/)")
    ap.add_argument("--check-footage", action="store_true", help="Vision-check and fix every shot (Gemini)")
    ap.add_argument("--vision", action="store_true", help="Let Gemini look at the first/last frame in the final check")
    args = ap.parse_args()

    if args.reference:
        return run_clone_flow(args)
    args.from_step = args.from_step or "transcribe"
    movies = list_movies()
    stem = movies[0].stem if movies else None
    brief = load_brief(args.brief, stem)
    brief_arg = ["--brief", args.brief] if args.brief else []
    title = args.title or brief.get("title")
    todo = STEPS[STEPS.index(args.from_step):]
    if args.stop_after:
        todo = todo[:todo.index(args.stop_after) + 1] if args.stop_after in todo else []
    anchors = (load_json(d("analysis") / f"{stem}_anchors.json", {}) or {}) if stem else {}

    if "transcribe" in todo:
        run_step("transcribe.py", "--model", args.whisper_model, *(["--language", args.language] if args.language else []))
    if "locate" in todo:
        wanted = []
        if brief.get("cliffhanger", {}).get("query") and not args.cliffhanger and "cliffhanger" not in anchors:
            wanted.append("cliffhanger")
        if args.opening_lock and brief.get("opening", {}).get("query") and "opening" not in anchors:
            wanted.append("opening")
        for name in wanted:
            rc = run_step("locate_scene.py", "--name", name, *brief_arg, allow_fail=True)
            if rc != 0:
                print(f"\n(Could not locate the '{name}' moment automatically. That step needs a Gemini key. "
                      "You can set it yourself:  python3 src/run_pipeline.py --cliffhanger HH:MM:SS )")
    if "analyze" in todo:
        run_step("analyze_movie.py", *(["--title", title] if title else []), *brief_arg,
                 *(["--cliffhanger", args.cliffhanger] if args.cliffhanger else []),
                 *(["--provider", args.provider] if args.provider else []), *(["--model", args.model] if args.model else []),
                 *([] if args.opening_lock else ["--no-opening-lock"]))
    if "footage" in todo and args.check_footage:
        run_step("verify_clips.py", "--fix")
    if "voiceover" in todo:
        run_step("voiceover.py", "--engine", args.engine, "--target", str(args.target),
                 *(["--voice", args.voice] if args.voice else []), *(["--style", args.style] if args.style else []))
    if "shots" in todo:
        run_step("cut_shots.py", "--resolution", args.resolution, *(["--poster", args.poster] if args.poster else []))
    if "render" in todo:
        run_step("render_final.py", *(["--music", args.music] if args.music else []))
    if "verify" in todo:
        rc = run_step("verify_video.py", *(["--vision"] if args.vision else []), allow_fail=True)
        print("\nPipeline finished. Open the images listed above and watch the video."
              + ("" if rc == 0 else "\nverify_video.py reported FAIL lines. Read them before publishing."))


if __name__ == "__main__":
    main()
