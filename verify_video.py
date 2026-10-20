#!/usr/bin/env python3
"""
verify_video.py - check the rendered video and produce images you can look at.

Automatic checks: length, streams, black frames, frozen frames, loudness/true peak, long silences,
and (if clips/<stem>/manifest.json exists) that the length matches the planned clip list.

It also writes, to output/<stem>_verify/:
    first_frame.png   -> the very first picture of the video (should be the golden drop)
    last_frame.png    -> the last picture (should be the poster end card)
    contact_sheet.jpg -> one frame every few seconds, to eyeball order and sync
    report.json

With --vision it also asks Gemini whether the first shot matches the opening described in the brief and whether the
last frame is a poster-style end card. That is an AI judgement, so it is a second opinion, not proof.

Usage (from the project root):
    python3 src/verify_video.py
    python3 src/verify_video.py --vision
Exit code is 1 if any check FAILS.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from common import d, load_brief, load_json, need_ffmpeg, probe_duration, run, to_hms


def ff_filter_log(video, vf=None, af=None):
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video)]
    cmd += ["-vf", vf, "-an"] if vf else ["-af", af, "-vn"]
    return subprocess.run(cmd + ["-f", "null", "-"], capture_output=True, text=True).stderr


def main():
    ap = argparse.ArgumentParser(description="Verify the rendered recap video.")
    ap.add_argument("--video", type=Path, help="Video to check (default: newest *_recap.mp4 in output/)")
    ap.add_argument("--min-sec", type=float, help="Shortest acceptable length (default depends on the recap style)")
    ap.add_argument("--max-sec", type=float, help="Longest acceptable length")
    ap.add_argument("--vision", action="store_true", help="Ask Gemini to look at the first and last frames")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    need_ffmpeg()

    video = args.video
    if not video:
        outs = sorted(d("output").glob("*_recap*.mp4"), key=lambda p: p.stat().st_mtime) if d("output").exists() else []
        if not outs:
            sys.exit("No *_recap.mp4 in output/. Run src/render_video.py first.")
        video = outs[-1]
    stem = video.stem.split("_recap")[0]
    dur = probe_duration(video)
    results = []
    analysis = load_json(d("analysis") / f"{stem}_analysis.json", {}) or {}
    lo, hi = {"reference": (90, 170), "long": (180, 300)}.get(analysis.get("style"), (90, 300))
    args.min_sec, args.max_sec = args.min_sec or lo, args.max_sec or hi

    def add(status, name, detail):
        results.append({"status": status, "check": name, "detail": detail})
        print(f"  [{status}] {name}: {detail}")

    print(f"Checking {video.name} ({to_hms(dur)})")
    add("PASS" if args.min_sec <= dur <= args.max_sec else "WARN", "length",
        f"{dur:.1f}s ({dur / 60:.1f} min); target {args.min_sec:.0f}-{args.max_sec:.0f}s")

    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
                              "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout
    add("PASS" if "video" in streams and "audio" in streams else "FAIL", "streams", streams.replace("\n", " ").strip())

    # black frames
    log = ff_filter_log(video, vf="blackdetect=d=0.12:pix_th=0.04")
    blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+)\s+black_end:([\d.]+)", log)]
    add("PASS" if not blacks else "FAIL", "black frames",
        "none" if not blacks else ", ".join(f"{to_hms(a, 1)}-{to_hms(b, 1)}" for a, b in blacks[:8]))

    # frozen frames (the end card's slow zoom is not a freeze; dark static shots can false-trigger)
    log = ff_filter_log(video, vf="freezedetect=n=-60dB:d=1.2")
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", log)]
    lens = [float(x) for x in re.findall(r"freeze_duration: ([\d.]+)", log)]
    freezes = [(s, l) for s, l in zip(starts, lens) if s + l < dur - 0.5 or l > 3]
    add("PASS" if not freezes else "WARN", "frozen frames",
        "none" if not freezes else "; ".join(f"{to_hms(s, 1)} for {l:.1f}s" for s, l in freezes[:8])
        + "  (very dark or very still shots can trigger this; check them in the contact sheet)")

    # loudness
    log = ff_filter_log(video, af="loudnorm=print_format=json")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", log, re.S)
    if m:
        j = json.loads(m.group(0))
        lufs, tp = float(j["input_i"]), float(j["input_tp"])
        add("PASS" if -19 <= lufs <= -13 and tp <= -0.5 else "WARN", "loudness",
            f"{lufs:.1f} LUFS, true peak {tp:.1f} dBTP (target about -16 LUFS, peak below -1)")
    else:
        add("WARN", "loudness", "could not measure")

    # silences
    log = ff_filter_log(video, af="silencedetect=n=-50dB:d=1.5")
    s_starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    s_ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    sil = [(a, b) for a, b in zip(s_starts, s_ends) if b < dur - 0.6]   # the silent hold at the end card is intentional
    add("PASS" if not sil else "WARN", "audio gaps >1.5s",
        "none" if not sil else ", ".join(f"{to_hms(a, 1)}-{to_hms(b, 1)}" for a, b in sil[:8]))

    # manifest
    man = load_json(d("clips") / stem / "manifest.json") if d("clips").exists() else None
    if man and man.get("expected_total_sec"):
        diff = abs(dur - man["expected_total_sec"])
        add("PASS" if diff <= 1.0 else "FAIL", "matches clip plan",
            f"{len(man['order'])} clips in order, expected {man['expected_total_sec']:.1f}s, got {dur:.1f}s")
        if man.get("size"):
            dims = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                                   "-of", "csv=p=0:s=x", str(video)], capture_output=True, text=True).stdout.strip()
            want = f"{man['size'][0]}x{man['size'][1]}"
            add("PASS" if dims == want else "FAIL", "picture size", f"{dims} (planned {want})")
        for w in man.get("warnings", []):
            add("WARN", "cut_clips warning", w)

    # images
    folder = d("output", create=True) / f"{stem}_verify"
    folder.mkdir(exist_ok=True)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-frames:v", "1", folder / "first_frame.png"], "first frame")
    run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", video, "-frames:v", "1", folder / "last_frame.png"],
        "last frame")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf",
         f"fps=1/{max(2.0, dur / 24):.2f},scale=384:-2,tile=6x4:padding=4", "-frames:v", "1", folder / "contact_sheet.jpg"],
        "contact sheet")

    if args.vision:
        from pydantic import BaseModel, Field
        from llm import DEFAULT_MODEL, call_gemini, image_part, make_gemini_client as make_client

        class VideoCheck(BaseModel):
            first_shot_matches_opening: bool = Field(description="First frames show the described opening shot")
            first_shot_seen: str
            last_frame_is_poster_end_card: bool = Field(description="Last frame is a poster-style end card showing the title")
            last_frame_seen: str

        brief = load_brief(None, stem)
        query = brief.get("opening", {}).get("query", "the opening shot of the movie")
        title = brief.get("title", stem)
        frames = []
        for t in (0.0, 0.4, 1.2):
            f = folder / f"_f{t}.jpg"
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", t, "-i", video, "-frames:v", "1", "-vf", "scale=640:-2", f], "frame")
            frames.append(f)
        last = folder / "_last.jpg"
        run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", video, "-frames:v", "1", "-vf", "scale=640:-2", last],
            "frame")
        contents = [f"The first 3 images are the first moments of a video. Does the first shot show: {query}? "
                    f"The 4th image is the final frame. Is it a movie-poster-style end card clearly showing the title '{title}'?"]
        for f in frames + [last]:
            contents.append(image_part(f.read_bytes()))
        r = call_gemini(make_client(), args.model or DEFAULT_MODEL, "You are a careful video QA reviewer.", contents,
                        VideoCheck, temperature=0.0)
        if analysis.get("opening_start_sec") is not None:
            add("PASS" if r.first_shot_matches_opening else "FAIL", "AI: opening shot", r.first_shot_seen)
        else:
            add("INFO", "opening shot", f"no opening lock was set, so it is not judged. First shot: {r.first_shot_seen}")
        add("PASS" if r.last_frame_is_poster_end_card else "FAIL", "AI: end card", r.last_frame_seen)
        for f in frames + [last]:
            f.unlink(missing_ok=True)

    (folder / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fails = sum(r["status"] == "FAIL" for r in results)
    print(f"\nImages to look at: {folder}\n  first_frame.png, last_frame.png, contact_sheet.jpg")
    print("Result:", "FAIL" if fails else "no failures (check any WARN lines and the images yourself)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
