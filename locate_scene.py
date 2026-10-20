#!/usr/bin/env python3
"""
locate_scene.py - find WHERE something happens on screen and save it as an "anchor".

Why this exists: the transcript only contains dialogue, so a purely visual moment (like the golden
drop of sunlight falling at the start of Tangled) can never be found from text. This script samples
frames from the movie, asks Gemini vision which frame first shows the described scene, then snaps to
the nearest shot cut so the clip starts on a clean first frame.

Result is cached in analysis/<stem>_anchors.json (so it never calls the API twice for the same thing).

Usage (from the project root):
    python3 src/locate_scene.py                       # uses opening.query from assets/<stem>_brief.json
    python3 src/locate_scene.py --name cliffhanger    # uses cliffhanger.query: where the recap should stop
    python3 src/locate_scene.py --query "a drop of golden sunlight falling from the sky" --window 0 420
    python3 src/locate_scene.py --start 00:01:12.5    # you already know the time: just store it
    python3 src/locate_scene.py --force               # search again

It also writes analysis/<stem>_opening_preview.jpg (frames from the found moment). LOOK AT IT before continuing.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from common import (d, file_fp, find_movie, fingerprint, load_brief, load_json, need_ffmpeg,
                    probe_duration, run, to_hms, to_sec)
from llm import DEFAULT_MODEL, call_gemini, image_part, make_gemini_client as make_client

SYSTEM = ("You are a meticulous film editor. You are shown still frames from a movie, each preceded by its "
          "timestamp in seconds. Answer strictly from what is visible in the frames.")


class SceneHit(BaseModel):
    found: bool = Field(description="True only if the described scene is clearly visible in at least one frame")
    start: str = Field(description="Timestamp label (exactly as given) of the FIRST frame showing the scene")
    confidence: int = Field(description="0-100")
    evidence: str = Field(description="One sentence: what is visible in that frame")


class ClipCheck(BaseModel):
    matches: bool = Field(description="True if the frames clearly show what the narration describes")
    confidence: int = Field(description="0-100")
    shown: str = Field(description="One sentence: what the frames actually show")


def extract_frames(movie, t0: float, t1: float, step: float, width: int = 384):
    """Return [(timestamp_sec, jpeg_bytes)] sampled every `step` seconds from t0 to t1."""
    tmp = Path(tempfile.mkdtemp(prefix="frames_"))
    try:
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-t", f"{max(0.1, t1 - t0):.3f}",
             "-i", movie, "-vf", f"fps={1 / step:.5f},scale={width}:-2", "-q:v", "5", tmp / "f_%05d.jpg"],
            "frame extraction")
        return [(t0 + i * step, p.read_bytes()) for i, p in enumerate(sorted(tmp.glob("f_*.jpg")))]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def snap_to_cut(movie, t: float, radius: float = 2.0, threshold: float = 0.25) -> float:
    """Return the nearest shot cut at or just before t (within radius), else t unchanged."""
    a = max(0.0, t - radius)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{a:.3f}", "-t", f"{2 * radius:.3f}",
                        "-i", str(movie), "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    cuts = [a + float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr)]
    cuts = [c for c in cuts if c <= t + 0.3]
    return max(cuts) if cuts else t


def ask_scene(client, model, frames, query: str):
    contents = [f"Below are {len(frames)} frames. Find the FIRST frame where this is clearly visible:\n"
                f"{query}\n\nBe strict: similar-looking things do not count. If it never appears, set found=false. "
                "'start' must be one of the timestamp labels exactly as written."]
    for t, jpg in frames:
        contents += [f"t={t:.2f}", image_part(jpg)]
    return call_gemini(client, model, SYSTEM, contents, SceneHit, temperature=0.0)


def find_scene(client, model, movie, query: str, t0: float, t1: float, coarse: float = 2.0):
    t1 = min(t1, probe_duration(movie))
    step = max(coarse, (t1 - t0) / 300)                    # cap at ~300 frames per request
    hit = ask_scene(client, model, extract_frames(movie, t0, t1, step), query)
    if not hit.found:
        return None
    c = to_sec(hit.start)
    fine = extract_frames(movie, max(t0, c - 1.5 * step), min(t1, c + 1.5 * step), 0.25)
    hit2 = ask_scene(client, model, fine, query) if fine else None
    ok2 = bool(hit2 and hit2.found)
    start = snap_to_cut(movie, to_sec(hit2.start) if ok2 else c)
    return {"start_sec": round(start, 2), "start": to_hms(start, 1),
            "confidence": min(hit.confidence, hit2.confidence) if ok2 else hit.confidence,
            "evidence": (hit2.evidence if ok2 else hit.evidence)}


def write_preview(movie, start: float, out: Path):
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, start - 0.5):.3f}", "-t", "8", "-i", movie,
         "-vf", "fps=1,scale=480:-2,tile=4x2:padding=4", "-frames:v", "1", out], "preview")


def main():
    ap = argparse.ArgumentParser(description="Locate a visual scene in the movie and store it as an anchor.")
    ap.add_argument("--movie", type=Path)
    ap.add_argument("--brief", type=Path)
    ap.add_argument("--name", default="opening")
    ap.add_argument("--query", help="What to look for on screen")
    ap.add_argument("--window", type=float, nargs=2, metavar=("START", "END"), help="Seconds range to search")
    ap.add_argument("--start", help="Skip the search and store this timestamp (HH:MM:SS[.s])")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    need_ffmpeg()
    movie = find_movie(explicit=args.movie)
    stem = movie.stem
    brief = load_brief(args.brief, stem)
    ob = brief.get(args.name, {})
    query = args.query or ob.get("query")
    window = args.window or ob.get("window_sec") or [0, 420]
    path = d("analysis", create=True) / f"{stem}_anchors.json"
    anchors = load_json(path, {}) or {}

    if args.start:
        anchors[args.name] = {"start_sec": round(to_sec(args.start), 2), "start": to_hms(to_sec(args.start), 1),
                              "manual": True, "query": query}
    else:
        if not query:
            sys.exit("Nothing to look for. Pass --query, or add <name>.query to assets/<stem>_brief.json.")
        fp = fingerprint(query, window, file_fp(movie), args.model)
        old = anchors.get(args.name, {})
        if old and (old.get("manual") or old.get("fp") == fp) and not args.force:
            print(f"Anchor '{args.name}' already stored: {old['start']} ({'manual' if old.get('manual') else 'cached'}). "
                  "Use --force to search again.")
            return
        print(f"Searching {to_hms(window[0])}-{to_hms(window[1])} for: {query}")
        res = find_scene(make_client(), args.model, movie, query, window[0], window[1])
        if not res:
            sys.exit("The scene was not found in that window. Try a wider --window, a clearer --query, "
                     "or give the time yourself with --start HH:MM:SS.")
        anchors[args.name] = {**res, "query": query, "fp": fp}
        print(f"  Found at {res['start']} (confidence {res['confidence']}): {res['evidence']}")
        if res["confidence"] < 60:
            print("  Low confidence - check the preview image before continuing.")

    path.write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    prev = path.with_name(f"{stem}_{args.name}_preview.jpg")
    write_preview(movie, anchors[args.name]["start_sec"], prev)
    print(f"Stored anchor '{args.name}' = {anchors[args.name]['start']}  ->  {path.name}")
    print(f"Preview (frames from that moment): {prev}\nOpen it and confirm the first frame is the right shot.")


if __name__ == "__main__":
    main()
