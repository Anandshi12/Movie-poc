#!/usr/bin/env python3
"""
verify_clips.py - check that the footage of every clip really shows what its narration says (Gemini vision).

Why: clip times are first chosen from DIALOGUE timestamps, but what the narrator describes and what is said in
the dialogue are not always the same moment. This step looks at actual frames.

Run it after analyze_movie.py and BEFORE narrate.py/cut_clips.py:
    python3 src/verify_clips.py            # report only
    python3 src/verify_clips.py --fix      # re-search mismatched clips between their neighbours and update the JSON

Cost: about 5 small images per clip, plus a frame search only for clips that do not match. The original analysis
is saved once as analysis/<stem>_analysis.pre_verify.json before anything is changed.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from common import (CLIFF_BUFFER_SEC, LEAD_SEC, TAIL_SEC, d, find_movie, need_ffmpeg, newest_analysis, probe_duration,
                    stem_of, to_hms, to_sec)
from llm import DEFAULT_MODEL, call_gemini, image_part, make_gemini_client as make_client
from locate_scene import SYSTEM, ClipCheck, extract_frames, find_scene


def check_clip(client, model, movie, clip):
    a, b = float(clip["clip_start_sec"]), float(clip["clip_end_sec"])
    need = len(clip["narration"].split()) / 2.6 + 0.3
    b = min(b, a + need)
    frames = extract_frames(movie, a, b, max(0.5, (b - a) / 4))
    contents = [f"Narration spoken over this footage:\n\"{clip['narration']}\"\n\nBelow are {len(frames)} consecutive "
                "frames of the footage. matches=true only if the frames plausibly show the event, characters and setting "
                "the narration describes (a part of the event is fine). If they show a different event, matches=false."]
    for t, jpg in frames:
        contents += [f"t={t:.1f}", image_part(jpg)]
    return call_gemini(client, model, SYSTEM, contents, ClipCheck, temperature=0.0)


def main():
    ap = argparse.ArgumentParser(description="Vision-check that footage matches narration.")
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--movie", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fix", action="store_true", help="Relocate clips that do not match")
    ap.add_argument("--min-confidence", type=int, default=60, help="Minimum confidence to accept a relocated clip")
    args = ap.parse_args()
    need_ffmpeg()

    apath = args.analysis or newest_analysis()
    data = json.loads(apath.read_text(encoding="utf-8"))
    stem = stem_of(apath)
    movie = find_movie(stem, args.movie)
    client, clips = make_client(), data["clips"]
    limit = to_sec(data["cliffhanger"]["timestamp"]) - CLIFF_BUFFER_SEC
    movie_len = probe_duration(movie)
    changed = bad = 0

    for i, clip in enumerate(clips):
        res = check_clip(client, args.model, movie, clip)
        clip["visual_check"] = {"matches": res.matches, "confidence": res.confidence, "shown": res.shown}
        mark = "OK " if res.matches else "BAD"
        print(f"  [{mark}] segment {clip['segment']:2d} ({to_hms(clip['clip_start_sec'], 1)}): {res.shown}")
        if res.matches:
            continue
        bad += 1
        if not args.fix:
            continue
        lo = float(clips[i - 1]["clip_end_sec"]) if i else 0.0
        hi = min(float(clips[i + 1]["clip_start_sec"]) if i + 1 < len(clips) else limit, limit, movie_len)
        if hi - lo > 240:
            mid = float(clip["clip_start_sec"])
            lo, hi = max(lo, mid - 120), min(hi, mid + 120)
        query = "Footage showing this: " + " ".join(re.split(r"(?<=[.!?])\s+", clip["narration"])[:2])
        print(f"      searching {to_hms(lo)}-{to_hms(hi)} for a better match ...")
        found = find_scene(client, args.model, movie, query, lo, hi, coarse=3.0)
        if not found or found["confidence"] < args.min_confidence:
            print("      no confident match; left unchanged (edit its times by hand or rewrite that narration)")
            continue
        length = max(float(clip["clip_end_sec"]) - float(clip["clip_start_sec"]),
                     len(clip["narration"].split()) / 2.6 + 1.3)
        new_a = found["start_sec"]
        new_b = min(new_a + length, hi)
        if new_b - new_a < 3:
            print("      match found too close to the next clip; left unchanged")
            continue
        clip.update({"clip_start_sec": round(new_a, 2), "clip_end_sec": round(new_b, 2),
                     "clip_start": to_hms(new_a, 1), "clip_end": to_hms(new_b, 1),
                     "clip_duration_sec": round(new_b - new_a, 1)})
        clip["visual_check"]["relocated_from"] = "see analysis/*_analysis.pre_verify.json"
        changed += 1
        print(f"      moved to {to_hms(new_a, 1)} (confidence {found['confidence']}): {found['evidence']}")

    if changed:
        backup = apath.with_name(f"{stem}_analysis.pre_verify.json")
        if not backup.exists():
            backup.write_text(apath.read_text(encoding="utf-8"), encoding="utf-8")
    apath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(clips) - bad}/{len(clips)} clips match their narration"
          + (f"; relocated {changed}." if args.fix else ("." if not bad else ". Re-run with --fix to try relocating the rest.")))
    if bad and not args.fix:
        sys.exit(1)


if __name__ == "__main__":
    main()
