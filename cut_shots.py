#!/usr/bin/env python3
"""
cut_shots.py - cut the footage for every narration slot, exactly like the reference edit: HARD CUTS on the narration.

Pipeline position:   input/ + analysis/ + narration/<stem>/timeline.json  ->  [cut_shots.py]  ->  clips/<stem>/

What it does
  * Each shot lasts exactly as long as its narration slot (cut points come from timeline.json), measured in whole
    video frames, so the picture never drifts away from the voice.
  * The footage is reframed to the output shape (full-screen 16:9, 1920x1080, by default; a 16:9 film needs no
    cropping). For other shapes, with --crop auto (default) the crop window slides sideways to the most detailed
    part of the frame (usually the characters); override any shot with
    "crop_x": 0.0-1.0 on that clip in the analysis JSON (0 = left edge, 0.5 = centre, 1 = right edge).
  * Pure-black frames at the start/end of a source window are trimmed; a short source window is extended into the
    unused footage before the next clip (never past the cliffhanger, never into footage already used). A gap of up to
    8% is covered by an almost invisible slow-down; a bigger gap freezes the last frame and prints a WARNING.
  * The last slot (the call to action) becomes the end card: your thumbnail shown whole, held after the last word.
  * Shots are silent: the voiceover (and optional music) is added by render_final.py.

Outputs:  clips/<stem>/shot_001.mp4 ... end_card.mp4, manifest.json. Unchanged shots are skipped on re-runs.

Usage (from the project root):
    python3 src/cut_shots.py
    python3 src/cut_shots.py --resolution 1920x1080 --crop center
    python3 src/cut_shots.py --poster assets/tangled_thumbnail.png --cta-text "Watch the full movie"
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from common import (CLIFF_BUFFER_SEC, FPS, PIPELINE_VERSION, assets_images, d, file_fp, find_movie, find_poster,
                    fingerprint, load_brief, load_json, need_ffmpeg, newest_analysis, pick_font, probe_duration,
                    run, stem_of, to_hms, to_sec, validate_analysis)

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError:
    sys.exit("Missing dependency. Run:  pip install pillow")

from cut_clips import best_frame, make_endcard, trim_black   # shared helpers (end card, black trimming)


# --------------------------------------------------------------------------- #
def plan_shot(movie, clip, need, lo, hi, trim):
    """Choose the source seconds for a shot that must last `need` seconds."""
    a, b = float(clip["clip_start_sec"]), float(clip["clip_end_sec"])
    notes = []
    if trim:
        a0, b0 = a, b
        a, b, n = trim_black(movie, a, b, need)
        if n:
            notes.append(n)
            if a > a0:
                lo = max(lo, a)            # black was removed at the start: never extend backwards into it
            if b < b0:
                hi = min(hi, b)            # black was removed at the end: never extend forwards into it
    slow, freeze, warn = 1.0, 0.0, None
    if b - a >= need:
        b = a + need
    else:
        b2 = min(a + need, hi)
        if b2 - b > 0.05:
            notes.append(f"extended {b2 - b:.1f}s forward")
        b = b2
        if b - a < need:
            a2 = max(lo, b - need)
            if a - a2 > 0.05:
                notes.append(f"extended {a - a2:.1f}s backward")
            a = a2
        short = need - (b - a)
        if short > 0.03:
            if short / max(b - a, 0.1) <= 0.08:
                slow = need / (b - a)
                notes.append(f"slowed {100 * (slow - 1):.1f}%")
            else:
                freeze = short
                warn = (f"shot {clip['segment']}: only {b - a:.1f}s of unused footage for {need:.1f}s of narration; "
                        f"froze the last frame for {short:.1f}s. Widen this clip's window or shorten its narration.")
    return {"a": a, "length": b - a, "slow": slow, "freeze": freeze, "need": need, "notes": notes, "warn": warn}


def auto_crop_x(movie, t, ar):
    """Fraction (0 left .. 1 right) at which to place the crop window: the most detailed part of the frame."""
    tmp = Path(tempfile_name())
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(movie), "-frames:v", "1",
                        "-vf", "scale=480:-2", str(tmp)], capture_output=True)
        if not tmp.exists():
            return 0.5
        img = Image.open(tmp).convert("L")
    finally:
        tmp.unlink(missing_ok=True)
    w, h = img.size
    win = int(h * ar)
    if win >= w - 4:
        return 0.5
    edges = img.filter(ImageFilter.FIND_EDGES)
    scores = []
    for k in range(9):
        x = int((w - win) * k / 8)
        scores.append(ImageStat.Stat(edges.crop((x, 0, x + win, h))).mean[0])
    best = max(range(9), key=lambda k: scores[k])
    return best / 8 if scores[best] > scores[4] * 1.15 else 0.5     # stay centred unless clearly better


def tempfile_name():
    import tempfile
    return tempfile.mktemp(suffix=".jpg", prefix="crop_")


def shot_command(movie, p, frames, w, h, fit, fx, out):
    pad = max(0.6, p["freeze"] + 0.5)          # frames the source cannot supply are filled with the last frame
    if fit == "pad":
        vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    else:
        ar = w / h
        vf = (f"crop=w='min(iw,ih*{ar:.5f})':h='min(ih,iw/{ar:.5f})':x='(iw-ow)*{fx:.3f}':y='(ih-oh)*0.5',"
              f"scale={w}:{h}:flags=lanczos,setsar=1")
    if p["slow"] > 1.0:
        vf += f",setpts={p['slow']:.5f}*PTS"
    vf += f",fps={FPS},tpad=stop_mode=clone:stop_duration={pad:.3f},format=yuv420p"
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{p['a']:.3f}", "-t", f"{p['length']:.3f}", "-i", str(movie),
            "-an", "-vf", vf, "-frames:v", str(frames), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


def card_command(card_png, frames, w, h, out):
    secs = frames / FPS
    fc = (f"[0:v]scale={2 * w}:{2 * h},zoompan=z='1+0.04*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
          f"d={frames}:s={w}x{h}:fps={FPS},format=yuv420p[v]")
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", str(card_png), "-filter_complex", fc, "-map", "[v]",
            "-frames:v", str(frames), "-t", f"{secs:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]


def squash(s):
    return re.sub(r"\W+", "", s.lower())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Cut frame-exact hard-cut shots from the movie and build the end card.")
    ap.add_argument("--analysis", type=Path)
    ap.add_argument("--movie", type=Path)
    ap.add_argument("--resolution", default="1920x1080", help="Output size WxH (default full-screen widescreen 1920x1080)")
    ap.add_argument("--fit", choices=["crop", "pad"], default="crop", help="crop = fill the frame (like the reference); "
                                                                          "pad = keep the whole picture with bars")
    ap.add_argument("--crop", choices=["auto", "center"], default="auto")
    ap.add_argument("--poster", type=Path, help="Thumbnail/poster image for the end card (default: search assets/)")
    ap.add_argument("--add-title", action="store_true", help="Draw the title on the end card even if the art has it")
    ap.add_argument("--cta-text", default="Watch the full movie")
    ap.add_argument("--font", help="Font file (.ttf) for the end card text")
    ap.add_argument("--no-trim-black", action="store_true")
    ap.add_argument("--allow-issues", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    need_ffmpeg()
    try:
        w, h = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        sys.exit("--resolution must look like 1920x1080")
    if w % 2 or h % 2:
        sys.exit("--resolution width and height must be even numbers")

    apath = args.analysis or newest_analysis()
    data = json.loads(apath.read_text(encoding="utf-8"))
    stem = stem_of(apath)
    brief = load_brief(None, stem)
    movie = find_movie(stem, args.movie)
    movie_len = probe_duration(movie)
    tl = load_json(d("narration") / stem / "timeline.json")
    if not tl:
        sys.exit(f"No narration/{stem}/timeline.json. Run  python3 src/voiceover.py  first.")
    clips, segs = data["clips"], tl["segments"]
    if len(segs) != len(clips) + 1 or segs[-1]["id"] != "cta":
        sys.exit("The voiceover timeline does not match the analysis (different number of segments). "
                 "Run  python3 src/voiceover.py --force  again.")
    for c, s in zip(clips, segs):
        if squash(c["narration"]) != squash(s["text"]):
            sys.exit(f"The narration text of segment {c['segment']} changed after the voiceover was made. "
                     "Run  python3 src/voiceover.py  again.")
    problems = validate_analysis(data, movie_len)
    if problems:
        print("Problems in the analysis JSON:\n  - " + "\n  - ".join(problems))
        if not args.allow_issues:
            sys.exit("Fix them (edit clip times in analysis/<stem>_analysis.json or re-run analyze_movie.py), "
                     "or pass --allow-issues.")
    limit = min(to_sec(data["cliffhanger"]["timestamp"]) - CLIFF_BUFFER_SEC, movie_len - 0.3)

    # end-card artwork first, so a wrong path fails before any cutting
    poster = find_poster(args.poster, brief)
    if args.poster and not poster:
        sys.exit(f"The poster/thumbnail file was not found: {args.poster}\n"
                 f"  Save the image inside {d('assets', create=True)} (e.g. assets/tangled_thumbnail.png) and run again.")
    draw_title = args.add_title or not (poster and brief.get("poster_has_title", True))
    if poster:
        print(f"End card artwork: {poster}" + (" (title will be drawn on it)" if draw_title else ""))
    else:
        print(f"No poster/thumbnail in {d('assets')} (images there: {[x.name for x in assets_images()] or 'none'}). "
              "Using a frame from the movie as the end card.")

    out_dir = d("clips", create=True) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    old = (load_json(out_dir / "manifest.json", {}) or {}).get("clips", {})
    new, order, durations, warnings = {}, [], {}, []

    def save(complete):
        total_frames = sum(v["frames"] for v in new.values())
        man = {"version": PIPELINE_VERSION, "flow": "reference", "title": data["title"], "order": list(order),
               "durations": dict(durations), "clips": dict(new), "warnings": list(warnings), "complete": complete,
               "size": [w, h], "fps": FPS, "expected_total_sec": round(total_frames / FPS, 3)}
        (out_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        return man

    movie_fp, prev_end, prev_frames = file_fp(movie), 0.0, 0
    ar = w / h
    print(f"Movie: {movie.name} ({to_hms(movie_len)}) | {len(clips)} shots + end card | {w}x{h} @ {FPS} fps | "
          f"spoiler limit {to_hms(limit)}")
    for i, (clip, seg) in enumerate(zip(clips, segs)):
        end_frame = round(seg["cut_end"] * FPS)
        frames = max(4, end_frame - prev_frames)
        prev_frames += frames
        need = frames / FPS
        if clip.get("exact"):
            # a reference clone: the source window is exact. Never borrow footage from outside it.
            p = plan_shot(movie, clip, need, float(clip["clip_start_sec"]), float(clip["clip_end_sec"]), False)
        else:
            hi = min(limit, float(clips[i + 1]["clip_start_sec"]) - 0.1 if i + 1 < len(clips) else limit)
            p = plan_shot(movie, clip, need, prev_end, hi, not args.no_trim_black)
        prev_end = p["a"] + p["length"]
        fx = clip.get("crop_x")
        if fx is None:
            fx = 0.5 if (args.crop == "center" or args.fit == "pad") else auto_crop_x(movie, p["a"] + p["length"] / 2, ar)
        name = f"shot_{i + 1:03d}.mp4"
        fp = fingerprint(PIPELINE_VERSION, movie_fp, round(p["a"], 3), round(p["length"], 3), round(p["slow"], 4),
                         round(p["freeze"], 3), frames, w, h, args.fit, round(float(fx), 3))
        status = "kept"
        if args.force or not (out_dir / name).exists() or old.get(name, {}).get("fp") != fp:
            run(shot_command(movie, p, frames, w, h, args.fit, float(fx), out_dir / name), f"Cutting {name}")
            status = "built"
        order.append(name)
        durations[name] = round(need, 3)
        new[name] = {"fp": fp, "frames": frames, "source_start": round(p["a"], 2), "source_end": round(prev_end, 2),
                     "crop_x": round(float(fx), 3), "notes": p["notes"]}
        if p["warn"]:
            warnings.append(p["warn"])
        save(False)
        print(f"  {name} [{status:5s}] {need:4.1f}s  src {to_hms(p['a'], 1)}-{to_hms(prev_end, 1)}  crop_x {float(fx):.2f}"
              + (f"  ({'; '.join(p['notes'])})" if p["notes"] else "") + ("  WARNING" if p["warn"] else ""))

    # end card (the call-to-action slot + the hold after the last word)
    card_frames = max(FPS, round(segs[-1]["cut_end"] * FPS) - prev_frames)
    if poster:
        art, note = poster, f"poster: {poster.name}"
    else:
        art = out_dir / "endcard_frame.jpg"
        mids = [(float(c["clip_start_sec"]) + float(c["clip_end_sec"])) / 2 for c in clips]
        note = f"movie frame at {to_hms(best_frame(movie, mids, art), 1)}"
        warnings.append("No poster/thumbnail found, so the end card uses a frame from the movie. Save your thumbnail in "
                        "assets/ (for example assets/tangled_thumbnail.png).")
    font = pick_font(args.font)
    card_fp = fingerprint(PIPELINE_VERSION, file_fp(art), data["title"], args.cta_text, card_frames, w, h, font, draw_title)
    out, png = out_dir / "end_card.mp4", out_dir / "endcard.png"
    status = "kept"
    if args.force or not out.exists() or old.get("end_card.mp4", {}).get("fp") != card_fp:
        make_endcard(data["title"], art, w, h, font, png, args.cta_text, draw_title)
        run(card_command(png, card_frames, w, h, out), "Building the end card")
        status = "built"
    order.append("end_card.mp4")
    durations["end_card.mp4"] = round(card_frames / FPS, 3)
    new["end_card.mp4"] = {"fp": card_fp, "frames": card_frames, "art": str(art)}
    print(f"  end_card.mp4 [{status:5s}] {card_frames / FPS:4.1f}s  ({note})")
    man = save(True)
    print(f"\nDone. {len(order)} shots in {out_dir}; video length will be {man['expected_total_sec']:.1f}s "
          f"({man['expected_total_sec'] / 60:.2f} min).")
    for msg in warnings:
        print("  WARNING:", msg)


if __name__ == "__main__":
    main()
