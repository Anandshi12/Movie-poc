#!/usr/bin/env python3
"""
render_video.py - join the clips into the final recap video.

Pipeline position:   clips/<stem>/  ->  [render_video.py]  ->  output/<stem>_recap.mp4

* Clips are joined in the exact order written by cut_clips.py (clip_01, clip_02, ... then cta.mp4). Nothing is
  globbed, so a stale or leftover clip file can never sneak in. Missing clips stop the render.
* Short crossfades (0.25 s) between clips: no black frames, no audio gaps. The narration starts after each
  crossfade, so no words are lost.
* The finished audio is loudness-normalized (about -16 LUFS, YouTube-friendly) with a peak limiter.
* Refuses to render if the analysis or narration changed after the clips were cut (stale clips).

Usage (from the project root):
    python3 src/render_video.py
    python3 src/render_video.py --transition cut      # fast, lossless joins without crossfades
"""

import argparse
import sys
from pathlib import Path

from common import (XFADE_SEC, d, load_json, need_ffmpeg, probe_duration, run, to_hms)


def newest_clip_folder() -> Path:
    base = d("clips")
    dirs = [x for x in base.glob("*") if (x / "manifest.json").exists()] if base.exists() else []
    if not dirs:
        sys.exit(f"No cut clips found in {base}. Run src/cut_clips.py first.")
    return max(dirs, key=lambda x: (x / "manifest.json").stat().st_mtime)


def crossfade_graph(durs, x, loudnorm):
    n = len(durs)
    parts = [f"[{i}:v]settb=AVTB,fps=30,format=yuv420p[v{i}]" for i in range(n)]
    vprev, aprev, acc = "v0", "0:a", durs[0]
    for k in range(1, n):
        parts.append(f"[{vprev}][v{k}]xfade=transition=fade:duration={x}:offset={acc - x:.3f}[xv{k}]")
        parts.append(f"[{aprev}][{k}:a]acrossfade=d={x}:c1=tri:c2=tri[xa{k}]")
        vprev, aprev, acc = f"xv{k}", f"xa{k}", acc + durs[k] - x
    if loudnorm:
        parts.append(f"[{aprev}]loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
        aprev = "aout"
    else:
        parts.append(f"[{aprev}]anull[aout]")
        aprev = "aout"
    return ";".join(parts), vprev, aprev


def main():
    ap = argparse.ArgumentParser(description="Join narrated clips into the final video.")
    ap.add_argument("--clips", type=Path, help="Clip folder (default: newest folder in clips/)")
    ap.add_argument("--transition", choices=["crossfade", "cut"], default="crossfade")
    ap.add_argument("--no-loudnorm", action="store_true", help="Skip loudness normalization")
    ap.add_argument("--allow-stale", action="store_true", help="Render even if the analysis/narration is newer than the clips")
    args = ap.parse_args()
    need_ffmpeg()

    folder = args.clips or newest_clip_folder()
    stem = folder.name
    manifest_path = folder / "manifest.json"
    manifest = load_json(manifest_path)
    if not manifest or not manifest.get("order"):
        sys.exit(f"{manifest_path} is missing or empty. Run src/cut_clips.py first.")
    if not manifest.get("complete", True):
        sys.exit("cut_clips.py did not finish, so the clip list is incomplete. Run  python3 src/cut_clips.py  again "
                 "(clips that are already built are reused).")
    order = manifest["order"]
    files = [folder / n for n in order]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"Missing clip files: {missing}. Run src/cut_clips.py again.")
    nums = [int(n[5:7]) for n in order if n.startswith("clip_")]
    if nums != list(range(1, len(nums) + 1)) or order[-1] != "cta.mp4":
        sys.exit(f"Clip order is wrong: {order}. Run src/cut_clips.py again.")

    # stale-output guard
    mtime = manifest_path.stat().st_mtime
    newer = [p.name for p in [d("analysis") / f"{stem}_analysis.json", d("narration") / stem / "manifest.json"]
             if p.exists() and p.stat().st_mtime > mtime + 1]
    if newer and not args.allow_stale:
        sys.exit(f"{newer} changed after the clips were cut. Run  python3 src/cut_clips.py  first "
                 "(unchanged clips are skipped), or pass --allow-stale.")

    out_dir = d("output", create=True)
    out = out_dir / f"{stem}_recap.mp4"
    durs = [probe_duration(f) for f in files]
    print(f"Joining {len(files)} clips from {folder.name}/ ({args.transition}) ...")

    if args.transition == "cut":
        lst = folder / "concat_list.txt"
        lst.write_text("".join(f"file '{f.resolve().as_posix()}'\n" for f in files), encoding="utf-8")
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
             "-movflags", "+faststart", out], "joining clips")
        expected = sum(durs)
    else:
        graph, vlab, alab = crossfade_graph(durs, XFADE_SEC, not args.no_loudnorm)
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for f in files:
            cmd += ["-i", str(f)]
        cmd += ["-filter_complex", graph, "-map", f"[{vlab}]", "-map", f"[{alab}]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-movflags", "+faststart", str(out)]
        run(cmd, "rendering")
        expected = sum(durs) - XFADE_SEC * (len(durs) - 1)

    total = probe_duration(out)
    print(f"\nDone. {out}\nLength: {to_hms(total)} ({total / 60:.1f} min)")
    if abs(total - expected) > 1.0:
        print(f"WARNING: expected about {expected:.1f}s but got {total:.1f}s. Run src/verify_video.py.")
    for w in manifest.get("warnings", []):
        print("  Note from cut_clips:", w)
    print("Next: python3 src/verify_video.py")


if __name__ == "__main__":
    main()
