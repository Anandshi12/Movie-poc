#!/usr/bin/env python3
"""
render_final.py - the final video: hard-cut shots + one continuous voiceover (+ optional music bed).

Pipeline position:   clips/<stem>/ + narration/<stem>/narration.wav  ->  [render_final.py]  ->  output/<stem>_recap.mp4

* Shots are joined in the exact order recorded by cut_shots.py, without re-encoding (no quality loss, no black
  frames, no audio gaps because the voice is one separate continuous track).
* Optional music bed: put a track you have the rights to in assets/ (for example assets/music.mp3) or pass --music.
  It plays under the voice, ducks automatically while the narrator speaks, fades in/out. Your reference has a steady
  bed under the voice; without a music file the video simply has voice only.
* Loudness is normalized to about -14 LUFS with a peak limiter (a common level for short-video platforms).
* Refuses to render if the voiceover or analysis changed after the shots were cut (stale shots).

Usage (from the project root):
    python3 src/render_final.py
    python3 src/render_final.py --music assets/music.mp3 --music-volume 0.18
    python3 src/render_final.py --music none
"""

import argparse
import re
import sys
from pathlib import Path

from common import d, load_json, need_ffmpeg, probe_duration, run, to_hms

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def find_music(explicit):
    if explicit:
        if explicit.lower() == "none":
            return None
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"Music file not found: {explicit}")
        return p
    base = d("assets")
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.suffix.lower() in AUDIO_EXTS and re.search(r"music|bed|bgm|score|ambient", p.stem, re.I):
                return p
    return None


def newest_clip_folder() -> Path:
    base = d("clips")
    dirs = [x for x in base.glob("*") if (x / "manifest.json").exists()] if base.exists() else []
    if not dirs:
        sys.exit(f"No cut shots found in {base}. Run src/cut_shots.py first.")
    return max(dirs, key=lambda x: (x / "manifest.json").stat().st_mtime)


def audio_graph(total, music, music_vol, loudness, do_norm, vi=1, mi=2):
    """vi / mi: input indexes of the voice and the music file on the ffmpeg command line."""
    parts = [f"[{vi}:a]aformat=sample_rates=44100:channel_layouts=stereo,apad=whole_dur={total:.3f}[voice]"]
    if music:
        parts += [
            f"[{mi}:a]aformat=sample_rates=44100:channel_layouts=stereo,aloop=loop=-1:size=2147483647,"
            f"atrim=0:{total:.3f},afade=t=in:d=1.5,afade=t=out:st={max(0.0, total - 2.5):.3f}:d=2.5,"
            f"volume={music_vol}[bed]",
            "[voice]asplit=2[v1][v2]",
            "[bed][v2]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=600:makeup=1[ducked]",
            "[v1][ducked]amix=inputs=2:duration=first:normalize=0[mix]"]
    else:
        parts.append("[voice]anull[mix]")
    tail = (f"loudnorm=I={loudness}:TP=-1.0:LRA=9," if do_norm else "") + \
           f"alimiter=limit=0.95,afade=t=out:st={max(0.0, total - 0.6):.3f}:d=0.6"
    parts.append(f"[mix]{tail}[a]")
    return ";".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Render the final recap video.")
    ap.add_argument("--clips", type=Path, help="Clip folder (default: newest in clips/)")
    ap.add_argument("--music", help="Music bed file, or 'none' (default: a music/bed/bgm file in assets/)")
    ap.add_argument("--music-volume", type=float, default=0.18, help="Music level before ducking (0-1)")
    ap.add_argument("--loudness", type=float, default=-14.0, help="Target loudness in LUFS")
    ap.add_argument("--no-loudnorm", action="store_true")
    ap.add_argument("--allow-stale", action="store_true")
    args = ap.parse_args()
    need_ffmpeg()

    folder = args.clips or newest_clip_folder()
    stem = folder.name
    man = load_json(folder / "manifest.json")
    if not man or not man.get("order"):
        sys.exit(f"{folder / 'manifest.json'} is missing. Run src/cut_shots.py first.")
    if man.get("flow") != "reference":
        sys.exit("These clips were made by the older cut_clips.py. Use render_video.py for them, or run "
                 "src/cut_shots.py to make reference-style shots.")
    if not man.get("complete", False):
        sys.exit("cut_shots.py did not finish. Run  python3 src/cut_shots.py  again (finished shots are reused).")
    order = man["order"]
    files = [folder / n for n in order]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        sys.exit(f"Missing shot files: {missing}. Run src/cut_shots.py again.")
    shots = [n for n in order if n.startswith("shot_")]
    if [int(n[5:8]) for n in shots] != list(range(1, len(shots) + 1)) or order[-1] != "end_card.mp4":
        sys.exit(f"Shot order is wrong: {order[:3]} ... {order[-2:]}. Run src/cut_shots.py again.")

    voice = d("narration") / stem / "narration.wav"
    timeline = d("narration") / stem / "timeline.json"
    if not voice.exists():
        sys.exit(f"Missing {voice}. Run src/voiceover.py first.")
    mt = (folder / "manifest.json").stat().st_mtime
    newer = [p.name for p in (timeline, d("analysis") / f"{stem}_analysis.json") if p.exists() and p.stat().st_mtime > mt + 1]
    if newer and not args.allow_stale:
        sys.exit(f"{newer} changed after the shots were cut. Run  python3 src/cut_shots.py  first "
                 "(unchanged shots are skipped), or pass --allow-stale.")

    out_dir = d("output", create=True)
    out = out_dir / f"{stem}_recap.mp4"
    lst = folder / "concat_list.txt"
    lst.write_text("".join(f"file '{f.resolve().as_posix()}'\n" for f in files), encoding="utf-8")
    video_only = folder / "video_only.mp4"
    print(f"Joining {len(files)} shots ...")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video_only], "joining shots")
    total = probe_duration(video_only)

    music = find_music(args.music)
    print(f"Mixing audio ({'voice + music: ' + music.name if music else 'voice only; no music file found in assets/'}) ...")
    graph = audio_graph(total, music, args.music_volume, args.loudness, not args.no_loudnorm)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_only, "-i", voice]
    if music:
        cmd += ["-i", music]
    cmd += ["-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-ar", "44100", "-t", f"{total:.3f}", "-movflags", "+faststart", out]
    run(cmd, "rendering")

    dur = probe_duration(out)
    print(f"\nDone. {out}\nLength: {to_hms(dur)} ({dur:.1f}s)")
    if abs(dur - man["expected_total_sec"]) > 0.3:
        print(f"WARNING: expected {man['expected_total_sec']:.1f}s. Run src/verify_video.py.")
    for w in man.get("warnings", []):
        print("  Note from cut_shots:", w)
    print("Next: python3 src/verify_video.py")


if __name__ == "__main__":
    main()
