#!/usr/bin/env python3
"""
cut_clips.py - cut the movie footage for each narration segment and build the poster end card.

Pipeline position:   input/ + analysis/ + narration/  ->  [cut_clips.py]  ->  clips/<stem>/

How it keeps picture and voice in sync
  * The real narration length (from narration/<stem>/seg_NN.wav) decides how long each clip is.
  * A clip longer than its narration is trimmed. A shorter one is extended into the free footage before the NEXT
    clip starts (never into the cliffhanger, never into footage already used), then backwards, never into the
    previous clip. A tiny remaining gap (<=8%) is covered by a barely visible slow-down; anything bigger is a
    short freeze and is reported as a WARNING so you can widen that clip's window.
  * Leading/trailing pure-black frames are trimmed off.
  * Narration starts 0.4 s into each clip, so the crossfade added by render_video.py never eats the first word.
  * The movie's own audio is lowered, and ducked further (sidechain compression) whenever the narrator speaks.
  * Aspect ratio is preserved (scaled to fit, padded if needed). No cropping, no stretching.

Outputs:
    clips/<stem>/clip_01.mp4 ... clip_NN.mp4, cta.mp4 (poster end card), manifest.json (order + fingerprints)
Clips whose inputs did not change are skipped; clips whose inputs changed are rebuilt automatically.

Needs ffmpeg + ffprobe and Pillow (pip install pillow).

Usage (from the project root):
    python3 src/cut_clips.py
    python3 src/cut_clips.py --poster assets/tangled_poster.jpg --resolution 1280x720
    python3 src/cut_clips.py --force
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from common import (assets_images, LEAD_SEC, PIPELINE_VERSION, TAIL_SEC, XFADE_SEC, CLIFF_BUFFER_SEC, d, file_fp, find_movie,
                    find_poster, fingerprint, load_brief, load_json, need_ffmpeg, newest_analysis, pick_font,
                    probe_duration, run, stem_of, to_hms, to_sec, validate_analysis)

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat
except ImportError:
    sys.exit("Missing dependency. Run:  pip install pillow")

GOLD = (255, 214, 120)


# --------------------------------------------------------------------------- #
# Footage planning
# --------------------------------------------------------------------------- #
def has_audio(movie) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", str(movie)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def trim_black(movie, a, b, need):
    """Cut pure-black frames off the start/end of [a, b] when enough footage remains."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(movie),
                        "-vf", "blackdetect=d=0.15:pix_th=0.06", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    spans = [(float(s), float(e)) for s, e in re.findall(r"black_start:([\d.]+)\s+black_end:([\d.]+)", r.stderr)]
    length, a2, b2, note = b - a, a, b, ""
    for s, e in spans:
        if s <= 0.2:
            a2 = a + e
        elif e >= length - 0.2:
            b2 = a + s
    if (a2, b2) != (a, b) and b2 - a2 >= 1.0:      # keep at least 1 s; any shortfall is refilled by plan_clip()
        return a2, b2, f"trimmed black ({a2 - a:.1f}s start, {b - b2:.1f}s end)"
    return a, b, note


def plan_clip(movie, clip, nar_dur, lo, hi, trim):
    """Decide exactly which source seconds to use. lo = end of the previous clip, hi = latest allowed source time."""
    need = nar_dur + LEAD_SEC + TAIL_SEC
    a, b = float(clip["clip_start_sec"]), float(clip["clip_end_sec"])
    notes = []
    if trim:
        a, b, n = trim_black(movie, a, b, need)
        if n:
            notes.append(n)
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
        if short > 0.02:
            if short / max(b - a, 0.1) <= 0.08:
                slow = need / (b - a)
                notes.append(f"slowed {100 * (slow - 1):.1f}%")
            else:
                freeze = short
                warn = (f"segment {clip['segment']}: only {b - a:.1f}s of unused footage for {need:.1f}s of narration; "
                        f"froze the last frame for {short:.1f}s. Widen this clip's window or shorten its narration.")
    return {"a": a, "length": b - a, "slow": slow, "freeze": freeze, "need": need, "notes": notes, "warn": warn}


# --------------------------------------------------------------------------- #
# ffmpeg command for one narrated clip
# --------------------------------------------------------------------------- #
def build_cut(movie, nar_wav, p, w, h, bg_vol, out, audio_ok):
    total = p["need"]
    vf = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    if p["slow"] > 1.0:
        vf += f",setpts={p['slow']:.5f}*PTS"
    vf += ",fps=30"
    if p["freeze"] > 0.02:
        vf += f",tpad=stop_mode=clone:stop_duration={p['freeze']:.3f}"
    vf += ",format=yuv420p[v]"
    if audio_ok:
        tempo = f",atempo={1 / p['slow']:.5f}" if p["slow"] > 1.0 else ""
        bg = f"[0:a]aresample=44100,aformat=channel_layouts=stereo{tempo},volume={bg_vol},apad[bg];"
    else:
        bg = "anullsrc=r=44100:cl=stereo[bg];"
    lead = int(round(LEAD_SEC * 1000))
    af = (bg +
          f"[1:a]adelay={lead}|{lead},aresample=44100,aformat=channel_layouts=stereo,apad,asplit=2[n1][n2];"
          "[bg][n2]sidechaincompress=threshold=0.015:ratio=10:attack=15:release=400:makeup=1[duck];"
          f"[duck][n1]amix=inputs=2:duration=first:normalize=0,afade=t=in:d=0.02,"
          f"afade=t=out:st={total - 0.02:.3f}:d=0.02,alimiter=limit=0.92[a]")
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{p['a']:.3f}", "-t", f"{p['length']:.3f}", "-i", str(movie),
            "-i", str(nar_wav), "-filter_complex", f"{vf};{af}", "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart", str(out)]


# --------------------------------------------------------------------------- #
# Poster-style end card
# --------------------------------------------------------------------------- #
def best_frame(movie, times, out: Path):
    """Fallback when no poster exists: pick the most colourful/bright frame from the footage already in the video."""
    best, tmp = None, out.with_name("_cand.jpg")
    for t in times:
        tmp.unlink(missing_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(movie), "-frames:v", "1",
                        "-vf", "scale=960:-2", str(tmp)], capture_output=True)
        if tmp.exists():
            _, s, v = ImageStat.Stat(Image.open(tmp).convert("HSV")).mean
            score = (s / 255) * ((v / 255) ** 0.5)
            if best is None or score > best[0]:
                best = (score, t)
    tmp.unlink(missing_ok=True)
    if best is None:
        sys.exit("Could not extract a frame for the end card.")
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{best[1]:.2f}", "-i", movie, "-frames:v", "1", "-q:v", "2", out],
        "end-card frame")
    return best[1]


def _font(path, size):
    try:
        return ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
    except (OSError, TypeError):
        return ImageFont.load_default()


def _fit_font(draw, text, path, size, max_w, stroke):
    f = _font(path, size)
    while size > 24:
        box = draw.textbbox((0, 0), text, font=f, stroke_width=stroke)
        if box[2] - box[0] <= max_w:
            break
        size -= 6
        f = _font(path, size)
        stroke = max(2, size // 28)
    return f, size, stroke


def make_endcard(title, art_path, w, h, font_path, out: Path, cta_text, draw_title=True):
    art = Image.open(art_path).convert("RGB")
    if not draw_title:
        # The artwork already has the movie title on it (a real poster/thumbnail): show it whole, uncropped, on a
        # blurred copy of itself, and add only the call-to-action underneath.
        card = ImageEnhance.Brightness(ImageOps.fit(art, (w, h), Image.LANCZOS)
                                       .filter(ImageFilter.GaussianBlur(32))).enhance(0.40)
        k = min(w * 0.94 / art.width, h * 0.84 / art.height)
        aw, ah = int(art.width * k), int(art.height * k)
        band = int(h * 0.14)                       # room for the call-to-action under the art
        ax, ay = (w - aw) // 2, max(int(h * 0.03), (h - ah - band) // 2)
        shadow = Image.new("RGBA", (aw + 80, ah + 80), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rectangle((40, 40, aw + 40, ah + 40), fill=(0, 0, 0, 200))
        blurred = shadow.filter(ImageFilter.GaussianBlur(24))
        card.paste(blurred, (ax - 40, ay - 40), blurred)
        card.paste(art.resize((aw, ah), Image.LANCZOS), (ax, ay))
        if cta_text:
            dr = ImageDraw.Draw(card)
            cf, _, _ = _fit_font(dr, cta_text, font_path, h // 20, int(w * 0.9), 2)
            cb = dr.textbbox((0, 0), cta_text, font=cf, stroke_width=2)
            band_mid = ay + ah + band // 2
            dr.text(((w - (cb[2] - cb[0])) // 2 - cb[0], band_mid - (cb[3] - cb[1]) // 2 - cb[1]), cta_text,
                    font=cf, fill=(255, 255, 255), stroke_width=2, stroke_fill=(20, 10, 30))
        card.convert("RGB").save(out)
        return
    landscape = art.width / art.height >= 1.3
    dr_probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    if landscape:
        card = ImageOps.fit(art, (w, h), Image.LANCZOS)
        gh = int(h * 0.50)
        grad = Image.linear_gradient("L").resize((w, gh)).point(lambda p: int(p * 0.88))
        card.paste(Image.new("RGB", (w, gh), (8, 4, 20)), (0, h - gh), grad)
        dr = ImageDraw.Draw(card)
        tsize = h // 7
        tf, tsize, sw = _fit_font(dr, title.upper(), font_path, tsize, int(w * 0.8), max(3, tsize // 28))
        tb = dr.textbbox((0, 0), title.upper(), font=tf, stroke_width=sw)
        ty = int(h * 0.64)
        dr.text(((w - (tb[2] - tb[0])) // 2 - tb[0], ty - tb[1]), title.upper(), font=tf, fill=GOLD,
                stroke_width=sw, stroke_fill=(50, 25, 10))
        cf = _font(font_path, h // 24)
        cb = dr.textbbox((0, 0), cta_text, font=cf, stroke_width=2)
        dr.text(((w - (cb[2] - cb[0])) // 2 - cb[0], ty + (tb[3] - tb[1]) + int(h * 0.05) - cb[1]), cta_text,
                font=cf, fill=(255, 255, 255), stroke_width=2, stroke_fill=(20, 10, 30))
    else:
        bg = ImageOps.fit(art, (w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(32))
        card = ImageEnhance.Brightness(bg).enhance(0.40)
        ph = int(h * 0.82)
        pw = int(art.width * ph / art.height)
        poster = art.resize((pw, ph), Image.LANCZOS)
        px, py = int(w * 0.10), (h - ph) // 2
        shadow = Image.new("RGBA", (pw + 80, ph + 80), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rectangle((40, 40, pw + 40, ph + 40), fill=(0, 0, 0, 200))
        card.paste(shadow.filter(ImageFilter.GaussianBlur(24)), (px - 40, py - 40), shadow.filter(ImageFilter.GaussianBlur(24)))
        card.paste(poster, (px, py))
        dr = ImageDraw.Draw(card)
        x0, avail = px + pw + int(w * 0.06), w - (px + pw + int(w * 0.06)) - int(w * 0.06)
        tsize = h // 6
        tf, tsize, sw = _fit_font(dr, title.upper(), font_path, tsize, avail, max(3, tsize // 28))
        tb = dr.textbbox((0, 0), title.upper(), font=tf, stroke_width=sw)
        ty = h // 2 - (tb[3] - tb[1]) - int(h * 0.02)
        dr.text((x0 - tb[0], ty - tb[1]), title.upper(), font=tf, fill=GOLD, stroke_width=sw, stroke_fill=(50, 25, 10))
        cf, _, _ = _fit_font(dr, cta_text, font_path, h // 22, avail, 2)
        cb = dr.textbbox((0, 0), cta_text, font=cf, stroke_width=2)
        dr.text((x0 - cb[0], h // 2 + int(h * 0.04) - cb[1]), cta_text, font=cf, fill=(255, 255, 255),
                stroke_width=2, stroke_fill=(20, 10, 30))
    card.convert("RGB").save(out)


def build_card_clip(card_png, nar_wav, total, w, h, out):
    frames = int(round(total * 30))
    lead = int(round(LEAD_SEC * 1000))
    fc = (f"[0:v]scale={2 * w}:{2 * h},zoompan=z='1+0.05*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
          f"d={frames}:s={w}x{h}:fps=30,format=yuv420p[v];"
          f"[1:a]adelay={lead}|{lead},aresample=44100,aformat=channel_layouts=stereo,apad,"
          f"afade=t=in:d=0.02,afade=t=out:st={total - 0.6:.3f}:d=0.6[a]")
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", str(card_png), "-i", str(nar_wav), "-filter_complex", fc,
            "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart", str(out)]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Cut narrated clips and the poster end card.")
    ap.add_argument("--analysis", type=Path, help="Analysis JSON (default: newest in analysis/)")
    ap.add_argument("--movie", type=Path, help="Movie file (default: matching file in input/)")
    ap.add_argument("--resolution", default="1920x1080")
    ap.add_argument("--bg-volume", type=float, default=0.30, help="Movie audio level before ducking (0-1)")
    ap.add_argument("--poster", type=Path, help="Poster/key-art image for the end card (default: search assets/)")
    ap.add_argument("--poster-frame", help="No poster? Use the movie frame at this HH:MM:SS for the end card")
    ap.add_argument("--cta-text", default="Watch the full movie", help="Line shown under the title on the end card")
    ap.add_argument("--font", help="Font file (.ttf) for the end card")
    ap.add_argument("--add-title", action="store_true", help="Draw the title on the end card even for a poster that already has it")
    ap.add_argument("--no-trim-black", action="store_true")
    ap.add_argument("--allow-issues", action="store_true", help="Render even if the analysis JSON has problems")
    ap.add_argument("--force", action="store_true", help="Rebuild every clip")
    args = ap.parse_args()
    need_ffmpeg()
    try:
        w, h = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        sys.exit("--resolution must look like 1920x1080")

    apath = args.analysis or newest_analysis()
    data = json.loads(apath.read_text(encoding="utf-8"))
    stem = stem_of(apath)
    brief = load_brief(None, stem)
    movie = find_movie(stem, args.movie)
    movie_len = probe_duration(movie)

    problems = validate_analysis(data, movie_len)
    if problems:
        print("Problems in the analysis JSON:\n  - " + "\n  - ".join(problems))
        if not args.allow_issues:
            sys.exit("Fix them (edit the clip times in analysis/<stem>_analysis.json, or re-run analyze_movie.py), "
                     "or pass --allow-issues.")
    limit = min(to_sec(data["cliffhanger"]["timestamp"]) - CLIFF_BUFFER_SEC, movie_len - 0.3)
    nar_dir, out_dir = d("narration") / stem, d("clips", create=True) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(out_dir / "manifest.json", {}) or {}
    old_clips = manifest.get("clips", {})
    audio_ok = has_audio(movie)
    print(f"Movie: {movie.name} ({to_hms(movie_len)}) | analysis: {apath.name} | spoiler limit: {to_hms(limit)} | "
          f"{w}x{h} | pipeline v{PIPELINE_VERSION}")

    # Resolve the end-card artwork FIRST, so a wrong path fails in a second instead of after all the cutting.
    poster = find_poster(args.poster, brief)
    if args.poster and not poster:
        sys.exit(f"The poster/thumbnail file was not found: {args.poster}\n"
                 f"  Save the image inside {d('assets', create=True)} (for example assets/tangled_thumbnail.png) and run again.\n"
                 "  With no --poster option, any image in assets/ whose name contains poster, cover or thumb is found "
                 "automatically.\n  Nothing has been cut yet.")
    draw_title = args.add_title or not (poster and brief.get("poster_has_title", True))
    if poster:
        print(f"End card artwork: {poster}" + (" (title will be drawn on it)" if draw_title else ""))
    else:
        found = [x.name for x in assets_images()]
        print(f"No poster/thumbnail found in {d('assets')}. Images there: {found or 'none'}.\n"
              "  Save the thumbnail as assets/tangled_thumbnail.png (any name containing poster, cover or thumb works), "
              "or pass --poster <path>.\n  Continuing with a frame from the movie as the end card.")

    clips, new_manifest, warnings = data["clips"], {}, []
    movie_fp, prev_end, order, durations = file_fp(movie), 0.0, [], {}

    def save_manifest(complete):
        man = {"version": PIPELINE_VERSION, "title": data["title"], "order": list(order), "durations": dict(durations),
               "xfade_sec": XFADE_SEC, "clips": dict(new_manifest), "warnings": list(warnings), "complete": complete,
               "expected_total_sec": round(sum(durations.values()) - XFADE_SEC * max(0, len(order) - 1), 2)}
        (out_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
        return man
    for i, clip in enumerate(clips):
        n = clip["segment"]
        name = f"clip_{n:02d}.mp4"
        nar = nar_dir / f"seg_{n:02d}.wav"
        if not nar.exists():
            sys.exit(f"Missing {nar}. Run src/narrate.py first.")
        hi = min(limit, float(clips[i + 1]["clip_start_sec"]) - 0.1 if i + 1 < len(clips) else limit)
        p = plan_clip(movie, clip, probe_duration(nar), prev_end, hi, not args.no_trim_black)
        prev_end = p["a"] + p["length"]
        fp = fingerprint(PIPELINE_VERSION, movie_fp, round(p["a"], 3), round(p["length"], 3), round(p["slow"], 4),
                         round(p["freeze"], 3), file_fp(nar), w, h, args.bg_volume, audio_ok)
        out = out_dir / name
        status = "kept"
        if args.force or not out.exists() or old_clips.get(name, {}).get("fp") != fp:
            run(build_cut(movie, nar, p, w, h, args.bg_volume, out, audio_ok), f"Cutting {name}")
            status = "built"
        order.append(name)
        durations[name] = round(p["need"], 3)
        new_manifest[name] = {"fp": fp, "source_start": round(p["a"], 2), "source_end": round(prev_end, 2),
                              "duration": round(p["need"], 2), "notes": p["notes"]}
        if p["warn"]:
            warnings.append(p["warn"])
        save_manifest(False)
        print(f"  {name} [{status:5s}] src {to_hms(p['a'], 1)}-{to_hms(prev_end, 1)}  voice {p['need'] - LEAD_SEC - TAIL_SEC:4.1f}s"
              + (f"  ({'; '.join(p['notes'])})" if p["notes"] else "") + ("  WARNING" if p["warn"] else ""))

    # ---- end card (poster) ------------------------------------------------
    cta_wav = nar_dir / "cta.wav"
    if not cta_wav.exists():
        sys.exit(f"Missing {cta_wav}. Run src/narrate.py first.")
    if poster:
        art, art_note = poster, f"poster: {poster}"
    else:
        art = out_dir / "endcard_frame.jpg"
        if args.poster_frame:
            run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{to_sec(args.poster_frame):.2f}", "-i", movie,
                 "-frames:v", "1", "-q:v", "2", art], "end-card frame")
            art_note = f"movie frame at {args.poster_frame}"
        else:
            mids = [(float(c["clip_start_sec"]) + float(c["clip_end_sec"])) / 2 for c in clips]
            art_note = f"auto-picked movie frame at {to_hms(best_frame(movie, mids, art), 1)}"
        warnings.append("No poster image found, so the end card uses a frame from the movie. Put the official poster "
                        "in assets/ (for example assets/tangled_poster.jpg) or pass --poster to use it.")
    font = pick_font(args.font)
    card_total = LEAD_SEC + probe_duration(cta_wav) + 2.2
    card_fp = fingerprint(PIPELINE_VERSION, file_fp(art), data["title"], args.cta_text, file_fp(cta_wav), w, h, font,
                          draw_title)
    cta_out, card_png = out_dir / "cta.mp4", out_dir / "endcard.png"
    status = "kept"
    if args.force or not cta_out.exists() or old_clips.get("cta.mp4", {}).get("fp") != card_fp:
        make_endcard(data["title"], art, w, h, font, card_png, args.cta_text, draw_title)
        run(build_card_clip(card_png, cta_wav, card_total, w, h, cta_out), "Building end card")
        status = "built"
    order.append("cta.mp4")
    durations["cta.mp4"] = round(card_total, 3)
    new_manifest["cta.mp4"] = {"fp": card_fp, "duration": round(card_total, 2), "art": str(art)}
    print(f"  cta.mp4     [{status:5s}] end card ({art_note})")

    manifest = save_manifest(True)
    print(f"\nDone. {len(order)} clips in {out_dir}. Expected final length about "
          f"{manifest['expected_total_sec'] / 60:.1f} min (with crossfades).")
    for wmsg in warnings:
        print("  WARNING:", wmsg)


if __name__ == "__main__":
    main()
