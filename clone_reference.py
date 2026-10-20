#!/usr/bin/env python3
"""
clone_reference.py - rebuild a reference video's edit from your movie file, shot for shot.

What it does (no AI guessing; everything is measured):
  1. Finds every CUT in the reference video.
  2. Transcribes the reference's narration with timestamps (faster-whisper), so we know which words are spoken at
     every cut.
  3. Finds where EACH reference shot comes from in the source movie by matching pictures. The reference is a cropped,
     rescaled version of the film, so matching tries several crop sizes and positions. Several frames per shot vote on
     the position, and the start is then refined to the exact frame.
  4. Writes the result as the analysis the rest of the pipeline already understands, in exact-clip mode: every shot
     keeps its exact source window, the story ends exactly where the reference ends.

Outputs (stem = the movie's file name without extension):
    analysis/<stem>_analysis.json         clips = the reference's shots (flagged reference_clone, exact windows)
    analysis/<stem>_reference.json        cuts, words, matches, confidence per shot
    analysis/<stem>_reference_text.txt    the narration found in the reference (read it; fix it in the JSON if needed)
    analysis/<stem>_matches.jpg           reference frame (top) next to the matched movie frame (bottom), every shot
    analysis/<stem>_source_thumbs.npz     cached small frames of your movie (so later runs are fast)

Setup:  pip install numpy pillow faster-whisper        (numpy comes with faster-whisper)
Usage (from the project root; keep the reference video OUTSIDE input/, for example in reference/):
    python3 src/clone_reference.py --reference reference/tangled_reference.mp4
    python3 src/clone_reference.py --reference reference/ref.mp4 --whisper-model medium --cut-threshold 0.2
Then:  python3 src/voiceover.py ; python3 src/cut_shots.py ; python3 src/render_final.py
"""

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("Missing dependency. Run:  pip install numpy pillow")

from common import (CLIFF_BUFFER_SEC, d, file_fp, find_movie, fingerprint, load_brief, load_json, need_ffmpeg,
                    probe_duration, run, to_hms)

SRC_FPS = 4                  # frames per second sampled from the movie for the coarse search
TH_H = 72                    # thumbnail height (width follows the movie's aspect ratio)
FP_W, FP_H = 16, 11          # picture fingerprint grid (covers the top 70% of a window; captions live below)
SCALES = (1.0, 0.9, 0.8, 0.7, 0.6)  # crop sizes tried (1.0 = full height; smaller = the reference zoomed in)
FINE_H = 144                 # picture height used for the exact-frame refinement
FINE_GRID = (40, 28)         # finer fingerprint grid used there
RELIABLE = 0.80              # matching score above which a shot's match is trusted without a second look
NAMES_PROMPT = ("Tangled. Rapunzel, Mother Gothel, Flynn Rider, Eugene, Pascal, Maximus, the king, the queen, "
                "the magic golden flower, the tower.")


# --------------------------------------------------------------------------- #
# picture fingerprints
# --------------------------------------------------------------------------- #
def _edges(lo, hi, n):
    e = np.linspace(lo, hi, n + 1).round().astype(int)
    for i in range(1, len(e)):
        e[i] = max(e[i], e[i - 1] + 1)
    return e


def grid_fp(frames, y0, y1, x0, x1, gw=FP_W, gh=FP_H):
    """frames (N,H,W) uint8 -> (N, gh*gw) zero-mean unit-length vectors of the region [y0:y1, x0:x1]."""
    ye, xe = _edges(y0, y1, gh), _edges(x0, x1, gw)
    sub = frames[:, ye[0]:ye[-1], xe[0]:xe[-1]].astype(np.float32)
    rows = np.add.reduceat(sub, ye[:-1] - ye[0], axis=1)
    cols = np.add.reduceat(rows, xe[:-1] - xe[0], axis=2)
    v = (cols / np.outer(np.diff(ye), np.diff(xe)).astype(np.float32)).reshape(len(frames), -1)
    v -= v.mean(axis=1, keepdims=True)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def make_combos(th_w):
    """Candidate square crop windows (scale, x, y, side) in thumbnail pixels."""
    out = []
    for s in SCALES:
        side = int(round(TH_H * s))
        xs = np.linspace(0, th_w - side, 7).round().astype(int)
        ys = [0] if side >= TH_H else [0, (TH_H - side) // 2, TH_H - side]
        out += [(s, int(x), int(y), side) for y in ys for x in xs]
    return out


def rows_top(side):
    return max(FP_H, int(round(side * 0.70)))


# --------------------------------------------------------------------------- #
# reference: cuts, words, sample frames
# --------------------------------------------------------------------------- #
def detect_cuts(ref, threshold, min_len, fps=30.0):
    dur = probe_duration(ref)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(ref), "-vf",
                        f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    times = sorted(round(float(m) * fps) / fps for m in re.findall(r"pts_time:([\d.]+)", r.stderr))
    cuts = [0.0]
    for t in times:
        if t - cuts[-1] >= min_len and dur - t >= min_len:
            cuts.append(t)
    return cuts, dur


def transcribe_reference(ref, model_name, prompt):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Missing dependency. Run:  pip install faster-whisper")
    wav = Path(tempfile.mkdtemp(prefix="ref_")) / "ref.wav"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", ref, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
        "reference audio")
    device, compute = "cpu", "int8"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            device, compute = "cuda", "float16"
    except Exception:  # noqa: BLE001
        pass
    model = WhisperModel(model_name, device=device, compute_type=compute)
    segments, _ = model.transcribe(str(wav), language="en", word_timestamps=True, beam_size=5,
                                   initial_prompt=prompt, condition_on_previous_text=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            if getattr(w, "probability", 1.0) < 0.15 or not w.word.strip():
                continue
            words.append({"w": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3)})
    return words


def ref_frame(ref, t, size=160):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(ref), "-frames:v", "1",
                        "-vf", f"scale={size}:{size},format=gray", "-f", "rawvideo", "-"], capture_output=True)
    if len(r.stdout) < size * size:
        return None
    return np.frombuffer(r.stdout[:size * size], np.uint8).reshape(size, size)


def sample_times(dur):
    if dur < 0.9:
        return [dur * 0.5]
    if dur < 1.6:
        return [0.15, dur - 0.15]
    return [0.12, dur * 0.33, dur * 0.66, dur - 0.12]


# --------------------------------------------------------------------------- #
# source movie: small frames
# --------------------------------------------------------------------------- #
def source_aspect(movie):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of",
             "csv=p=0:s=x", movie], "ffprobe size")
    w, h = (int(x) for x in r.stdout.strip().split("x"))
    return w / h


def source_thumbs(movie, cache: Path):
    ar = source_aspect(movie)
    th_w = int(round(TH_H * ar / 2) * 2)
    fp = fingerprint(file_fp(movie), SRC_FPS, th_w, TH_H)
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z["fp"]) == fp:
                print(f"  Using cached movie thumbnails ({len(z['frames'])} frames).")
                return z["frames"], th_w
        except Exception:  # noqa: BLE001
            pass
    dur = probe_duration(movie)
    print(f"  Sampling {SRC_FPS} frames/second from the movie ({dur / 60:.0f} min); this takes a few minutes once ...")
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(movie), "-vf", f"fps={SRC_FPS},scale={th_w}:{TH_H}:flags=area,format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fsize, chunks, got, shown = th_w * TH_H, [], 0, -1
    total = int(dur * SRC_FPS)
    while True:
        buf = proc.stdout.read(fsize * 512)
        if not buf:
            break
        chunks.append(buf)
        got += len(buf) // fsize
        pct = int(100 * got / max(1, total))
        if pct // 10 != shown:
            shown = pct // 10
            print(f"    {min(pct, 100)}%")
    proc.wait()
    raw = b"".join(chunks)
    n = len(raw) // fsize
    frames = np.frombuffer(raw[:n * fsize], np.uint8).reshape(n, TH_H, th_w)
    np.savez(cache, frames=frames, fp=np.array(fp))
    return frames, th_w


def src_frames_at(movie, start, length, th_w, th_h, fps=30):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{length:.3f}", "-i", str(movie),
                        "-vf", f"fps={fps},scale={th_w}:{th_h}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    fsize = th_w * th_h
    n = len(r.stdout) // fsize
    return np.frombuffer(r.stdout[:n * fsize], np.uint8).reshape(n, th_h, th_w)


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def score_all(frames, combos, ref_vecs):
    """Best picture-similarity of every reference sample against every movie frame, over all crop windows."""
    n, s = len(frames), len(ref_vecs)
    best = np.full((s, n), -1.0, np.float32)
    arg = np.zeros((s, n), np.uint8)
    R = ref_vecs.astype(np.float32)
    for ci, (_, x, y, side) in enumerate(combos):
        rt = rows_top(side)
        for a in range(0, n, 4096):
            v = grid_fp(frames[a:a + 4096], y, y + rt, x, x + side)
            sc = R @ v.T
            view_b, view_a = best[:, a:a + 4096], arg[:, a:a + 4096]
            m = sc > view_b
            view_b[m] = sc[m]
            view_a[m] = ci
    return best, arg


def top_candidates(scores, k=8, lo_i=0, hi_i=None, min_gap=4):
    hi_i = len(scores) if hi_i is None else hi_i
    seg = scores[lo_i:hi_i]
    if len(seg) == 0:
        return []
    order = np.argsort(-seg)[: k * 12]
    picked = []
    for j in order:
        if all(abs(j - p) >= min_gap for p in picked):
            picked.append(int(j))
        if len(picked) == k:
            break
    return [(p + lo_i, float(seg[p])) for p in picked]


def vote(cands_per_sample, args_per_sample, u_list, width=0.35):
    """Each (sample, candidate frame) proposes a shot start; the start most samples agree on wins."""
    votes = []
    for i, cands in enumerate(cands_per_sample):
        for j, sc in cands:
            votes.append((j / SRC_FPS - u_list[i], sc, int(args_per_sample[i][j]), i))
    if not votes:
        return None
    best = None
    for c, _, _, _ in votes:
        members = {}
        for t, sc, combo, i in votes:
            if abs(t - c) <= width and (i not in members or sc > members[i][1]):
                members[i] = (t, sc, combo)
        total = sum(m[1] for m in members.values())
        if best is None or total > best[0]:
            best = (total, members)
    total, members = best
    wsum = sum(m[1] for m in members.values())
    est = sum(m[0] * m[1] for m in members.values()) / max(wsum, 1e-6)
    top = max(members.values(), key=lambda m: m[1])
    conf = total / len(u_list)
    return {"start": est, "combo": top[2], "confidence": float(conf), "agree": len(members)}


def refine_start(movie, est, combo, th_w, ref_imgs, us, span=0.8):
    """Frame-exact start. Uses a larger, finer picture of the movie around the coarse estimate, first adjusts the crop
    window slightly (position and size), then finds the start time where ALL the reference frames (taken at the
    offsets `us` after the shot start) match best at the same time. ref_imgs: 320x320 gray frames."""
    gw, gh = FINE_GRID
    fine_w = int(round(FINE_H * th_w / TH_H / 2) * 2)
    lo = max(0.0, est - span)
    frames = src_frames_at(movie, lo, 2 * span + max(us) + 0.2, fine_w, FINE_H, 30)
    if len(frames) < 8:
        return est, 0.0
    rv = np.stack([grid_fp(im[None], 0, 224, 0, 320, gw, gh)[0] for im in ref_imgs])        # (J, dims)
    k = FINE_H / TH_H
    _, x0, y0, side0 = combo
    x0, y0, side0 = x0 * k, y0 * k, side0 * k
    base = [int(round((est + u - lo) * 30)) for u in us]

    def geom_score(x, y, side):
        rt = int(round(side * 0.70))
        if x < 0 or y < 0 or x + side > fine_w or y + rt > FINE_H:
            return -9.0
        top = -9.0
        for sh in (-4, -2, 0, 2, 4):
            idx = [b_ + sh for b_ in base]
            if min(idx) >= 0 and max(idx) < len(frames):
                v = grid_fp(frames[idx], y, y + rt, x, x + side, gw, gh)
                top = max(top, float(np.mean(np.sum(rv * v, axis=1))))
        return top

    best_g, best_gs = (int(x0), int(y0), int(side0)), -9.0
    for ds in (0.94, 1.0, 1.06):
        side = int(round(side0 * ds))
        for dy in (-6, -3, 0, 3, 6):
            for dx in range(-10, 11, 2):
                x, y = int(round(x0 + dx - (side - side0) / 2)), int(round(y0 + dy - (side - side0) / 2))
                sc = geom_score(x, y, side)
                if sc > best_gs:
                    best_g, best_gs = (x, y, side), sc
    x, y, side = best_g
    v_all = grid_fp(frames, y, y + int(round(side * 0.70)), x, x + side, gw, gh)
    best, best_s, m = est, -9.0, 0
    while True:
        t = lo + m / 30.0
        if t > est + span:
            break
        idx = [int(round((t + u - lo) * 30)) for u in us]
        if max(idx) < len(frames):
            sc = float(np.mean(np.sum(rv * v_all[idx], axis=1)))
            if sc > best_s:
                best, best_s = t, sc
        m += 1
    return best, best_s


def merge_continuous(shots, tol=0.07, min_conf=0.6):
    """Cuts the reference does not really have (camera moves, flashes) split one continuous piece of film into
    several shots. When two neighbouring shots match consecutive movie time, put them back together."""
    out = [dict(shots[0])]
    for sh in shots[1:]:
        last = out[-1]
        if (last["confidence"] >= min_conf and sh["confidence"] >= min_conf
                and abs(sh["src_start"] - last["src_end"]) <= tol):
            last["ref_end"] = sh["ref_end"]
            last["src_end"] = round(last["src_start"] + (last["ref_end"] - last["ref_start"]), 3)
            last["confidence"] = min(last["confidence"], sh["confidence"])
        else:
            out.append(dict(sh))
    for i, sh in enumerate(out, 1):
        sh["id"] = i
    return out


# --------------------------------------------------------------------------- #
def build_sheet(ref, movie, shots, out_path, th_w, combos, cols=6):
    """Reference frame (top) above the matched movie frame (bottom), for every shot, so the match can be eyeballed."""
    cell = 150
    rows = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cell + 4), rows * (2 * cell + 20)), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, sh in enumerate(shots):
        cx, cy = (i % cols) * (cell + 4), (i // cols) * (2 * cell + 20)
        mid = (sh["ref_start"] + sh["ref_end"]) / 2
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{mid:.3f}", "-i", str(ref), "-frames:v", "1",
                            "-vf", f"scale={cell}:{cell}", str(a)], capture_output=True)
            smid = sh["src_start"] + (mid - sh["ref_start"])
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{smid:.3f}", "-i", str(movie), "-frames:v", "1",
                            "-vf", "scale=640:-2", str(b)], capture_output=True)
            if a.exists():
                sheet.paste(Image.open(a).convert("RGB"), (cx, cy + 16))
            if b.exists():
                im = Image.open(b).convert("RGB")
                s, x, y, side = combos[sh["combo"]]
                k = im.width / th_w
                crop = im.crop((int(x * k), int(y * k), int((x + side) * k), int((y + side) * k)))
                sheet.paste(crop.resize((cell, cell)), (cx, cy + 16 + cell))
        col = (120, 255, 120) if sh["confidence"] >= RELIABLE else (255, 120, 120)
        dr.text((cx + 2, cy + 2), f"#{sh['id']} {sh['confidence']:.2f}", fill=col)
    sheet.save(out_path, quality=80)


def main():
    ap = argparse.ArgumentParser(description="Rebuild a reference video's edit from your movie file.")
    ap.add_argument("--reference", type=Path, required=True, help="The reference video (keep it outside input/)")
    ap.add_argument("--movie", type=Path, help="Your movie file (default: the video in input/)")
    ap.add_argument("--title", help="Movie title for the call to action (default: assets brief, else the file name)")
    ap.add_argument("--cta", help="Call-to-action text")
    ap.add_argument("--cut-threshold", type=float, default=0.10,
                    help="Scene-change sensitivity (lower = more cuts). Extra cuts are harmless: shots that turn out "
                         "to be continuous are merged again after matching.")
    ap.add_argument("--min-shot", type=float, default=0.3, help="Ignore cuts closer together than this (seconds)")
    ap.add_argument("--whisper-model", default="small", help="Speech model for the reference narration (medium = more accurate)")
    ap.add_argument("--prompt", default=NAMES_PROMPT, help="Names/words that help the speech recognizer")
    ap.add_argument("--force", action="store_true", help="Redo the narration transcription")
    args = ap.parse_args()
    need_ffmpeg()
    if not args.reference.is_file():
        sys.exit(f"Reference video not found: {args.reference}")

    movie = find_movie(explicit=args.movie)
    stem = movie.stem
    out_dir = d("analysis", create=True)
    brief = load_brief(None, stem)
    title = args.title or brief.get("title") or stem.replace("_", " ").title()
    cta = args.cta or f"Want to know what happens next? Watch the full movie, {title}."
    print(f"Reference: {args.reference.name} | movie: {movie.name} | title: {title}")

    # 1. cuts
    cuts, rdur = detect_cuts(args.reference, args.cut_threshold, args.min_shot)
    shots = [{"id": i + 1, "ref_start": c, "ref_end": (cuts[i + 1] if i + 1 < len(cuts) else rdur)} for i, c in enumerate(cuts)]
    lens = sorted(s["ref_end"] - s["ref_start"] for s in shots)
    print(f"1/4 Cuts: {len(shots)} shots in {rdur:.1f}s (median {lens[len(lens) // 2]:.1f}s)")

    # 2. narration
    wcache = out_dir / f"{stem}_reference_words.json"
    wfp = fingerprint(file_fp(args.reference), args.whisper_model, args.prompt)
    cached = load_json(wcache, {}) or {}
    if cached.get("fp") == wfp and not args.force:
        words = cached["words"]
        print(f"2/4 Narration: {len(words)} words (cached)")
    else:
        print("2/4 Transcribing the reference narration ...")
        words = transcribe_reference(args.reference, args.whisper_model, args.prompt)
        wcache.write_text(json.dumps({"fp": wfp, "words": words}, indent=1), encoding="utf-8")
        print(f"    {len(words)} words")
    if not words:
        sys.exit("No speech was found in the reference video.")
    text = " ".join(w["w"] for w in words)
    (out_dir / f"{stem}_reference_text.txt").write_text(text + "\n", encoding="utf-8")
    print("    Narration found:\n    " + text[:600] + (" ..." if len(text) > 600 else ""))

    # 3. match every shot to the movie
    print("3/4 Matching shots to the movie ...")
    frames, th_w = source_thumbs(movie, out_dir / f"{stem}_source_thumbs.npz")
    combos = make_combos(th_w)
    samples = []                      # (shot index, local time u, vector)
    for si, sh in enumerate(shots):
        d_ = sh["ref_end"] - sh["ref_start"]
        for u in sample_times(d_):
            fr = ref_frame(args.reference, sh["ref_start"] + u)
            if fr is not None:
                samples.append((si, u, grid_fp(fr[None], 0, 112, 0, 160)[0]))
    ref_vecs = np.stack([s[2] for s in samples])
    best, arg = score_all(frames, combos, ref_vecs)
    by_shot = {}
    for k, (si, u, _) in enumerate(samples):
        by_shot.setdefault(si, []).append(k)

    def match(si, lo=None, hi=None):
        ks = by_shot.get(si, [])
        if not ks:
            return None
        lo_i = 0 if lo is None else max(0, int(lo * SRC_FPS))
        hi_i = None if hi is None else min(len(frames), int(hi * SRC_FPS))
        cands = [top_candidates(best[k], 8, lo_i, hi_i) for k in ks]
        return vote(cands, [arg[k] for k in ks], [samples[k][1] for k in ks])

    def continuation(i, prev_end):
        """Does shot i look like the movie carrying on from where the previous shot ended?"""
        ks = by_shot.get(i, [])
        if not ks or prev_end is None:
            return None
        scores, picks = [], []
        for k in ks:
            c = int(round((prev_end + samples[k][1]) * SRC_FPS))
            lo, hi = max(0, c - 1), min(len(frames), c + 2)
            if hi <= lo:
                return None
            j = lo + int(np.argmax(best[k, lo:hi]))
            scores.append(float(best[k, j]))
            picks.append(int(arg[k, j]))
        mean = sum(scores) / len(scores)
        return {"start": prev_end, "combo": max(set(picks), key=picks.count), "confidence": mean, "agree": len(ks)} \
            if mean >= 0.75 else None

    results = [match(i) for i in range(len(shots))]
    for _ in range(2):                                   # second look at doubtful shots, between their reliable neighbours
        for i, r in enumerate(results):
            if r is not None and r["confidence"] >= RELIABLE:
                continue
            prev = next((results[j] for j in range(i - 1, -1, -1) if results[j] and results[j]["confidence"] >= RELIABLE), None)
            nxt = next((results[j] for j in range(i + 1, len(results)) if results[j] and results[j]["confidence"] >= RELIABLE), None)
            if i > 0 and results[i - 1] is not None and results[i - 1]["confidence"] >= RELIABLE:
                cont = continuation(i, results[i - 1]["start"] + (shots[i - 1]["ref_end"] - shots[i - 1]["ref_start"]))
                if cont and (r is None or cont["confidence"] > r["confidence"]):
                    results[i] = cont
                    continue
            lo = (prev["start"] - 2) if prev else None
            hi = (nxt["start"] + 4) if nxt else None
            if lo is None and hi is None:
                continue
            again = match(i, lo, hi)
            if again and (r is None or again["confidence"] > r["confidence"]):
                results[i] = again

    print("4/4 Refining each start to the exact frame ...")
    movie_len = probe_duration(movie)
    low = []
    for i, (sh, r) in enumerate(zip(shots, results)):
        d_ = sh["ref_end"] - sh["ref_start"]
        if r is None:
            sh.update(src_start=0.0, src_end=d_, confidence=0.0, combo=0)
            low.append(sh)
            continue
        us = sorted({round(max(0.05, min(d_ - 0.05, d_ * f)), 3) for f in (0.08, 0.25, 0.4, 0.55, 0.7, 0.9)})
        imgs = [ref_frame(args.reference, sh["ref_start"] + u, 320) for u in us]
        if any(im is None for im in imgs):
            sh.update(src_start=round(r["start"], 3), src_end=round(r["start"] + d_, 3), confidence=round(r["confidence"], 3),
                      combo=r["combo"], agree=r["agree"])
            continue
        start, fine = refine_start(movie, r["start"], combos[r["combo"]], th_w, imgs, us)
        start = max(0.0, min(start, movie_len - d_ - 0.05))
        sh.update(src_start=round(start, 3), src_end=round(start + d_, 3), confidence=round(r["confidence"], 3),
                  fine=round(fine, 3), combo=r["combo"], agree=r["agree"])
        if r["confidence"] < RELIABLE:
            low.append(sh)

    before = len(shots)
    shots = merge_continuous(shots)
    if len(shots) != before:
        print(f"    {before - len(shots)} extra cut(s) were camera moves inside one continuous piece of film and were merged "
              f"back: {len(shots)} shots remain.")
    low = [s for s in shots if s["confidence"] < RELIABLE]

    # words per shot (a word belongs to the shot in which it starts)
    for sh in shots:
        inside = [w["w"] for w in words if sh["ref_start"] - 1e-6 <= w["start"] < sh["ref_end"] - 1e-6]
        sh["text"] = " ".join(inside)
    first_w, last_w = words[0]["start"], words[-1]["end"]

    # 5. write the analysis the rest of the pipeline reads
    end_src = max(s["src_end"] for s in shots)
    clips = [{"segment": s["id"], "role": "reference", "narration": s["text"], "exact": True,
              "clip_start": to_hms(s["src_start"], 1), "clip_end": to_hms(s["src_end"], 1),
              "clip_start_sec": s["src_start"], "clip_end_sec": s["src_end"],
              "clip_duration_sec": round(s["src_end"] - s["src_start"], 2), "match_confidence": s["confidence"]}
             for s in shots]
    apath = out_dir / f"{stem}_analysis.json"
    if apath.exists() and not (out_dir / f"{stem}_analysis.pre_clone.json").exists():
        (out_dir / f"{stem}_analysis.pre_clone.json").write_text(apath.read_text(encoding="utf-8"), encoding="utf-8")
    analysis = {
        "title": title, "style": "reference-clone", "reference_clone": True, "source_transcript": args.reference.name,
        "movie_duration": to_hms(movie_len), "model": "reference-match",
        "cliffhanger": {"timestamp": to_hms(math.ceil((end_src + CLIFF_BUFFER_SEC) * 10) / 10, 1),
                        "hidden_event": "", "reason": "the recap stops exactly where the reference video stops"},
        "clips": clips, "cta": cta, "word_count": len(words), "validation_issues": [],
    }
    apath.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}_reference.json").write_text(json.dumps({
        "reference": args.reference.name, "duration": round(rdur, 3), "first_word_start": first_w, "last_word_end": last_w,
        "text": text, "words": words, "geometry": [list(c) for c in combos], "shots": shots}, indent=1), encoding="utf-8")
    sheet = out_dir / f"{stem}_matches.jpg"
    build_sheet(args.reference, movie, shots, sheet, th_w, combos)

    conf = [s["confidence"] for s in shots]
    print(f"\nDone. {len(shots)} shots matched. Average match score {sum(conf) / len(conf):.2f}; "
          f"{sum(c >= RELIABLE for c in conf)}/{len(conf)} reliable.")
    print(f"  Last shot ends at {to_hms(end_src, 1)} in the movie (this is where the recap stops).")
    if low:
        print("  CHECK these doubtful shots in the picture sheet (red numbers):",
              ", ".join(f"#{s['id']} ({s['confidence']:.2f})" for s in low[:15]))
    print(f"  Picture sheet: {sheet}\n  Analysis: {apath}")


if __name__ == "__main__":
    main()
