#!/usr/bin/env python3
"""
common.py - helpers shared by every pipeline script.

* Project root is found relative to this file (src/ lives directly under the project root).
* Folder names are matched case-insensitively (input/ == Input/), and missing folders are created
  in lowercase, so you never end up with two folder trees and it works on case-sensitive disks.
* Everything that decides "is this generated file still valid?" (fingerprints) lives here.
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Bump this when rendering logic changes: every clip is then rebuilt on the next run.
PIPELINE_VERSION = "3.1"

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CLIFF_BUFFER_SEC = 3.0    # never show footage closer than this to the cliffhanger moment
WORDS_PER_SEC = 2.5       # ~150 wpm, used only for estimates before real audio exists
XFADE_SEC = 0.25          # (v2 only) crossfade between clips
FPS = 30                  # every shot is cut to an exact number of frames at this rate
LEAD_SEC = 0.40           # narration starts this long after a clip starts (lets the crossfade finish)
TAIL_SEC = 0.40           # breathing room after each narration line

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",     # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",   # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/georgiab.ttf",                           # Windows
    "C:/Windows/Fonts/arialbd.ttf",
]


# ------------------------------------------------------------------ folders
def d(name: str, create: bool = False) -> Path:
    """Return ROOT/<name>, matching an existing folder ignoring case; create it (lowercase) if asked."""
    want = name.lower()
    if ROOT.exists():
        for p in ROOT.iterdir():
            if p.is_dir() and p.name.lower() == want:
                return p
    path = ROOT / want
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ time
def to_sec(ts) -> float:
    """'HH:MM:SS', 'MM:SS', 'HH:MM:SS.s' (or a plain number) -> seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = str(ts).strip().replace(",", ".").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp: {ts!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def to_hms(sec: float, decimals: int = 0) -> str:
    scale = 10 ** decimals
    total = int(round(max(0.0, float(sec)) * scale))
    whole, frac = divmod(total, scale)
    s = f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"
    return f"{s}.{frac:0{decimals}d}" if decimals else s


# ------------------------------------------------------------------ ffmpeg
def need_ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit("ffmpeg/ffprobe not found. Install ffmpeg (Mac: brew install ffmpeg) and reopen the terminal.")


def run(cmd, what: str = "command"):
    cmd = [str(c) for c in cmd]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{what} failed:\n{' '.join(cmd)}\n\n{r.stderr[-2000:]}")
    return r


def probe_duration(path) -> float:
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path], f"ffprobe {Path(path).name}")
    return float(r.stdout.strip())


def has_filter(name: str) -> bool:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return re.search(rf"\s{re.escape(name)}\s", r.stdout) is not None


# ------------------------------------------------------------------ fingerprints
def fingerprint(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_fp(path) -> str:
    st = Path(path).stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ------------------------------------------------------------------ project files
def list_movies():
    folder = d("input")
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXTS) if folder.exists() else []


def find_movie(stem: str = None, explicit=None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"Movie file not found: {p}")
        return p
    vids = list_movies()
    if not vids:
        sys.exit(f"No video file found in {d('input')}. Put the movie there or pass --movie.")
    if stem:
        for v in vids:
            if v.stem.lower() == stem.lower():
                return v
    if len(vids) > 1:
        print(f"Multiple videos in input/; using {vids[0].name}. Pass --movie to choose another.")
    return vids[0]


def newest_analysis() -> Path:
    folder = d("analysis")
    files = sorted(folder.glob("*_analysis.json"), key=lambda p: p.stat().st_mtime) if folder.exists() else []
    if not files:
        sys.exit(f"No *_analysis.json in {folder}. Run src/analyze_movie.py first.")
    return files[-1]


def stem_of(analysis_path) -> str:
    return Path(analysis_path).name.replace("_analysis.json", "")


def load_brief(path=None, stem: str = None) -> dict:
    """Optional per-movie brief (assets/<stem>_brief.json): title, opening shot query, story beats, poster name."""
    cands = [Path(path)] if path else []
    if stem:
        cands.append(d("assets") / f"{stem}_brief.json")
    cands.append(d("assets") / "brief.json")
    for c in cands:
        if c.is_file():
            data = load_json(c)
            if isinstance(data, dict):
                return data
            sys.exit(f"Brief file {c} is not valid JSON.")
    if path:
        sys.exit(f"Brief file not found: {path}")
    return {}


def find_poster(explicit=None, brief: dict = None):
    """Return an existing poster image path, or None. Never returns a path that does not exist."""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    bases = [d("assets"), ROOT, d("input")]
    name = (brief or {}).get("poster")
    if name:
        for b in bases:
            if (b / name).is_file():
                return b / name
    for b in (d("assets"), d("input")):
        if b.exists():
            for p in sorted(b.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS and re.search(r"poster|cover|thumb|key.?art", p.stem, re.I):
                    return p
    only = assets_images()
    return only[0] if len(only) == 1 else None      # a single image in assets/ is taken to be the poster


def assets_images():
    """Images in assets/ (ignoring files this pipeline generates)."""
    base = d("assets")
    if not base.exists():
        return []
    return [p for p in sorted(base.iterdir())
            if p.suffix.lower() in IMAGE_EXTS and not p.stem.startswith(("endcard", "_"))]


def pick_font(user_font=None):
    if user_font:
        return user_font if Path(user_font).is_file() else None
    return next((f for f in FONT_CANDIDATES if Path(f).is_file()), None)


# ------------------------------------------------------------------ analysis validation
def validate_analysis(data: dict, movie_len: float = None, tol: float = 0.25):
    """Return a list of human-readable problems with an analysis JSON's clip list (empty list = valid)."""
    try:
        cliff = to_sec(data["cliffhanger"]["timestamp"])
    except (KeyError, TypeError, ValueError):
        return ["analysis JSON has no valid cliffhanger.timestamp"]
    limit = cliff - CLIFF_BUFFER_SEC
    clips, problems = data.get("clips"), []
    clone = bool(data.get("reference_clone"))      # exact copy of a reference edit: shots may be short, silent, or not in story order
    if not clips:
        return ["analysis JSON has no clips"]
    if not str(data.get("cta", "")).strip():
        problems.append("missing 'cta' text")
    if not str(data.get("title", "")).strip():
        problems.append("missing 'title'")
    prev_end = None
    for i, c in enumerate(clips, 1):
        missing = [k for k in ("segment", "narration", "clip_start_sec", "clip_end_sec") if k not in c]
        if missing:
            problems.append(f"clip {i}: missing {missing}")
            continue
        if c["segment"] != i:
            problems.append(f"clip {i}: segment number is {c['segment']} (must be {i}, in order)")
        if not str(c["narration"]).strip() and not clone:
            problems.append(f"clip {i}: empty narration")
        a, b = float(c["clip_start_sec"]), float(c["clip_end_sec"])
        if b <= a:
            problems.append(f"clip {i}: clip_end must be after clip_start")
        if a < 0:
            problems.append(f"clip {i}: negative start")
        if b > limit + 0.01:
            problems.append(f"clip {i}: ends at {to_hms(b)}, after the spoiler limit {to_hms(limit)}")
        if movie_len and b > movie_len + 0.5:
            problems.append(f"clip {i}: ends after the movie does ({to_hms(movie_len)})")
        if prev_end is not None and a < prev_end - tol and not clone:
            problems.append(f"clip {i}: starts at {to_hms(a, 1)}, before clip {i - 1} ends ({to_hms(prev_end, 1)}); "
                            "out of order or repeated footage")
        prev_end = b
    return problems
