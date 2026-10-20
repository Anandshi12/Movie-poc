#!/usr/bin/env python3
"""
transcribe.py  -  Movie file -> audio -> timestamped transcript.

Pipeline position:   input/  ->  [transcribe.py]  ->  audio/ + transcript/

Outputs (stem = movie file name without extension):
    audio/<stem>.wav            16 kHz mono audio extracted with ffmpeg
    transcript/<stem>.json      {"segments":[{"start":..,"end":..,"text":..}, ...]}
    transcript/<stem>.srt       standard subtitles (same content)

Setup:
    1. Install ffmpeg and make sure `ffmpeg -version` works in your terminal
         Windows:  winget install ffmpeg      Mac: brew install ffmpeg      Linux: sudo apt install ffmpeg
    2. pip install faster-whisper

Usage (from the project root):
    python3 src/transcribe.py                          # first video found in input/
    python3 src/transcribe.py --input input/tangle.mp4
    python3 src/transcribe.py --model medium --language en
    python3 src/transcribe.py --force                  # redo even if outputs already exist

Model size guide (bigger = more accurate, slower, more memory):
    tiny, base, small (default), medium, large-v3
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import VIDEO_EXTS, d

INPUT_DIR, AUDIO_DIR, TRANSCRIPT_DIR = d("input"), d("audio", create=True), d("transcript", create=True)


def find_movie() -> Path:
    files = sorted(p for p in INPUT_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS) if INPUT_DIR.exists() else []
    if not files:
        sys.exit(f"No video file found in {INPUT_DIR}. Put your movie there or use --input.")
    if len(files) > 1:
        print(f"Multiple videos found; using {files[0].name}. Use --input to choose another.")
    return files[0]


def extract_audio(movie: Path, wav: Path) -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found. Install it (see the top of this file) and reopen your terminal.")
    wav.parent.mkdir(exist_ok=True)
    print(f"Extracting audio -> {wav.name} ...")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(movie),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed:\n{result.stderr}")


def srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def pick_device() -> tuple[str, str]:
    """Use the GPU if CUDA is available, otherwise CPU with int8 (fast and memory-light)."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:  # noqa: BLE001
        pass
    return "cpu", "int8"


def transcribe(wav: Path, model_name: str, language: str | None):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Missing dependency. Run:  pip install faster-whisper")

    device, compute = pick_device()
    print(f"Loading Whisper model '{model_name}' on {device} ({compute}). First run downloads the model...")
    model = WhisperModel(model_name, device=device, compute_type=compute)

    segments_iter, info = model.transcribe(
        str(wav),
        language=language,          # None = auto-detect
        beam_size=5,
        vad_filter=True,            # skips long silences / music, speeds things up and reduces hallucinations
        condition_on_previous_text=False,
    )
    print(f"Detected language: {info.language} | duration: {info.duration / 60:.1f} min")

    segments, last_pct = [], -1
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        pct = int(seg.end / info.duration * 100) if info.duration else 0
        if pct != last_pct and pct % 5 == 0:
            print(f"  {pct}% transcribed")
            last_pct = pct
    return segments


def write_outputs(segments, stem: str) -> tuple[Path, Path]:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    json_path = TRANSCRIPT_DIR / f"{stem}.json"
    srt_path = TRANSCRIPT_DIR / f"{stem}.srt"

    json_path.write_text(json.dumps({"segments": segments}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{srt_time(s['start'])} --> {srt_time(s['end'])}", s["text"], ""]
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, srt_path


def main():
    ap = argparse.ArgumentParser(description="Extract audio from a movie and create a timestamped transcript.")
    ap.add_argument("--input", type=Path, help="Movie file (default: first video in input/)")
    ap.add_argument("--model", default="small", help="tiny | base | small | medium | large-v3 (default: small)")
    ap.add_argument("--language", default=None, help="Language code like en, hi, ta (default: auto-detect)")
    ap.add_argument("--force", action="store_true", help="Redo audio extraction and transcription")
    args = ap.parse_args()

    movie = args.input or find_movie()
    if not movie.exists():
        sys.exit(f"File not found: {movie}")
    stem = movie.stem

    json_path, srt_path = TRANSCRIPT_DIR / f"{stem}.json", TRANSCRIPT_DIR / f"{stem}.srt"
    if json_path.exists() and srt_path.exists() and not args.force:
        print(f"Transcript already exists: {srt_path.name}. Use --force to redo it.")
        return

    wav = AUDIO_DIR / f"{stem}.wav"
    if args.force or not wav.exists():
        extract_audio(movie, wav)
    else:
        print(f"Using existing audio: {wav.name}")

    segments = transcribe(wav, args.model, args.language)
    if not segments:
        sys.exit("No speech was detected. Check that the movie has audible dialogue.")

    json_path, srt_path = write_outputs(segments, stem)
    print(f"\nDone. {len(segments)} segments.")
    print(f"  {json_path}\n  {srt_path}")


if __name__ == "__main__":
    main()
