#!/usr/bin/env python3
"""
narrate.py - analysis JSON -> voiceover (one WAV per segment + one for the CTA).

Pipeline position:   analysis/ -> [narrate.py] -> narration/<stem>/

Outputs:
    narration/<stem>/seg_01.wav ... seg_NN.wav   leveled, silence-trimmed narration (what cut_clips.py uses)
    narration/<stem>/cta.wav
    narration/<stem>/raw/*.mp3                   the untouched TTS output
    narration/<stem>/manifest.json               fingerprints: changed text is re-voiced, unchanged text is not

Uses Microsoft Edge neural voices via the free `edge-tts` package (no API key, needs internet).

Setup:      pip install edge-tts
Usage (from the project root):
    python3 src/narrate.py                         # picks the first AVAILABLE voice from a preference list
    python3 src/narrate.py --list-voices           # English neural voices available right now
    python3 src/narrate.py --audition              # voice the opening line with several voices so you can listen
    python3 src/narrate.py --voice en-GB-RyanNeural --rate -2%
    python3 src/narrate.py --force                 # re-voice everything

To change wording: edit "narration"/"cta" in analysis/<stem>_analysis.json and run again.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

try:
    import edge_tts
except ImportError:
    sys.exit("Missing dependency. Run:  pip install edge-tts")

from common import d, fingerprint, load_json, need_ffmpeg, newest_analysis, run, stem_of

# Tried in this order; the first one the service actually offers is used. None of these is assumed to exist.
VOICE_PREFERENCE = ["en-US-AndrewMultilingualNeural", "en-US-BrianMultilingualNeural", "en-US-AndrewNeural",
                    "en-US-ChristopherNeural", "en-GB-RyanNeural", "en-US-GuyNeural"]
FALLBACK_VOICE = "en-US-ChristopherNeural"


def prep_text(text: str) -> str:
    """Make text TTS-friendly: no dashes/ellipses, tidy spacing, always ends with punctuation."""
    t = text.replace("\n", " ")
    t = re.sub(r"\s*[—–]\s*|\s+-\s+", ", ", t)
    t = re.sub(r"\.{2,}|…", ". ", t)
    t = re.sub(r"[\"“”]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t and t[-1] in ".!?" else t + "."


async def available_voices():
    try:
        voices = await edge_tts.list_voices()
    except Exception as e:  # noqa: BLE001
        print(f"  (Could not reach the voice list: {e})")
        return []
    return sorted((v for v in voices if v.get("Locale", "").startswith("en-") and v["ShortName"].endswith("Neural")),
                  key=lambda v: v["ShortName"])


def choose_voice(requested, available_names):
    if requested:
        if available_names and requested not in available_names:
            print(f"  Voice '{requested}' is not available; choosing from the preference list instead.")
        else:
            return requested
    for v in VOICE_PREFERENCE:
        if v in available_names:
            return v
    if available_names:
        print(f"  None of the preferred voices are available; using {FALLBACK_VOICE if FALLBACK_VOICE in available_names else available_names[0]}.")
        return FALLBACK_VOICE if FALLBACK_VOICE in available_names else available_names[0]
    return requested or FALLBACK_VOICE


async def synth(text, voice, rate, pitch, out: Path, retries=4):
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            try:
                comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            except TypeError:                     # older edge-tts without pitch support
                comm = edge_tts.Communicate(text, voice, rate=rate)
            await comm.save(str(out))
            if out.exists() and out.stat().st_size > 0:
                return
            raise RuntimeError("empty audio file")
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"edge-tts failed for {out.name}: {e}") from e
            await asyncio.sleep(2 * (attempt + 1))


def finish_audio(raw_mp3: Path, wav_out: Path) -> None:
    """Trim leading/trailing silence, even out loudness, write a precise-length mono WAV."""
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_mp3, "-af",
         f"{trim},areverse,{trim},areverse,loudnorm=I=-16:TP=-1.5:LRA=9,aresample=44100",
         "-ac", "1", "-c:a", "pcm_s16le", wav_out], f"audio cleanup for {raw_mp3.name}")


async def run_jobs(jobs, voice, rate, pitch, out_dir: Path, manifest: dict):
    for i, (name, text, fp) in enumerate(jobs, 1):
        print(f"  [{i}/{len(jobs)}] {name}")
        raw = out_dir / "raw" / f"{name}.mp3"
        await synth(text, voice, rate, pitch, raw)
        finish_audio(raw, out_dir / f"{name}.wav")
        manifest["files"][name] = fp
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Generate voiceover files from the analysis JSON.")
    ap.add_argument("--analysis", type=Path, help="Analysis JSON (default: newest in analysis/)")
    ap.add_argument("--voice", help="Voice name (default: first available from the preference list)")
    ap.add_argument("--rate", default="-4%", help='Speaking speed, e.g. "-4%%" (slightly slower reads more cinematic)')
    ap.add_argument("--pitch", default="-2Hz", help='Pitch shift, e.g. "-2Hz"')
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--audition", action="store_true", help="Voice the opening line with several voices")
    ap.add_argument("--force", action="store_true", help="Re-voice everything")
    args = ap.parse_args()
    need_ffmpeg()

    voices = asyncio.run(available_voices())
    names = [v["ShortName"] for v in voices]
    if args.list_voices:
        for v in voices:
            print(f"{v['ShortName']:38s} {v.get('Gender', ''):7s} {v['Locale']}")
        if not voices:
            print("No voices returned (offline?).")
        return

    apath = args.analysis or newest_analysis()
    data = json.loads(apath.read_text(encoding="utf-8"))
    stem = stem_of(apath)
    out_dir = d("narration", create=True) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.audition:
        if not names:
            sys.exit("Cannot reach the voice service, so nothing can be auditioned right now.")
        cands = [v for v in VOICE_PREFERENCE if v in names] or names[:4]
        text = prep_text(data["clips"][0]["narration"])
        aud = out_dir.parent / "_auditions"
        for v in cands:
            asyncio.run(synth(text, v, args.rate, args.pitch, aud / f"{v}.mp3"))
            print(f"  {v:38s} -> {aud / (v + '.mp3')}")
        print("Listen to them, then run:  python3 src/narrate.py --voice <name> --force")
        return

    voice = choose_voice(args.voice, names)
    print(f"Analysis: {apath.name} | voice: {voice} | rate {args.rate} | pitch {args.pitch}")

    manifest = load_json(out_dir / "manifest.json", {}) or {}
    manifest.setdefault("files", {})
    jobs = []
    items = [(f"seg_{c['segment']:02d}", c["narration"]) for c in data["clips"]] + [("cta", data["cta"])]
    for name, raw_text in items:
        text = prep_text(raw_text)
        fp = fingerprint(text, voice, args.rate, args.pitch)
        if args.force or manifest["files"].get(name) != fp or not (out_dir / f"{name}.wav").exists():
            jobs.append((name, text, fp))
    manifest["voice"] = voice
    if not jobs:
        print("All narration is up to date (text, voice and settings unchanged). Use --force to re-voice.")
        return
    print(f"Voicing {len(jobs)} of {len(items)} lines...")
    asyncio.run(run_jobs(jobs, voice, args.rate, args.pitch, out_dir, manifest))
    print(f"\nDone. Narration saved in {out_dir}")


if __name__ == "__main__":
    main()
