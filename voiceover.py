#!/usr/bin/env python3
"""
voiceover.py - ONE continuous narration take, plus the exact time every segment is spoken.

Why one take: the reference video has a single flowing voice with no gaps between shots. Voicing each shot
separately sounds choppy and robotic; one take keeps natural intonation across sentences. The cut points are then
placed on the narration itself (hard cuts exactly where each segment's words begin).

Pipeline position:   analysis/<stem>_analysis.json  ->  [voiceover.py]  ->  narration/<stem>/

Outputs
    narration/<stem>/narration.wav    final voice: trimmed, leveled (-16 LUFS), fitted to the target length
    narration/<stem>/timeline.json    for every segment: when it is spoken and where its shot starts and ends
    narration/<stem>/manifest.json    fingerprints (unchanged text/voice/settings are not re-voiced)

Voice engines
    edge    (default) Microsoft Edge voices via the free edge-tts package (pip install edge-tts). The SAME voice as
            your previous video is reused automatically (read from narration/<stem>/manifest.json), with the same
            speed and pitch settings. No API key needed.
    gemini  Optional, more natural: Gemini text-to-speech (gemini-3.8-flash-tts). Needs GEMINI_API_KEY and
            pip install -U "google-genai>=2.25.0". Check Google's current pricing/quota.
    auto    gemini when a Gemini key is set and the SDK is new enough, otherwise edge.

Timing (alignment) uses faster-whisper on the finished narration (pip install faster-whisper; the model you
already use in transcribe.py is reused). Without it, timing is estimated from word counts (less exact; a warning
is shown).

Usage (from the project root):
    python3 src/voiceover.py                       # your previous (edge) voice, ~1:56 total video length
    python3 src/voiceover.py --voice en-GB-RyanNeural          # a different edge voice
    python3 src/voiceover.py --target 116          # total video length in seconds (default 116 = 1:56)
    python3 src/voiceover.py --engine gemini --voice Charon    # optional: the more natural Gemini voice
    python3 src/voiceover.py --engine gemini --audition        # listen to several Gemini voices first
"""

import argparse
import base64
import difflib
import io
import json
import os
import re
import sys
import time
import wave
from pathlib import Path

from common import d, fingerprint, load_json, need_ffmpeg, newest_analysis, probe_duration, run, stem_of

SR = 24000                      # Gemini TTS output rate (mono, 16-bit)
LEAD_IN = 0.25                  # seconds of silence before the first word, so the first shot is not abrupt
HOLD = 2.2                      # seconds the end card stays after the last word
CHUNK_CHARS = 1100              # one TTS request per ~180 words, split at segment boundaries
GEMINI_VOICES = ["Charon", "Orus", "Puck", "Fenrir", "Sulafat", "Achird", "Kore", "Algieba"]
MAX_TEMPO = 0.12                # never speed up / slow down the voice by more than 12% to hit the target length


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def prep_text(text: str, engine: str) -> str:
    t = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    t = re.sub(r"\s*[—–]\s*|\s+-\s+", ", ", t)
    if engine == "edge":
        t = re.sub(r"\.{2,}|…", ". ", t)
    t = re.sub(r"[\"“”]", "", t)
    return t if t and t[-1] in ".!?…" else t + "."


def chunk_segments(texts, max_chars=CHUNK_CHARS):
    """Group consecutive segment texts into chunks (never splitting a segment)."""
    chunks, cur = [], []
    for t in texts:
        if cur and len(" ".join(cur + [t])) > max_chars:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(t)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #
def gemini_ready() -> bool:
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return False
    try:
        from google import genai
        return hasattr(genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")), "interactions")
    except Exception:  # noqa: BLE001
        return False


def tts_gemini(text, voice, style, model, retries=5) -> bytes:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if not hasattr(client, "interactions"):
        sys.exit('Your google-genai package is too old for Gemini 3.8 text-to-speech.\n'
                 '  Run:  pip install -U "google-genai>=2.25.0"')
    block = {"type": "text", "text": text}
    if style:
        block["annotations"] = [{"type": "speech_metadata", "style": style}]
    for attempt in range(retries):
        try:
            interaction = client.interactions.create(
                model=model, input=[{"type": "user_input", "content": [block]}],
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": voice}]})
            return base64.b64decode(interaction.output_audio.data)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = any(k in msg for k in ("429", "500", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL"))
            if not transient or attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"  Gemini temporary error ({msg[:70]}...) - retrying in {wait}s")
            time.sleep(wait)


def tts_edge(text, voice, rate, pitch, tmp_mp3: Path) -> bytes:
    import asyncio
    try:
        import edge_tts
    except ImportError:
        sys.exit("Missing dependency. Run:  pip install edge-tts")

    async def go():
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        except TypeError:
            comm = edge_tts.Communicate(text, voice, rate=rate)
        await comm.save(str(tmp_mp3))

    asyncio.run(go())
    wav = tmp_mp3.with_suffix(".wav")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_mp3, "-ac", "1", "-ar", SR, "-c:a", "pcm_s16le", wav], "edge audio convert")
    return wav.read_bytes()


def wav_to_pcm(data: bytes) -> bytes:
    """Return raw 16-bit mono 24 kHz samples from the WAV bytes an engine returned (or raw PCM as a fallback)."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            frames = w.readframes(w.getnframes())
            if w.getframerate() != SR or w.getnchannels() != 1:
                raise ValueError(f"unexpected audio format {w.getframerate()} Hz / {w.getnchannels()} ch")
            return frames
    except wave.Error:
        return data


def write_wav(path: Path, pcm: bytes):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


# --------------------------------------------------------------------------- #
# audio processing
# --------------------------------------------------------------------------- #
def trim_and_level(src: Path, dst: Path):
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af",
         f"{trim},areverse,{trim},areverse,loudnorm=I=-16:TP=-1.5:LRA=9,aresample=44100", "-ac", "1", dst],
        "voice cleanup")


def fit_length(src: Path, dst: Path, target_voice_sec: float, lead_in: float = LEAD_IN):
    """Gently speed up/slow down (pitch preserved, at most 12%) so the video reaches the target length;
    then add the lead-in silence."""
    dur = probe_duration(src)
    ratio = dur / target_voice_sec if target_voice_sec > 0 else 1.0
    note = "no tempo change"
    af = []
    if abs(ratio - 1) > 0.01:
        used = max(1 - MAX_TEMPO, min(1 + MAX_TEMPO, ratio))
        af.append(f"atempo={used:.4f}")
        note = f"tempo x{used:.3f}" + ("" if used == ratio else f" (limited from x{ratio:.2f}; the script is too "
                                                              f"{'long' if ratio > 1 else 'short'} for the target)")
    ms = int(lead_in * 1000)
    af.append(f"adelay={ms}|{ms}")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", ",".join(af), "-ac", "1", dst], "voice fitting")
    return note


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def whisper_words(path: Path, model_name: str):
    from faster_whisper import WhisperModel
    device, compute = "cpu", "int8"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            device, compute = "cuda", "float16"
    except Exception:  # noqa: BLE001
        pass
    model = WhisperModel(model_name, device=device, compute_type=compute)
    segments, _ = model.transcribe(str(path), language="en", word_timestamps=True, beam_size=5,
                                   condition_on_previous_text=False)
    return [(w.word, float(w.start), float(w.end)) for seg in segments for w in (seg.words or [])]


def align_segments(asr_words, seg_texts, total):
    """Map each script segment to (start, end) seconds using the recognized words.
    Returns (spans, match_ratio)."""
    script, owner = [], []
    for i, t in enumerate(seg_texts):
        for w in t.split():
            n = norm(w)
            if n:
                script.append(n)
                owner.append(i)
    asr = [(norm(w), s, e) for w, s, e in asr_words if norm(w)]
    sm = difflib.SequenceMatcher(None, script, [a[0] for a in asr], autojunk=False)
    times = [None] * len(script)
    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = (asr[j1 + k][1], asr[j1 + k][2])
                matched += 1
    # fill words that were not recognized by interpolating between recognized neighbours
    known = [i for i, t in enumerate(times) if t]
    if not known:
        return None, 0.0
    first, last = known[0], known[-1]
    for i in range(first):
        times[i] = (0.0, times[first][0])
    for i in range(last + 1, len(times)):
        times[i] = (times[last][1], total)
    for a, b in zip(known, known[1:]):
        gap = b - a - 1
        if gap:
            t0, t1 = times[a][1], times[b][0]
            for k in range(gap):
                s = t0 + (t1 - t0) * k / gap
                e = t0 + (t1 - t0) * (k + 1) / gap
                times[a + 1 + k] = (s, max(s, e))
    spans = []
    for i in range(len(seg_texts)):
        idx = [k for k, o in enumerate(owner) if o == i]
        spans.append((times[idx[0]][0], times[idx[-1]][1]) if idx else None)
    return spans, matched / max(1, len(script))


def proportional_spans(seg_texts, total):
    """Fallback without speech recognition: spread the voice time over segments by word count."""
    weights = [max(1, len(t.split())) + 1.5 for t in seg_texts]
    t0, span = LEAD_IN, max(1.0, total - LEAD_IN - 0.3)
    spans, acc = [], t0
    for w in weights:
        dur = span * w / sum(weights)
        spans.append((acc, acc + dur))
        acc += dur
    return spans


def build_timeline(spans, texts, ids, total):
    """Cut points: the first shot starts at 0; every later shot starts midway between the previous segment's last
    word and this segment's first word; the final shot (end card) holds for HOLD seconds after the voice ends."""
    spans = list(spans)
    for i, sp in enumerate(spans):                       # repair None / out-of-order spans
        if sp is None:
            spans[i] = (spans[i - 1][1], spans[i - 1][1]) if i else (LEAD_IN, LEAD_IN)
    segs = []
    bounds = [0.0] + [(spans[i - 1][1] + spans[i][0]) / 2 for i in range(1, len(spans))] + [total + HOLD]
    for i, (sp, text, sid) in enumerate(zip(spans, texts, ids)):
        cs, ce = bounds[i], max(bounds[i + 1], bounds[i] + 0.2)
        segs.append({"id": sid, "start": round(sp[0], 3), "end": round(sp[1], 3),
                     "cut_start": round(cs, 3), "cut_end": round(ce, 3), "text": text})
    return segs



# --------------------------------------------------------------------------- #
# reference-clone mode: the cut points are the reference's, moved onto our voice word by word
# --------------------------------------------------------------------------- #
def split_sentences(text):
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    out = []
    for p in parts:                       # no punctuation from the recognizer: cut long runs every ~25 words
        w = p.split()
        out += [" ".join(w[i:i + 25]) for i in range(0, len(w), 25)] if len(w) > 40 else [p]
    return out


def word_map(ref_words, our_words):
    """Anchors (reference time -> our time) from words that appear in both transcripts. Returns (anchors, last_our_index)."""
    ref = [(i, norm(w["w"])) for i, w in enumerate(ref_words) if norm(w["w"])]
    our = [(j, norm(w[0])) for j, w in enumerate(our_words) if norm(w[0])]
    sm = difflib.SequenceMatcher(None, [r[1] for r in ref], [o[1] for o in our], autojunk=False)
    anchors, last_j = [], -1
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            rw, ow = ref_words[ref[i1 + k][0]], our_words[our[j1 + k][0]]
            anchors += [(rw["start"], ow[1]), (rw["end"], ow[2])]
            last_j = max(last_j, our[j1 + k][0])
    clean = []
    for r, o in sorted(anchors):          # keep the mapping strictly increasing
        if not clean or (r > clean[-1][0] and o > clean[-1][1]):
            clean.append((r, o))
    return clean, last_j


def make_mapper(anchors):
    xs, ys = [a[0] for a in anchors], [a[1] for a in anchors]

    def f(t):
        if t <= xs[0]:
            return ys[0] + (t - xs[0])
        if t >= xs[-1]:
            return ys[-1] + (t - xs[-1])
        k = max(i for i in range(len(xs)) if xs[i] <= t)
        return ys[k] + (t - xs[k]) * (ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k])
    return f


def run_clone(args, data, apath, stem, out_dir, engine, voice):
    ref = load_json(d("analysis") / f"{stem}_reference.json")
    if not ref:
        sys.exit(f"Missing analysis/{stem}_reference.json. Run  python3 src/clone_reference.py --reference <file>  first.")
    clips, shots, words = data["clips"], ref["shots"], ref["words"]
    if len(clips) != len(shots):
        sys.exit("The analysis and the reference file disagree on the number of shots. Run clone_reference.py again.")
    ref_text = prep_text(" ".join(w["w"] for w in words), engine)
    cta = prep_text(data["cta"], engine)
    sents = split_sentences(ref_text) + [cta]
    chunks = chunk_segments(sents)
    n_ref = len(words)
    n_cta = len(cta.split())
    fp = fingerprint("clone", chunks, engine, voice, args.style, args.model, args.rate, args.pitch,
                     [(s["ref_start"], s["ref_end"]) for s in shots], ref["first_word_start"])
    manifest = load_json(out_dir / "manifest.json", {}) or {}
    wav_out, tl_path = out_dir / "narration.wav", out_dir / "timeline.json"
    if not args.force and manifest.get("fp") == fp and wav_out.exists() and tl_path.exists():
        print("Voiceover is up to date (reference, text, voice and settings unchanged). Use --force to redo it.")
        return
    print(f"Reference clone | engine: {engine} | voice: {voice} | {len(chunks)} request(s) | {n_ref} reference words + {n_cta} call-to-action words")
    pcm, gap = b"", b"\x00\x00" * int(SR * 0.18)
    for i, chunk in enumerate(chunks, 1):
        print(f"  voicing part {i}/{len(chunks)} ({len(chunk.split())} words)")
        audio = tts_gemini(chunk, voice, args.style, args.model) if engine == "gemini" else \
            tts_edge(chunk, voice, args.rate, args.pitch, out_dir / f"_part{i}.mp3")
        pcm += (gap if pcm else b"") + wav_to_pcm(audio)
    raw, level = out_dir / "voice_raw.wav", out_dir / "voice_level.wav"
    write_wav(raw, pcm)
    trim_and_level(raw, level)

    # match the reference's pace: our reading of the reference text should last as long as the reference's narration
    ref_speech = ref["last_word_end"] - ref["first_word_start"]
    ours_est = probe_duration(level) * n_ref / (n_ref + n_cta)
    lead = max(0.0, ref["first_word_start"] - 0.05)            # keep the reference's intro before the first word
    note = fit_length(level, wav_out, probe_duration(level) * ref_speech / ours_est, lead)
    total = probe_duration(wav_out)
    print(f"  voice length {total:.1f}s ({note}); the reference narration lasts {ref_speech:.1f}s")

    method = "whisper"
    try:
        ours = whisper_words(wav_out, args.align_model)
        anchors, last_j = word_map(words, ours)
        if len(anchors) < 20:
            raise RuntimeError(f"only {len(anchors) // 2} words could be matched")
        cta_start = ours[last_j + 1][1] if last_j + 1 < len(ours) else anchors[-1][1] + 0.3
    except ImportError:
        print("  (faster-whisper is not installed: timing is estimated by stretching. Run: pip install faster-whisper)")
        method = "linear"
        o_end = total - HOLD * 0 - (n_cta / (n_ref + n_cta)) * (total - lead)
        anchors = [(ref["first_word_start"], lead), (ref["last_word_end"], max(lead + 1.0, o_end))]
        cta_start = anchors[-1][1] + 0.3
    except Exception as e:  # noqa: BLE001
        print(f"  (speech alignment failed: {e}; timing is estimated by stretching instead)")
        method = "linear"
        o_end = total - (n_cta / (n_ref + n_cta)) * (total - lead)
        anchors = [(ref["first_word_start"], lead), (ref["last_word_end"], max(lead + 1.0, o_end))]
        cta_start = anchors[-1][1] + 0.3
    f = make_mapper(anchors)

    # every reference cut, moved onto our voice
    cuts = [0.0] + [f(s["ref_start"]) for s in shots[1:]]
    end_footage = min(max(f(ref["duration"]), anchors[-1][1] + 0.25), cta_start - 0.12)
    end_footage = max(end_footage, cuts[-1] + 0.3)
    for i in range(1, len(cuts)):                               # never let a shot shrink below 0.15 s
        cuts[i] = max(cuts[i], cuts[i - 1] + 0.15)
    end_footage = max(end_footage, cuts[-1] + 0.15)
    bounds = cuts + [end_footage]
    segs = []
    for k, (clip, sh) in enumerate(zip(clips, shots)):
        segs.append({"id": clip["segment"], "start": round(f(sh["ref_start"]), 3), "end": round(f(sh["ref_end"]), 3),
                     "cut_start": round(bounds[k], 3), "cut_end": round(bounds[k + 1], 3), "text": clip["narration"]})
    segs.append({"id": "cta", "start": round(cta_start, 3), "end": round(total, 3), "cut_start": round(end_footage, 3),
                 "cut_end": round(max(total + HOLD, end_footage + 2.5), 3), "text": cta})
    tl_path.write_text(json.dumps({
        "stem": stem, "audio": wav_out.name, "audio_duration": round(total, 3), "hold": HOLD, "mode": "reference-clone",
        "video_duration": segs[-1]["cut_end"], "method": method, "anchors": len(anchors) // 2, "segments": segs}, indent=1),
        encoding="utf-8")
    manifest.update(fp=fp, engine=engine, voice=voice, method=method)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shots_len = [s["cut_end"] - s["cut_start"] for s in segs[:-1]]
    ref_len = [s["ref_end"] - s["ref_start"] for s in shots]
    drift = sorted(abs(a - b) for a, b in zip(shots_len, ref_len))
    print(f"\nDone. {len(shots)} shots; timing: {method} ({len(anchors) // 2} words matched). Shot lengths differ from the "
          f"reference by a median of {drift[len(drift) // 2]:.2f}s (worst {drift[-1]:.2f}s).")
    print(f"  Video length will be about {segs[-1]['cut_end']:.1f}s (reference {ref['duration']:.1f}s + call to action).")
    print(f"  {wav_out}\n  {tl_path}")

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Create one continuous voiceover and its timing.")
    ap.add_argument("--analysis", type=Path, help="Analysis JSON (default: newest in analysis/)")
    ap.add_argument("--engine", choices=["auto", "gemini", "edge"], default="edge")
    ap.add_argument("--voice", help="Voice name (edge default: the voice of your previous video; Gemini default: Charon)")
    ap.add_argument("--style", default="", help='Short delivery note for Gemini, e.g. "warm, gripping storyteller" '
                                                '(empty = plain, which Google recommends trying first)')
    ap.add_argument("--model", default=os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts"))
    ap.add_argument("--rate", default="-4%", help="edge-tts speed (same as the previous video)")
    ap.add_argument("--pitch", default="-2Hz", help="edge-tts pitch")
    ap.add_argument("--target", type=float, default=116.0, help="Total video length in seconds (0 = do not fit)")
    ap.add_argument("--align-model", default="small", help="faster-whisper model used to time the words")
    ap.add_argument("--audition", action="store_true", help="Voice your first lines with several voices")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    need_ffmpeg()

    apath = args.analysis or newest_analysis()
    data = json.loads(apath.read_text(encoding="utf-8"))
    stem = stem_of(apath)
    out_dir = d("narration", create=True) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    engine = args.engine if args.engine != "auto" else ("gemini" if gemini_ready() else "edge")
    prev = load_json(out_dir / "manifest.json", {}) or {}
    prev_edge_voice = prev.get("voice") if prev.get("engine", "edge") == "edge" else None
    voice = args.voice or ("Charon" if engine == "gemini" else (prev_edge_voice or "en-US-AndrewNeural"))
    if engine == "edge" and not args.voice:
        print(f"Voice: {voice}" + (" (the same voice as your previous video)" if prev_edge_voice else
                                   " (no previous video found; use --voice to choose another)"))

    if data.get("reference_clone"):
        return run_clone(args, data, apath, stem, out_dir, engine, voice)
    ids = [c["segment"] for c in data["clips"]] + ["cta"]
    texts = [prep_text(c["narration"], engine) for c in data["clips"]] + [prep_text(data["cta"], engine)]
    chunks = chunk_segments(texts)

    if args.audition:
        aud = out_dir.parent / "_auditions"
        aud.mkdir(exist_ok=True)
        sample = " ".join(texts[:3])
        if engine != "gemini":
            sys.exit("Auditioning Gemini voices needs a Gemini key and google-genai >= 2.25.0.")
        for v in GEMINI_VOICES:
            pcm = wav_to_pcm(tts_gemini(sample, v, args.style, args.model))
            write_wav(aud / f"{v}.wav", pcm)
            print(f"  {v:12s} -> {aud / (v + '.wav')}")
        print("Listen, then run:  python3 src/voiceover.py --voice <name> --force")
        return

    fp = fingerprint(chunks, engine, voice, args.style, args.model, args.rate, args.pitch, args.target)
    manifest = load_json(out_dir / "manifest.json", {}) or {}
    wav_out, tl_path = out_dir / "narration.wav", out_dir / "timeline.json"
    if not args.force and manifest.get("fp") == fp and wav_out.exists() and tl_path.exists():
        print("Voiceover is up to date (text, voice and settings unchanged). Use --force to redo it.")
        return

    print(f"Engine: {engine} | voice: {voice} | {len(chunks)} request(s) | {sum(len(t.split()) for t in texts)} words")
    pcm = b""
    gap = b"\x00\x00" * int(SR * 0.18)
    for i, chunk in enumerate(chunks, 1):
        print(f"  voicing part {i}/{len(chunks)} ({len(chunk.split())} words)")
        if engine == "gemini":
            audio = tts_gemini(chunk, voice, args.style, args.model)
        else:
            audio = tts_edge(chunk, voice, args.rate, args.pitch, out_dir / f"_part{i}.mp3")
        pcm += (gap if pcm else b"") + wav_to_pcm(audio)
    raw, level = out_dir / "voice_raw.wav", out_dir / "voice_level.wav"
    write_wav(raw, pcm)
    trim_and_level(raw, level)

    target_voice = (args.target - HOLD - LEAD_IN) if args.target else probe_duration(level)
    note = fit_length(level, wav_out, target_voice)
    total = probe_duration(wav_out)
    print(f"  voice length {total:.1f}s ({note}); video will be about {total + HOLD:.1f}s")

    method, ratio = "whisper", 0.0
    try:
        words = whisper_words(wav_out, args.align_model)
        spans, ratio = align_segments(words, texts, total)
        if spans is None or ratio < 0.6:
            raise RuntimeError(f"only {ratio:.0%} of the words matched")
    except ImportError:
        print("  (faster-whisper is not installed: timing is estimated from word counts. Run: pip install faster-whisper)")
        method, spans = "estimated", proportional_spans(texts, total)
    except Exception as e:  # noqa: BLE001
        print(f"  (speech alignment failed: {e}; timing is estimated from word counts instead)")
        method, spans = "estimated", proportional_spans(texts, total)
    segs = build_timeline(spans, texts, ids, total)
    tl_path.write_text(json.dumps({
        "stem": stem, "audio": wav_out.name, "audio_duration": round(total, 3), "hold": HOLD,
        "video_duration": round(total + HOLD, 3), "method": method, "match_ratio": round(ratio, 3), "segments": segs,
    }, indent=1), encoding="utf-8")
    manifest.update(fp=fp, engine=engine, voice=voice, method=method)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shots = [s["cut_end"] - s["cut_start"] for s in segs]
    print(f"\nDone. {len(segs)} shots, median {sorted(shots)[len(shots) // 2]:.1f}s, timing: {method}"
          + (f" ({ratio:.0%} of words matched)" if method == "whisper" else ""))
    print(f"  {wav_out}\n  {tl_path}")


if __name__ == "__main__":
    main()
