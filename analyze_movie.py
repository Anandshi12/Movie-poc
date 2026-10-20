#!/usr/bin/env python3
"""
analyze_movie.py - transcript -> cliffhanger recap script + clip timestamps.

Pipeline position:  transcript/ (+ analysis/<stem>_anchors.json from locate_scene.py)
                    ->  [analyze_movie.py]  ->  script/ + analysis/

What is different from v1
  * Stage 1 (plot points + cliffhanger) is REUSED from analysis/<stem>_analysis.json when it exists, so a
    re-run only makes ONE API call (the script writer). Use --fresh to redo stage 1.
  * If a visual opening anchor exists (locate_scene.py) or --opening-start is given, segment 1 is LOCKED to start
    at that exact moment and its narration must describe only that shot.
  * Clips must be chronological, must not overlap (no repeated footage), and must be long enough for their
    narration (clip windows are extended into the free gap before the next clip instead of freezing later).
  * Narration is written for a text-to-speech voice: short sentences, natural rhythm.

Usage (from the project root):
    python3 src/analyze_movie.py --title "Tangled"
    python3 src/analyze_movie.py --opening-start 00:01:12.5
    python3 src/analyze_movie.py --fresh            # redo the plot/cliffhanger analysis too
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from common import CLIFF_BUFFER_SEC, d, load_brief, load_json, to_hms, to_sec, validate_analysis
from common import fingerprint
from llm import (PromptTooLarge, call_llm, default_model, est_tokens, make_client, pick_provider,
                 tpm_limit)


# --------------------------------------------------------------------------- #
# Schemas (Gemini returns JSON matching these)
# --------------------------------------------------------------------------- #
class PlotPoint(BaseModel):
    timestamp: str = Field(description="HH:MM:SS when this beat happens")
    summary: str = Field(description="One or two sentences")
    importance: int = Field(description="1 (minor) to 5 (essential)")


class Cliffhanger(BaseModel):
    timestamp: str = Field(description="HH:MM:SS of the last safe moment BEFORE the twist/climax/reveal begins")
    hidden_event: str = Field(description="Internal note: the twist/climax being withheld. Never shown to the audience.")
    reason: str = Field(description="Why stopping here maximizes curiosity")


class Analysis(BaseModel):
    genre: str
    logline: str = Field(description="Spoiler-free one-sentence premise")
    main_characters: list[str]
    plot_points: list[PlotPoint]
    cliffhanger: Cliffhanger


class Segment(BaseModel):
    narration: str = Field(description="Narration for this beat: short sentences, spoken aloud")
    clip_start: str = Field(description="HH:MM:SS start of the matching movie clip")
    clip_end: str = Field(description="HH:MM:SS end of the matching movie clip")


class RecapScript(BaseModel):
    segments: list[Segment]
    cta: str = Field(description="1-3 sentence call to action ending with the movie title")


# --------------------------------------------------------------------------- #
# Transcript loading -> list of (start_sec, end_sec, text)
# --------------------------------------------------------------------------- #
TIME_RANGE = re.compile(r"(\d{1,2}:\d{2}:\d{2}[.,]?\d*|\d{1,2}:\d{2}[.,]\d+)\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]?\d*|\d{1,2}:\d{2}[.,]\d+)")
TXT_LINE = re.compile(r"^\[?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s*(?:-+>?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?))?\s*\]?\s*[:\-]?\s*(.+)$")


def parse_subtitles(raw):
    segs = []
    for block in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        for i, line in enumerate(lines):
            m = TIME_RANGE.search(line)
            if m:
                text = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:])).strip()
                if text:
                    segs.append((to_sec(m.group(1)), to_sec(m.group(2)), text))
                break
    return segs


def parse_json(raw):
    data = json.loads(raw)
    items = data.get("segments", data) if isinstance(data, dict) else data
    segs = []
    for s in items:
        text = str(s.get("text", "")).strip()
        if text:
            segs.append((float(s["start"]), float(s.get("end", s["start"])), text))
    return segs


def parse_txt(raw):
    segs = []
    for line in raw.splitlines():
        m = TXT_LINE.match(line.strip())
        if m:
            start = to_sec(m.group(1))
            segs.append((start, to_sec(m.group(2)) if m.group(2) else start, m.group(3).strip()))
    return segs


def load_transcript(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    ext = path.suffix.lower()
    if ext in (".srt", ".vtt"):
        segs = parse_subtitles(raw)
    elif ext == ".json":
        segs = parse_json(raw)
    else:
        segs = parse_txt(raw) or parse_subtitles(raw)
    if not segs:
        sys.exit(f"Could not find timestamps in {path.name}. Clip selection needs timestamps "
                 "(re-run transcribe.py to get .srt/.json).")
    segs.sort(key=lambda s: s[0])
    return segs


def format_for_model(segs) -> str:
    return "\n".join(f"[{to_hms(s)}] {t}" for s, _, t in segs)


def newest_transcript() -> Path:
    folder = d("transcript")
    files = [p for p in folder.glob("*") if p.suffix.lower() in (".srt", ".vtt", ".json", ".txt")] if folder.exists() else []
    if not files:
        sys.exit(f"No transcript in {folder}. Run src/transcribe.py first or pass --transcript.")
    return max(files, key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
ANALYSIS_SYSTEM = """You are a veteran story analyst and trailer editor. You read full movie transcripts \
and identify the narrative structure with precision. Timestamps in the transcript look like [HH:MM:SS]. \
Only use timestamps that actually exist in the transcript."""

ANALYSIS_PROMPT = """Analyze this movie transcript.

1. List 12-20 key plot points in chronological order with timestamps (setup, inciting incident, rising \
action, reversals, escalating stakes).
2. Choose the CLIFFHANGER point: the last safe moment right BEFORE the movie's biggest twist, reveal, \
betrayal, or climax begins. By this timestamp the audience should understand the characters and stakes, \
but must not yet know the twist. This is typically 60-90% through the film. If the film has several \
twists, pick the most important one.
3. Fill 'hidden_event' with what is being withheld (internal use only).

Movie duration: about {duration}.

TRANSCRIPT:
{transcript}"""

SCRIPT_SYSTEM = """You are an elite YouTube movie-recap narrator. You write for a text-to-speech voice that will \
be heard over movie footage: conversational, suspenseful, cinematic.

Voice and rhythm rules:
- Short sentences (most under 16 words, never over 24). Mix short punchy lines with a slightly longer one.
- Use commas and full stops to create natural pauses. No ellipses, no dashes, no parentheses, no emojis, no markdown.
- Present tense. Concrete images and character names, not vague summary.
- Make the viewer feel danger and emotion rising. Cut exposition that the footage already shows.
- Never repeat the same sentence opening twice in a row. Avoid filler like "in this video", "meanwhile", "little does she know".
- You NEVER reveal or hint at the specifics of the twist."""

SCRIPT_PROMPT = """Write a recap narration for the movie described below and pick matching footage for every segment.

Hard rules:
- Total narration (all segments + cta) must be between {min_words} and {max_words} words (target about {target_words}).
- Cover the story chronologically. Stop at the cliffhanger timestamp {cliff_ts}. Nothing from {cliff_ts} onward \
may be told, quoted, or hinted at. The last segment builds maximum tension and ends on an unanswered question.
- Use {min_segments}-{max_segments} segments. Each segment is ONE visual moment of the story, {seg_words}. \
The editor cuts to a new shot at every segment, so a segment must never describe two different scenes.
- FOOTAGE MUST MATCH THE WORDS. For each segment pick the transcript time range where the thing you narrate is \
actually happening. Never describe an event over footage from a different event. Never use footage from an event \
the narration has not reached yet.
- Clips must be strictly chronological and must NOT overlap or repeat. Each clip's start must be at or after the \
previous clip's end. Every clip ends at or before {latest_end}.
- Each clip window must be at least (words / 2.6 + 1) seconds long, so the picture never has to freeze or stretch. \
Typical clip length: {clip_len}.
- Only use timestamps that exist in the transcript (or the locked opening time below).
- 'cta' is 1-3 short sentences telling viewers to watch the full movie to find out what happens, ENDING with the \
exact title "{title}". Do not explain what was withheld.
- Do not mention "transcript", "the script", or yourself.
{opening_block}{feedback}
MOVIE TITLE: {title}

ANALYSIS (internal, do NOT reveal the hidden event):
{analysis_json}

TRANSCRIPT (only the part before the cliffhanger is provided):
{transcript}"""

OPENING_BLOCK = """
OPENING LOCK (mandatory):
- Segment 1's clip_start MUST be exactly {open_ts}. That is the first shot of the video: {open_desc}. Nothing before it is shown.
- Segment 1's narration describes ONLY what is visible in that shot and opens with a suspenseful line. Suggested \
opening (adapt it so it fits the shot and its duration): "{hint}"
- Segment 1's clip should run until the moment the narration of that beat ends (about {open_len} seconds).
{beats}"""


# --------------------------------------------------------------------------- #
# Fixing and validating the model's output
# --------------------------------------------------------------------------- #
STYLES = {
    # "reference": fast square recap, about 2 minutes, a new shot every ~3 seconds, one continuous voice
    "reference": dict(min_words=260, max_words=340, target=300, min_seg=34, max_seg=52,
                      seg_words="4-10 words (one short phrase or sentence)", clip_len="2-6 seconds", sent_max=16),
    # "long": slower recap, 3-5 minutes, longer scenes
    "long": dict(min_words=450, max_words=750, target=600, min_seg=8, max_seg=14,
                 seg_words="30-70 words", clip_len="12-30 seconds", sent_max=24),
}
SPEECH_WPS = 2.6      # words per second of the narrator (estimate; the real timing comes from the audio later)


def est_sec(text: str) -> float:
    return len(text.split()) / SPEECH_WPS


def need_sec(text: str) -> float:
    return est_sec(text) + 0.3


def normalize_script(script, opening_start, limit):
    """Deterministic repairs: lock the opening start; extend short clip windows into the free gap before the next clip."""
    segs = script.segments
    try:
        if opening_start is not None and segs:
            segs[0].clip_start = to_hms(opening_start, 1)
            if to_sec(segs[0].clip_end) < opening_start + need_sec(segs[0].narration) + 1.0:
                segs[0].clip_end = to_hms(opening_start + need_sec(segs[0].narration) + 1.5, 1)
        for i, s in enumerate(segs):
            a, b = to_sec(s.clip_start), to_sec(s.clip_end)
            need = need_sec(s.narration) + 0.5
            nxt = to_sec(segs[i + 1].clip_start) - 0.1 if i + 1 < len(segs) else limit
            if b > nxt and nxt - a >= need - 0.5:
                b = nxt                                   # overlap with the next clip: end this one where the next begins
            elif b - a < need:
                b = max(b, min(a + need, nxt - 0.1, limit))   # too short: grow into the free gap before the next clip
            s.clip_start, s.clip_end = to_hms(a, 1), to_hms(b, 1)
    except ValueError:
        pass  # unparseable timestamps are reported by validate()


def word_count(script) -> int:
    return sum(len(s.narration.split()) for s in script.segments) + len(script.cta.split())


def validate(script, title, cliff_sec, duration, min_words, max_words, min_seg, opening_start, sent_max=24):
    problems, limit = [], cliff_sec - CLIFF_BUFFER_SEC
    wc = word_count(script)
    if not min_words <= wc <= max_words:
        problems.append(f"Word count is {wc}; it must be between {min_words} and {max_words}.")
    if len(script.segments) < min_seg:
        problems.append(f"Only {len(script.segments)} segments; use at least {min_seg}.")
    if title.lower() not in script.cta.lower():
        problems.append(f'The cta must contain the exact movie title "{title}".')
    prev_end = None
    for i, seg in enumerate(script.segments, 1):
        try:
            a, b = to_sec(seg.clip_start), to_sec(seg.clip_end)
        except ValueError:
            problems.append(f"Segment {i} has an unparseable timestamp.")
            continue
        if b <= a:
            problems.append(f"Segment {i}: clip_end must be after clip_start.")
        if b > limit + 0.01:
            problems.append(f"Segment {i}: clip_end {seg.clip_end} is past the allowed limit {to_hms(limit)}.")
        if b > duration + 1:
            problems.append(f"Segment {i}: clip_end {seg.clip_end} exceeds the movie length.")
        if prev_end is not None and a < prev_end - 0.25:
            problems.append(f"Segment {i}: starts at {seg.clip_start}, before segment {i - 1} ends "
                            f"({to_hms(prev_end, 1)}). Clips must be chronological and must not overlap or repeat.")
        if b - a < need_sec(seg.narration) - 0.5:
            problems.append(f"Segment {i}: clip is {b - a:.0f}s but its narration needs about {need_sec(seg.narration):.0f}s; "
                            "choose a longer stretch of footage for that moment.")
        for sentence in re.split(r"(?<=[.!?])\s+", seg.narration.strip()):
            if len(sentence.split()) > sent_max + 2:
                problems.append(f"Segment {i}: a sentence has {len(sentence.split())} words; split it (max {sent_max}).")
                break
        prev_end = max(prev_end or 0.0, b)
    if opening_start is not None and script.segments:
        try:
            if abs(to_sec(script.segments[0].clip_start) - opening_start) > 0.6:
                problems.append(f"Segment 1 must start at {to_hms(opening_start, 1)}.")
        except ValueError:
            pass
    return problems


def guess_title(stem: str) -> str:
    s = re.sub(r"[\[\(].*?[\]\)]", "", stem)
    s = re.sub(r"[._]+", " ", s).strip()
    return s.title() if s.islower() else s


# --------------------------------------------------------------------------- #
# Small-budget mode (e.g. Groq free plan: 8,000 tokens per MINUTE)
# When the whole transcript does not fit in one request, the work is split:
#   stage 1: summarize the transcript chunk by chunk, then analyze the summaries
#   stage 2: write the script in parts (one part per slice of the story), then join the parts
# --------------------------------------------------------------------------- #
class DigestItem(BaseModel):
    timestamp: str = Field(description="HH:MM:SS of the beat, copied from the transcript")
    summary: str = Field(description="What happens, at most 25 words, concrete")
    importance: int = Field(description="1 (minor) to 5 (essential)")


class DigestChunk(BaseModel):
    items: list[DigestItem]


class RecapPart(BaseModel):
    segments: list[Segment]


DIGEST_SYSTEM = "You are a story analyst. You condense movie transcript excerpts into precise story beats."
DIGEST_PROMPT = """Condense this part of a movie transcript into 4-8 story beats in chronological order.
Each beat: a timestamp copied from the transcript, one concrete sentence (max 25 words) saying what happens
and who is involved, and an importance from 1 to 5. Skip small talk.

TRANSCRIPT PART:
{transcript}"""

PART_PROMPT = """Write PART {k} of {K} of a recap narration for the movie "{title}" and pick matching footage for \
each segment.

This part covers the story between {t0} and {t1} of the transcript below. Write {n_min}-{n_max} segments and \
about {w} words in total for this part. {ending}
Narration written so far (continue naturally, never repeat it): {prior}

Rules:
- Each segment is ONE visual moment of the story, {seg_words}. The editor cuts to a new shot at every segment.
- FOOTAGE MUST MATCH THE WORDS: choose the transcript time range where the thing you narrate is actually happening.
- Clips are strictly chronological and must NOT overlap or repeat. {earliest}Every clip ends at or before {latest_end}.
- Each clip window must be at least (words / 2.6 + 1) seconds long. Typical clip length: {clip_len}.
- Only use timestamps that exist in the transcript below (or the locked opening time).
- Do not describe anything from {cliff_ts} onward.
{opening_block}{feedback}
STORY ANALYSIS FOR THIS PART (internal):
{analysis_json}

TRANSCRIPT (this part only):
{transcript}"""


def chunk_segments(segs, max_tokens):
    """Group transcript lines into chunks whose estimated size stays under max_tokens."""
    chunks, cur, used = [], [], 0
    for seg in segs:
        n = est_tokens(f"[00:00:00] {seg[2]}\n")
        if cur and used + n > max_tokens:
            chunks.append(cur)
            cur, used = [], 0
        cur.append(seg)
        used += n
    if cur:
        chunks.append(cur)
    return chunks


def build_digest(provider, client, model, tpm, segs, digest_path):
    fp = fingerprint(len(segs), round(segs[-1][0]), model, tpm)
    cached = load_json(digest_path, {}) or {}
    if cached.get("fp") == fp:
        print("  Using the saved chunk summaries (no API calls).")
        return [DigestItem(**x) for x in cached["items"]]
    max_out = 1100
    chunks = chunk_segments(segs, int(tpm * 0.9) - max_out - 450)
    items = []
    for i, ch in enumerate(chunks, 1):
        print(f"  Summarizing chunk {i}/{len(chunks)} ({to_hms(ch[0][0])}-{to_hms(ch[-1][1])})")
        res = call_llm(provider, client, model, DIGEST_SYSTEM, DIGEST_PROMPT.format(transcript=format_for_model(ch)),
                       DigestChunk, 0.2, max_out=max_out)
        items += res.items
    digest_path.write_text(json.dumps({"fp": fp, "items": [it.model_dump() for it in items]}, indent=1), encoding="utf-8")
    return items


def run_stage1(provider, client, model, tpm, segs, duration, digest_path):
    prompt = ANALYSIS_PROMPT.format(duration=to_hms(duration), transcript=format_for_model(segs))
    if not tpm or est_tokens(ANALYSIS_SYSTEM + prompt) + 3500 <= tpm * 0.9:
        try:
            return call_llm(provider, client, model, ANALYSIS_SYSTEM, prompt, Analysis, 0.3, max_out=3500)
        except PromptTooLarge:
            pass
    print("  The transcript is too long for one request on this plan; summarizing it in chunks first...")
    digest = build_digest(provider, client, model, tpm, segs, digest_path)
    text = "(Condensed story beats, not full dialogue; use these timestamps.)\n" + "\n".join(
        f"[{it.timestamp}] (importance {it.importance}) {it.summary}" for it in digest)
    return call_llm(provider, client, model, ANALYSIS_SYSTEM,
                    ANALYSIS_PROMPT.format(duration=to_hms(duration), transcript=text), Analysis, 0.3, max_out=3500)


def story_json(analysis, cliff_sec, t0=None, t1=None):
    """The analysis the script writer sees. Never includes the withheld twist (hidden_event)."""
    pts = [p for p in analysis.plot_points if to_sec(p.timestamp) < cliff_sec
           and (t0 is None or t0 - 1 <= to_sec(p.timestamp) <= t1 + 1)]
    return json.dumps({"genre": analysis.genre, "logline": analysis.logline, "main_characters": analysis.main_characters,
                       "plot_points": [{"t": p.timestamp, "what": p.summary} for p in pts]}, separators=(",", ":"))


STATE = {"parts": False}      # set when the script had to be written in several requests


def write_script_parts(provider, client, model, tpm, *, st, title, analysis, cliff_sec, before, limit,
                       opening_block, feedback, cta, beats_block=""):
    max_out = 3000
    STATE["parts"] = True
    cliff_ts = analysis.cliffhanger.timestamp
    fixed = est_tokens(SCRIPT_SYSTEM + PART_PROMPT + opening_block + feedback) + est_tokens(story_json(analysis, cliff_sec)) // 2
    parts = chunk_segments(before, max(800, int(tpm * 0.9) - max_out - fixed - 300))
    total_span = max(1.0, before[-1][1] - before[0][0])
    segments = []
    for k, part in enumerate(parts, 1):
        t0, t1 = part[0][0], part[-1][1]
        share = (t1 - t0) / total_span
        n_min = max(3, round(st["min_seg"] * share))
        n_max = max(n_min + 2, round(st["max_seg"] * share) + 1)
        ending = (f"This is the FINAL part: build maximum tension and end on an unanswered question just before {cliff_ts}. "
                  "Resolve nothing." if k == len(parts) else
                  "The story continues after this part: do not wrap up or hint at an ending.")
        earliest = (f"Every clip must start at or after {segments[-1].clip_end}. " if segments else "")
        prior = " ".join(s.narration for s in segments[-3:]) or "(nothing yet: this is the beginning)"
        print(f"  Writing part {k}/{len(parts)} ({to_hms(t0)}-{to_hms(t1)})")
        res = call_llm(provider, client, model, SCRIPT_SYSTEM, PART_PROMPT.format(
            k=k, K=len(parts), title=title, t0=to_hms(t0), t1=to_hms(t1), n_min=n_min, n_max=n_max,
            w=int(st["target"] * share), ending=ending, prior=prior, seg_words=st["seg_words"], earliest=earliest,
            latest_end=to_hms(limit, 1), clip_len=st["clip_len"], cliff_ts=cliff_ts,
            opening_block=opening_block if k == 1 else beats_block, feedback=feedback,
            analysis_json=story_json(analysis, cliff_sec, t0, t1), transcript=format_for_model(part)),
            RecapPart, 0.8, max_out=max_out)
        segments += res.segments
    return RecapScript(segments=segments, cta=cta)


def write_script(provider, client, model, tpm, *, st, min_words, max_words, title, analysis, cliff_sec, before, limit,
                 opening_block, feedback, cta, beats_block=""):
    prompt = SCRIPT_PROMPT.format(
        min_words=min_words, max_words=max_words, target_words=st["target"], cliff_ts=analysis.cliffhanger.timestamp,
        latest_end=to_hms(limit, 1), min_segments=st["min_seg"], max_segments=st["max_seg"], seg_words=st["seg_words"],
        clip_len=st["clip_len"], title=title, opening_block=opening_block, feedback=feedback,
        analysis_json=story_json(analysis, cliff_sec), transcript=format_for_model(before))
    if not tpm or est_tokens(SCRIPT_SYSTEM + prompt) + 3500 <= tpm * 0.9:
        try:
            return call_llm(provider, client, model, SCRIPT_SYSTEM, prompt, RecapScript, 0.8, max_out=3500)
        except PromptTooLarge:
            pass
    print("  The story is too long for one request on this plan; writing the script in parts...")
    return write_script_parts(provider, client, model, tpm, st=st, title=title, analysis=analysis, cliff_sec=cliff_sec,
                              before=before, limit=limit, opening_block=opening_block, feedback=feedback, cta=cta,
                              beats_block=beats_block)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Generate a cliffhanger recap script + clip timestamps from a transcript.")
    ap.add_argument("--transcript", type=Path, help="Transcript file (default: newest in transcript/)")
    ap.add_argument("--title", help="Movie title for the call to action (default: brief, else file name)")
    ap.add_argument("--brief", type=Path, help="assets/<stem>_brief.json (opening shot, story beats, poster)")
    ap.add_argument("--opening-start", help="Lock segment 1 to this HH:MM:SS[.s] (overrides the stored anchor)")
    ap.add_argument("--provider", choices=["gemini", "groq"], help="LLM provider (default: from your API keys)")
    ap.add_argument("--model", help="Model name (default: GROQ_MODEL / GEMINI_MODEL, else a sensible default)")
    ap.add_argument("--cta", help="Call-to-action text (only used when the script is written in parts)")
    ap.add_argument("--style", choices=list(STYLES), default="reference",
                    help="reference = ~2 min fast square recap (default); long = 3-5 min slower recap")
    ap.add_argument("--min-words", type=int, help="Override the style's minimum word count")
    ap.add_argument("--max-words", type=int, help="Override the style's maximum word count")
    ap.add_argument("--no-opening-lock", action="store_true", help="Ignore any stored opening anchor")
    ap.add_argument("--cliffhanger", help="Stop the story here (HH:MM:SS[.s]); overrides the saved cliffhanger")
    ap.add_argument("--attempts", type=int, default=None,
                    help="Rewrite attempts if validation fails (default 3; 1 when the script is written in parts, "
                         "because every attempt re-sends the whole story)")
    ap.add_argument("--fresh", action="store_true", help="Also redo stage 1 (plot points + cliffhanger)")
    args = ap.parse_args()

    st = STYLES[args.style]
    min_words, max_words = args.min_words or st["min_words"], args.max_words or st["max_words"]
    tpath = args.transcript or newest_transcript()
    stem = tpath.stem
    segs = load_transcript(tpath)
    duration = max(e for _, e, _ in segs)
    brief = load_brief(args.brief, stem)
    title = args.title or brief.get("title") or guess_title(stem)
    if not args.title and not brief.get("title"):
        print(f"  Note: no assets/{stem}_brief.json was found, so the title is guessed from the file name "
              f"('{title}'). Pass --title \"Tangled\" (the real title) or add the brief file.")
    analysis_path = d("analysis", create=True) / f"{stem}_analysis.json"
    script_dir = d("script", create=True)
    provider = pick_provider(args.provider)
    model = args.model or default_model(provider)
    client, tpm = make_client(provider), tpm_limit(provider, model)
    cta = args.cta or f"Want to know what happens next? Watch the full movie, {title}."
    print(f"Transcript: {tpath.name} | {len(segs)} lines | ~{to_hms(duration)} | title: {title} | "
          f"{provider}:{model}" + (f" | {tpm} tokens/min limit" if tpm else ""))

    # ---- Stage 1: plot analysis + cliffhanger (reused when possible) ------------
    old = load_json(analysis_path, {}) or {}
    analysis = None
    if not args.fresh and old.get("cliffhanger") and old.get("plot_points"):
        try:
            analysis = Analysis.model_validate({k: old[k] for k in ("genre", "logline", "main_characters",
                                                                      "plot_points", "cliffhanger")})
            print(f"Stage 1/2: reusing saved analysis (cliffhanger {analysis.cliffhanger.timestamp}). No API call.")
        except Exception:  # noqa: BLE001
            analysis = None
    if analysis is None:
        print("Stage 1/2: analyzing plot and locating the cliffhanger...")
        analysis = run_stage1(provider, client, model, tpm, segs, duration,
                              d("analysis", create=True) / f"{stem}_digest.json")
    anchors_all = load_json(d("analysis") / f"{stem}_anchors.json", {}) or {}
    override = args.cliffhanger or (anchors_all.get("cliffhanger") or {}).get("start_sec")
    if override is not None:
        analysis.cliffhanger.timestamp = to_hms(to_sec(override), 1)
        print(f"  Cliffhanger moved to {analysis.cliffhanger.timestamp} ({'--cliffhanger' if args.cliffhanger else 'stored anchor'})")
    cliff_sec = to_sec(analysis.cliffhanger.timestamp)
    limit = cliff_sec - CLIFF_BUFFER_SEC
    print(f"  Cliffhanger point: {analysis.cliffhanger.timestamp}")

    # ---- Opening anchor (visual start of the video) -------------------------------
    opening_start = None
    if args.no_opening_lock:
        pass
    elif args.opening_start:
        opening_start = to_sec(args.opening_start)
    else:
        anchors = load_json(d("analysis") / f"{stem}_anchors.json", {}) or {}
        if anchors.get("opening"):
            opening_start = float(anchors["opening"]["start_sec"])
    ob = brief.get("opening", {})
    beats = brief.get("beats") or []
    beats_block = ("\n- STORY ORDER: tell these beats in this order, one after another (skip none, add none, reorder none):\n"
                   + "\n".join(f"    {i}. {b}" for i, b in enumerate(beats, 1)) + "\n") if beats else ""
    opening_block = beats_block
    if opening_start is not None:
        opening_block = OPENING_BLOCK.format(
            open_ts=to_hms(opening_start, 1), open_desc=ob.get("query", "the opening shot of the movie"),
            hint=ob.get("narration_hint", "What if one moment could change everything?"),
            open_len=int(need_sec("x " * 40)), beats="") + beats_block
        print(f"  Opening locked at {to_hms(opening_start, 1)}")
    else:
        print("  No opening anchor stored (run src/locate_scene.py to lock a visual opening).")

    # ---- Stage 2: script + clips (only sees transcript BEFORE the cliffhanger) ----
    before = [s for s in segs if s[0] < cliff_sec]
    feedback, script, problems = "", None, []
    max_attempts, attempt = args.attempts or 3, 0
    while attempt < max_attempts:
        attempt += 1
        print(f"Stage 2/2: writing script (attempt {attempt}/{max_attempts})...")
        script = write_script(provider, client, model, tpm, st=st, min_words=min_words, max_words=max_words,
                              title=title, analysis=analysis, cliff_sec=cliff_sec, before=before, limit=limit,
                              opening_block=opening_block, feedback=feedback, cta=cta, beats_block=beats_block)
        if STATE["parts"] and args.attempts is None:
            max_attempts = 1
        normalize_script(script, opening_start, limit)
        problems = validate(script, title, cliff_sec, duration, min_words, max_words, st["min_seg"], opening_start,
                            st["sent_max"])
        if not problems:
            break
        print("  Validation issues:\n   - " + "\n   - ".join(problems))
        feedback = "\nYour previous attempt had these problems. Fix ALL of them:\n- " + "\n- ".join(problems) + "\n"

    if problems:
        print("Warning: script still has issues after all attempts. They are saved in validation_issues; "
              "cut_clips.py will refuse to render until they are fixed.")
        if title.lower() not in script.cta.lower():
            script.cta = script.cta.rstrip() + f" Watch {title} to find out."

    # ---- Save ---------------------------------------------------------------
    full_text = "\n\n".join([s.narration.strip() for s in script.segments] + [script.cta.strip()])
    script_path = script_dir / f"{stem}_script.txt"
    script_path.write_text(full_text + "\n", encoding="utf-8")

    clips = []
    for i, s in enumerate(script.segments, 1):
        try:
            a, b = to_sec(s.clip_start), to_sec(s.clip_end)
        except ValueError:
            a = b = 0.0
        clips.append({
            "segment": i, "role": "opening" if (i == 1 and opening_start is not None) else "story",
            "narration": s.narration.strip(),
            "clip_start": to_hms(a, 1), "clip_end": to_hms(b, 1),
            "clip_start_sec": round(a, 2), "clip_end_sec": round(b, 2),
            "clip_duration_sec": round(b - a, 1), "narration_est_sec": round(est_sec(s.narration), 1),
        })
    out = {
        "title": title, "source_transcript": tpath.name, "movie_duration": to_hms(duration), "model": f"{provider}:{model}",
        "genre": analysis.genre, "logline": analysis.logline, "main_characters": analysis.main_characters,
        "plot_points": [p.model_dump() for p in analysis.plot_points],
        "cliffhanger": analysis.cliffhanger.model_dump(),
        "opening_start_sec": opening_start, "clips": clips, "cta": script.cta.strip(),
        "word_count": word_count(script), "style": args.style,
        "estimated_narration_minutes": round(word_count(script) / SPEECH_WPS / 60, 2),
        "validation_issues": problems,
    }
    out["validation_issues"] = problems + validate_analysis(out, duration)
    analysis_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. {out['word_count']} words (~{out['estimated_narration_minutes']} min), {len(clips)} clips.")
    print(f"  Script:   {script_path}\n  Analysis: {analysis_path}")
    if out["validation_issues"]:
        print("  Issues to fix:\n   - " + "\n   - ".join(out["validation_issues"]))


if __name__ == "__main__":
    main()
