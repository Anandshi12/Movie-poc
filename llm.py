#!/usr/bin/env python3
"""
llm.py - the one place that talks to a language model. Works with Gemini OR Groq.

Which one is used
    --provider groq|gemini  (analyze_movie.py), or  LLM_PROVIDER=groq|gemini,
    otherwise: Gemini if GEMINI_API_KEY/GOOGLE_API_KEY is set, else Groq if GROQ_API_KEY is set.

Setup
    Gemini:  pip install google-genai pydantic      export GEMINI_API_KEY="..."      (model: GEMINI_MODEL)
    Groq:    pip install groq pydantic              export GROQ_API_KEY="..."        (model: GROQ_MODEL,
                                                    default openai/gpt-oss-120b)

Groq free-plan limits are per MINUTE (tokens per minute, TPM), counted as prompt + requested output:
    openai/gpt-oss-120b / -20b: 8,000      llama-3.3-70b-versatile: 12,000
    meta-llama/llama-4-scout-17b-16e-instruct: 30,000      (from console.groq.com/docs/rate-limits)
Override with LLM_TPM=<number> if your plan is different (Developer plan: much higher). This module:
    * refuses to send a request that can never fit (PromptTooLarge) so callers can split it,
    * spaces requests so the rolling 60-second token budget is never exceeded,
    * retries rate-limit / overload / bad-JSON replies.
Vision features (locate_scene.py, verify_clips.py) are Gemini-only.
"""

import json
import os
import re
import sys
import time

GROQ_TPM = {
    "openai/gpt-oss-120b": 8000, "openai/gpt-oss-20b": 8000, "openai/gpt-oss-safeguard-20b": 8000,
    "llama-3.3-70b-versatile": 12000, "meta-llama/llama-4-scout-17b-16e-instruct": 30000,
    "llama-3.1-8b-instant": 6000,
}
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")      # Gemini default (used by the vision scripts)


class PromptTooLarge(Exception):
    """The request alone exceeds the provider's per-minute token limit; split it into smaller requests."""


def pick_provider(explicit=None) -> str:
    p = (explicit or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if p in ("gemini", "groq"):
        return p
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "groq" if os.getenv("GROQ_API_KEY") else "gemini"


def default_model(provider: str) -> str:
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b") if provider == "groq" else DEFAULT_MODEL


def tpm_limit(provider: str, model: str):
    """Tokens-per-minute budget, or None when the provider is not rate-limited this way."""
    if provider != "groq":
        return None
    env = os.getenv("LLM_TPM")
    return int(env) if env else GROQ_TPM.get(model, 6000)


def est_tokens(text: str) -> int:
    return int(len(text) / 3.0) + 1            # deliberately pessimistic (English is ~4 chars/token)


# ----------------------------------------------------------------------------- Gemini
_G = {}


def _gemini():
    if not _G:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            sys.exit("Missing dependency. Run:  pip install google-genai pydantic")
        _G.update(genai=genai, types=types)
    return _G["genai"], _G["types"]


def make_gemini_client():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY). The vision steps need Gemini.")
    return _gemini()[0].Client(api_key=key)


def image_part(jpeg_bytes: bytes):
    return _gemini()[1].Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")


def call_gemini(client, model, system, contents, schema, temperature=0.7, retries=5):
    """contents: a string, or a list mixing strings and image_part(...) objects."""
    types = _gemini()[1]
    safety = [types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
              for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
    config = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                         response_schema=schema, temperature=temperature, safety_settings=safety)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            if not resp.text:
                reason = getattr(resp, "prompt_feedback", None) or (
                    resp.candidates[0].finish_reason if resp.candidates else "unknown")
                raise RuntimeError(f"Gemini returned no text (blocked or empty). Reason: {reason}")
            return schema.model_validate_json(resp.text)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = any(k in msg for k in ("429", "500", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
                                               "DEADLINE", "INTERNAL"))
            if not transient or attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"  Gemini temporary error ({msg[:80]}...) - retrying in {wait}s")
            time.sleep(wait)


# ----------------------------------------------------------------------------- Groq
_WINDOW = []        # (time, tokens) of recent requests, for the rolling 60-second budget


def _wait_for_budget(tokens: int, tpm: int):
    while True:
        now = time.time()
        _WINDOW[:] = [(t, n) for t, n in _WINDOW if now - t < 60]
        if sum(n for _, n in _WINDOW) + tokens <= tpm * 0.95:
            return
        wait = max(1.0, 60 - (now - _WINDOW[0][0]) + 0.5) if _WINDOW else 1.0
        print(f"  (waiting {wait:.0f}s: Groq allows {tpm} tokens per minute)")
        time.sleep(wait)


def _extract_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = text.find("{"), text.rfind("}")
    return text[a:b + 1] if a != -1 and b > a else text


def _retry_after(msg: str):
    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s|m)\b", msg)
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    return v / 1000 if unit == "ms" else v * 60 if unit == "m" else v


def call_groq(client, model, system, prompt, schema, temperature=0.7, max_out=3000, retries=6):
    tpm = tpm_limit("groq", model)
    sys_msg = (system + "\n\nReply with ONE JSON object only (no markdown, no commentary) that follows this JSON "
               "Schema:\n" + json.dumps(schema.model_json_schema(), separators=(",", ":")))
    base_tokens = est_tokens(sys_msg + prompt)
    if tpm and base_tokens + max_out > tpm:
        raise PromptTooLarge(f"prompt ~{base_tokens} + output {max_out} tokens exceeds the {tpm} tokens-per-minute "
                             f"limit of {model}")
    extra = ""
    for attempt in range(retries):
        out = int(max_out * (1.5 if attempt and extra else 1))
        need = base_tokens + out
        kwargs = dict(model=model, temperature=temperature, max_completion_tokens=out,
                      messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt + extra}],
                      response_format={"type": "json_object"})
        if model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "low"
        try:
            if tpm:
                _wait_for_budget(min(need, int(tpm * 0.95)), tpm)
            resp = client.chat.completions.create(**kwargs)
            used = getattr(getattr(resp, "usage", None), "total_tokens", None) or need
            _WINDOW.append((time.time(), used))
            return schema.model_validate_json(_extract_json(resp.choices[0].message.content or ""))
        except PromptTooLarge:
            raise
        except ValueError as e:                     # bad/empty JSON or schema mismatch
            if attempt == retries - 1:
                raise RuntimeError(f"The model did not return valid JSON for {schema.__name__}: {e}") from e
            extra = "\n\nYour previous reply was not valid JSON for the schema. Reply with the JSON object only."
            print(f"  Reply was not valid JSON; retrying ({attempt + 1}/{retries - 1})")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "413" in msg or "Request too large" in msg:
                raise PromptTooLarge(msg[:200]) from e
            transient = any(k in msg for k in ("429", "500", "502", "503", "504", "rate_limit", "overloaded"))
            if not transient or attempt == retries - 1:
                raise
            wait = (_retry_after(msg) or 5 * 2 ** attempt) + 1
            print(f"  Groq temporary error ({msg[:70]}...) - retrying in {wait:.0f}s")
            time.sleep(wait)


def make_client(provider=None):
    provider = pick_provider(provider)
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            sys.exit("Set GROQ_API_KEY.")
        try:
            from groq import Groq
        except ImportError:
            sys.exit("Missing dependency. Run:  pip install groq pydantic")
        return Groq(api_key=key)
    return make_gemini_client()


def call_llm(provider, client, model, system, prompt, schema, temperature=0.7, max_out=3000):
    """Text-only structured call on whichever provider is active (max_out only matters for Groq)."""
    if provider == "groq":
        return call_groq(client, model, system, prompt, schema, temperature, max_out)
    return call_gemini(client, model, system, prompt, schema, temperature)
