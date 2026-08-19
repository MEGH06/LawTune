"""Script detection and text hygiene.

Used in two places that must agree: the Sangraha sampler (throw away a "Tamil"
document that is actually 90% English boilerplate) and the evaluator (did the model
actually answer in Tamil, or did it answer in English and pretend?).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Dict, Optional

# Unicode blocks that matter for the 22 scheduled languages.
SCRIPT_RANGES = {
    "Latin":       [(0x0041, 0x024F)],
    "Devanagari":  [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali":     [(0x0980, 0x09FF)],
    "Gurmukhi":    [(0x0A00, 0x0A7F)],
    "Gujarati":    [(0x0A80, 0x0AFF)],
    "Oriya":       [(0x0B00, 0x0B7F)],
    "Tamil":       [(0x0B80, 0x0BFF)],
    "Telugu":      [(0x0C00, 0x0C7F)],
    "Kannada":     [(0x0C80, 0x0CFF)],
    "Malayalam":   [(0x0D00, 0x0D7F)],
    "Arabic":      [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Ol_Chiki":    [(0x1C50, 0x1C7F)],
    "Meetei":      [(0xABC0, 0xABFF), (0xAAE0, 0xAAFF)],
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"[ \t ]+")
_NL_RE = re.compile(r"\n{3,}")


def script_histogram(text: str) -> Dict[str, int]:
    """Count characters per script block. Ignores digits, punctuation, whitespace."""
    hist: Dict[str, int] = {}
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("Z", "C", "P", "N") or cat in ("Mn", "Mc"):
            continue
        cp = ord(ch)
        for name, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                hist[name] = hist.get(name, 0) + 1
                break
    return hist


def dominant_script(text: str) -> Optional[str]:
    hist = script_histogram(text)
    return max(hist, key=hist.get) if hist else None


def script_ratio(text: str, script: str) -> float:
    """Fraction of scripted characters that belong to `script`. 0.0 if none."""
    hist = script_histogram(text)
    total = sum(hist.values())
    return hist.get(script, 0) / total if total else 0.0


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("​", "").replace("﻿", "")
    text = _URL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def looks_like_junk(text: str) -> bool:
    """Cheap boilerplate / scrape-artefact filter."""
    if not text:
        return True
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(len(text), 1) < 0.55:
        return True
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines and len(set(lines)) / len(lines) < 0.5:      # navbars, repeated menus
        return True
    if text.count("|") > len(text) / 40:                  # table dumps
        return True
    return False


def fingerprint(text: str) -> str:
    """Normalised hash for near-duplicate removal."""
    norm = re.sub(r"\W+", "", text.lower())[:2000]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()
