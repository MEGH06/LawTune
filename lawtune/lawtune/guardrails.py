"""Runtime guardrails.

Training makes hallucination rarer. It does not make it impossible, and on a 2B model
"rarer" is not good enough to ship. This module is the deterministic layer that runs
around every generation:

  INPUT   scope check + unsafe-intent check -> refuse before the model ever runs
  OUTPUT  citation validation -> a fabricated "Section 812 IPC" is caught by arithmetic,
          not by hope; language check; disclaimer injection

The citation validator is the piece that earns its keep. Every Indian statute has a
known highest section number. A model that emits Section 812 of the IPC is provably
wrong -- the IPC stops at 511 -- and we can say so without a retrieval index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .acts import LIMITS, identify_act
from .textutil import dominant_script

# ------------------------------------------------------------------ statute bounds
# Sourced from lawtune/acts.py so the graph extractor and this validator can never
# disagree about what "the IPC" means or where it stops.
STATUTE_LIMITS = LIMITS

SECTION_RE = re.compile(
    r"\b(?P<kind>Section|Sec\.?|S\.|Article|Art\.?|अनुच्छेद|धारा|பிரிவு|విభాగం|ವಿಧಿ|അനുച്ഛേദം)"
    r"\s*[-–]?\s*(?P<num>\d{1,4})(?P<suffix>[A-Za-z]{0,2})",
    re.IGNORECASE,
)
# "AIR 2050 SC 1", "(2031) 4 SCC 12"
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# ------------------------------------------------------------------ input screening
UNSAFE_PATTERNS = [
    (re.compile(r"\b(?:how (?:do|can) i |help me |best way to )?"
                r"(?:forge|fabricate|backdate|falsif\w+)\b", re.I), "document forgery"),
    (re.compile(r"\bbrib\w+\b.{0,40}\b(?:officer|police|judge|official|clerk)\b", re.I), "bribery"),
    (re.compile(r"\b(?:destroy|delete|burn|shred|wipe)\b.{0,30}\bevidence\b", re.I),
     "destruction of evidence"),
    (re.compile(r"\b(?:threaten|intimidat\w+|silence|bribe)\b.{0,30}\bwitness\b", re.I),
     "witness tampering"),
    (re.compile(r"\b(?:hide|conceal|launder)\b.{0,30}"
                r"\b(?:asset|money|income|property)\b", re.I), "concealment of assets"),
    (re.compile(r"\bfalse\b.{0,20}\b(?:fir|case|complaint|affidavit|testimony)\b", re.I),
     "filing a false complaint"),
    (re.compile(r"\b(?:evade|dodge|escape)\b.{0,25}\b(?:tax|summons|warrant|arrest)\b", re.I),
     "evading legal process"),
    (re.compile(r"\bfake\b.{0,25}\b(?:certificate|document|stamp|signature|id)\b", re.I),
     "document fraud"),
]

FOREIGN_LAW_PATTERNS = [
    (re.compile(r"\bmiranda\b|\b(?:first|second|fourth|fifth|14th|fourteenth) amendment\b", re.I),
     "United States"),
    (re.compile(r"\bgdpr\b|\beuropean union law\b", re.I), "European Union"),
    (re.compile(r"\b(?:uk|british|england)\b.{0,20}\b(?:act|law|court)\b", re.I),
     "United Kingdom"),
    (re.compile(r"\bsharia\b|\bcommon law of (?:england|canada|australia)\b", re.I), "foreign"),
]

REFUSAL_UNSAFE = (
    "I will not help with that -- {topic} is itself an offence under Indian law.\n\n"
    "If there is a lawful goal behind the question, tell me what outcome you want and I "
    "will explain the legal route to it. For anything specific to your matter, consult a "
    "licensed advocate."
)
REFUSAL_FOREIGN = (
    "That looks like {place} law, which is outside my scope -- I only cover Indian law. "
    "Ask me for the Indian equivalent and I will give you the provision with its Article "
    "or Section number."
)
DISCLAIMER = (
    "\n\n---\n*Legal information, not legal advice. Verify every provision on "
    "indiacode.nic.in and consult a licensed advocate before acting.*"
)


@dataclass
class Verdict:
    allowed: bool = True
    replacement: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    citations: List[dict] = field(default_factory=list)


def _statute_context(text: str, pos: int, window: int = 120) -> Optional[str]:
    """Which Act is the citation at `pos` talking about? Look right, then left."""
    right = text[pos: pos + window]
    left = text[max(0, pos - window): pos]
    return identify_act(right) or identify_act(left)


def check_input(question: str) -> Verdict:
    """Screen the user turn before the model sees it."""
    v = Verdict()
    for pattern, topic in UNSAFE_PATTERNS:
        if pattern.search(question):
            v.allowed = False
            v.flags.append(f"unsafe:{topic}")
            v.replacement = REFUSAL_UNSAFE.format(topic=topic)
            return v
    for pattern, place in FOREIGN_LAW_PATTERNS:
        if pattern.search(question):
            v.flags.append(f"foreign:{place}")
            v.allowed = False
            v.replacement = REFUSAL_FOREIGN.format(place=place)
            return v
    if len(question.strip()) < 3:
        v.allowed = False
        v.replacement = "Ask me a question about Indian law and I will answer it."
    return v


def validate_citations(answer: str) -> Tuple[List[dict], List[str]]:
    """Range-check every Section/Article the model emitted.

    Returns (citations, flags). A citation is 'suspect' when its number exceeds the
    highest provision that exists in the Act it was attributed to.
    """
    citations, flags = [], []
    for m in SECTION_RE.finditer(answer):
        num = int(m.group("num"))
        suffix = m.group("suffix") or ""
        kind = m.group("kind").lower()
        key = _statute_context(answer, m.end())
        if key is None:
            key = "constitution" if kind.startswith(("article", "art", "अनुच्छेद")) else None

        entry = {"text": m.group(0), "number": num, "suffix": suffix,
                 "statute": key, "status": "unverified"}
        if key and key in STATUTE_LIMITS:
            limit, name = STATUTE_LIMITS[key]
            if num == 0 or num > limit:
                entry["status"] = "impossible"
                entry["reason"] = f"{name} has no provision numbered {num} (max {limit})"
                flags.append(f"bad_citation:{m.group(0)}")
            else:
                entry["status"] = "in_range"
        citations.append(entry)

    for m in YEAR_RE.finditer(answer):
        year = int(m.group(0))
        if year > 2026 or year < 1860:
            flags.append(f"implausible_year:{year}")
    return citations, flags


def check_output(answer: str, question: str = "",
                 expected_script: Optional[str] = None,
                 add_disclaimer: bool = True) -> Verdict:
    """Screen the model's answer. Rewrites rather than blocks where it safely can."""
    v = Verdict()
    citations, flags = validate_citations(answer)
    v.citations, v.flags = citations, flags

    impossible = [c for c in citations if c["status"] == "impossible"]
    if impossible:
        # Do not paper over it and do not silently emit it. Say what happened.
        lines = "\n".join(f"  - {c['text']}: {c['reason']}" for c in impossible)
        v.allowed = False
        v.replacement = (
            "I generated a citation that does not exist, so I am withholding that "
            "answer rather than misleading you:\n\n" + lines +
            "\n\nRephrase the question with the subject matter (the offence, right, or "
            "procedure) instead of a section number, and I will try again. Cross-check "
            "any provision on indiacode.nic.in."
        )
        return v

    if expected_script:
        got = dominant_script(answer)
        if got and got != expected_script:
            v.flags.append(f"language_mismatch:expected={expected_script},got={got}")

    if not citations and len(answer.split()) > 60:
        # A long, confident, uncited legal answer is the classic shape of a fluent lie.
        v.flags.append("uncited_long_answer")

    text = answer
    if add_disclaimer and DISCLAIMER.strip()[:20] not in text:
        text += DISCLAIMER
    v.replacement = text
    return v
