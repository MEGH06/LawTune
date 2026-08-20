"""Pull graph facts out of judgment text and metadata.

Rule-based on purpose. An LLM extraction pass over ~100k judgments costs either money or
days of local GPU; regexes cost seconds and, more importantly, are *inspectable* -- when
an edge looks wrong you can see exactly which pattern produced it. Indian legal citation
is highly conventional, so the ceiling on rules here is high.

What gets extracted:
  citations  -> CITES edges (with treatment: followed / distinguished / overruled ...)
  provisions -> INTERPRETS edges, resolved against lawtune/acts.py
  areas      -> ABOUT edges, weighted (the taxonomy covers all of law, not one vertical)
  doctrines  -> INVOKES edges
  bench      -> DECIDED_BY edges, and bench strength
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from ..acts import BY_KEY, identify_act, nearest_act
from .schema import AREAS, DOCTRINES

# --------------------------------------------------------------------- citations
# Indian reporters, in the forms that actually appear in judgment text.
CITATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 2023 INSC 456  -- the Supreme Court's neutral citation, used from 2023
    ("INSC", re.compile(r"\b((?:19|20)\d{2})\s+INSC\s+(\d{1,5})\b", re.I)),
    # (2020) 5 SCC 1   |   2020 (5) SCC 1
    ("SCC", re.compile(r"\(?\b((?:19|20)\d{2})\)?\s*\(?(\d{1,2})\)?\s*SCC\s+(\d{1,5})\b")),
    # 2021 SCC OnLine SC 123
    ("SCC_ONLINE", re.compile(r"\b((?:19|20)\d{2})\s+SCC\s+OnLine\s+([A-Z]{2,4})\s+(\d{1,6})\b", re.I)),
    # AIR 1973 SC 1461
    ("AIR", re.compile(r"\bAIR\s+((?:19|20)\d{2})\s+([A-Z]{2,4})\s+(\d{1,5})\b")),
    # [2024] 10 S.C.R. 1   -- the official Supreme Court Reports
    ("SCR", re.compile(r"\[?\b((?:19|20)\d{2})\]?\s*(\d{1,2})\s*S\.?\s?C\.?\s?R\.?\s+(\d{1,5})\b")),
    # (2019) 3 SCC (Cri) 45
    ("SCC_SUPP", re.compile(r"\b((?:19|20)\d{2})\s+Supp\s*\(?(\d)\)?\s*SCC\s+(\d{1,5})\b", re.I)),
]

# "X v. Y" -- deliberately conservative. Requires capitalised tokens on both sides so
# that "the appeal v. the respondent" in running prose does not become a case name.
_NAME_TOKEN = r"(?:[A-Z][\w.&'-]*|of|the|and|for|&)"
CASE_NAME_RE = re.compile(
    rf"\b([A-Z][\w.&'-]*(?:\s+{_NAME_TOKEN}){{0,7}})\s+"
    rf"(?:v\.?|vs\.?|versus)\s+"
    rf"([A-Z][\w.&'-]*(?:\s+{_NAME_TOKEN}){{0,7}})"
)
# Lowercase connectors are allowed mid-name ("State of Kerala") but must not dangle
# off the end ("Government of" -> "Government").
_TRAILING_CONNECTOR = re.compile(r"\s+(?:of|the|and|for|&)$", re.I)

# How the citing court treated the cited case. Checked in a window around the citation.
#
# Inflections are spelled out rather than written as `\bfollow\w+`. That form requires
# at least one trailing character, so it matched "followed" but NOT the bare "we follow
# X" -- and courts write the base form constantly ("we follow", "we distinguish", "we
# affirm", "we refer to"). Five of these seven verbs had that defect, which silently
# downgraded real treatment to a plain "cited" on the CITES edge, and in turn cost the
# retriever its overruled/distinguished warnings.
TREATMENTS: list[tuple[str, re.Pattern]] = [
    ("overruled",     re.compile(r"\boverrul(?:e|es|ed|ing)\b|\bno longer good law\b",
                                 re.I)),
    ("reversed",      re.compile(r"\brevers(?:e|es|ed|ing)\b|"
                                 r"\bset aside the judgment\b", re.I)),
    ("distinguished", re.compile(r"\bdistinguish(?:es|ed|ing|able)?\b|"
                                 r"\bis not applicable\b|\bturns on its own facts\b",
                                 re.I)),
    ("dissented",     re.compile(r"\bdissent(?:s|ed|ing)?\b|"
                                 r"\bwith respect, we are unable to agree\b", re.I)),
    ("affirmed",      re.compile(r"\baffirm(?:s|ed|ing)?\b|"
                                 r"\buph(?:eld|olds?|olding)\b", re.I)),
    ("followed",      re.compile(r"\bfollow(?:s|ed|ing)?\b|"
                                 r"\brel(?:y|ies|ied) (?:up)?on\b|"
                                 r"\breiterat(?:e|es|ed|ing)\b|"
                                 r"\bapplied\b|\bapplying\b|"
                                 r"\bapprov(?:e|es|ed|ing)\b", re.I)),
    ("referred",      re.compile(r"\brefer(?:s|red|ring)?\s+to\b|\bsee also\b|\bcf\.\b",
                                 re.I)),
]

# NOTE: no global re.IGNORECASE here. With it, [A-Z] also matches lowercase, so
# "Section 302 of the Indian Penal Code" parsed "of" as the suffix and produced
# "Section 302OF". Case-insensitivity is scoped to the kind alternation instead, and
# the suffix must be adjacent, uppercase, and not the head of a longer word.
PROVISION_RE = re.compile(
    r"\b(?P<kind>(?i:Sections?|Secs?\.?|S\.|Articles?|Arts?\.?|Rules?|Orders?|Clauses?))"
    r"\s*(?P<num>\d{1,4})(?P<suffix>[A-Z]{1,2})?(?![A-Za-z])"
    r"(?:\s*\(\s*(?P<sub>[0-9a-z]{1,3})\s*\))?"
)

_KIND_CANON = {"section": "Section", "sec": "Section", "s": "Section",
               "article": "Article", "art": "Article",
               "rule": "Rule", "order": "Order", "clause": "Clause"}


def _canon_kind(raw: str) -> str:
    base = raw.lower().rstrip(".").rstrip("s") or raw.lower()
    return _KIND_CANON.get(base, _KIND_CANON.get(raw.lower().rstrip("."), "Section"))


def normalise_citation(kind: str, groups: tuple) -> str:
    """A stable key so two spellings of the same citation collapse to one node."""
    if kind == "INSC":
        return f"insc:{groups[0]}:{groups[1]}"
    if kind == "SCC":
        return f"scc:{groups[0]}:{groups[1]}:{groups[2]}"
    if kind == "SCC_ONLINE":
        return f"sconline:{groups[0]}:{groups[1].lower()}:{groups[2]}"
    if kind == "AIR":
        return f"air:{groups[0]}:{groups[1].lower()}:{groups[2]}"
    if kind == "SCR":
        return f"scr:{groups[0]}:{groups[1]}:{groups[2]}"
    if kind == "SCC_SUPP":
        return f"sccsupp:{groups[0]}:{groups[1]}:{groups[2]}"
    return ":".join([kind.lower(), *map(str, groups)])


def _nearest_treatment(text: str, start: int, end: int, window: int) -> str:
    """The treatment word closest to the citation wins.

    Priority order alone is wrong: a judgment that distinguishes case A and follows
    case B mentions both verbs, and whichever sits higher in TREATMENTS would be
    stamped on every citation in the vicinity.
    """
    lo, hi = max(0, start - window), end + window
    scope = text[lo:hi]
    best, best_dist = "cited", 10 ** 9
    for name, tre in TREATMENTS:
        for m in tre.finditer(scope):
            pos = lo + m.start()
            dist = start - pos if pos < start else pos - end
            if dist < best_dist:
                best, best_dist = name, dist
    return best


def extract_citations(text: str, window: int = 160) -> list[dict]:
    """Every reporter citation in the text, with how it was treated."""
    out: dict[str, dict] = {}
    for kind, pattern in CITATION_PATTERNS:
        for m in pattern.finditer(text):
            key = normalise_citation(kind, m.groups())
            treatment = _nearest_treatment(text, m.start(), m.end(), window)
            prev = out.get(key)
            if prev is None:
                # PDFs hard-wrap citations across lines. Collapse the whitespace, or
                # the line break ends up inside the graph node's title.
                out[key] = {"key": key, "reporter": kind,
                            "raw": re.sub(r"\s+", " ", m.group(0).strip()),
                            "treatment": treatment, "count": 1}
            else:
                prev["count"] += 1
                # A stronger signal anywhere in the judgment wins over a bare mention.
                if prev["treatment"] == "cited":
                    prev["treatment"] = treatment
    return list(out.values())


def extract_case_names(text: str, limit: int = 60) -> list[str]:
    """Party-style case names, deduped and length-filtered."""
    seen, out = set(), []
    for m in CASE_NAME_RE.finditer(text):
        left = _TRAILING_CONNECTOR.sub("", m.group(1).strip())
        right = _TRAILING_CONNECTOR.sub("", m.group(2).strip())
        if len(left) < 3 or len(right) < 3:
            continue
        name = re.sub(r"\s+", " ", f"{left} v. {right}")
        if len(name) > 140:
            continue
        k = re.sub(r"\W+", "", name.lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(name)
        if len(out) >= limit:
            break
    return out


def extract_provisions(text: str, window: int = 140,
                       max_act_distance: int = 60) -> list[dict]:
    """Section/Article references, resolved to the Act they belong to.

    A reference we cannot attribute to an Act is dropped rather than guessed at -- an
    unattributed "Section 5" is noise, and noise in a knowledge graph is worse than a
    gap because it looks like knowledge.
    """
    found: dict[str, dict] = {}
    for m in PROVISION_RE.finditer(text):
        kind = _canon_kind(m.group("kind"))
        num = int(m.group("num"))
        suffix = (m.group("suffix") or "").upper()

        lo = max(0, m.start() - window)
        scope = text[lo: m.end() + window]
        act_key = nearest_act(scope, m.start() - lo, m.end() - lo,
                              max_distance=max_act_distance)

        if kind == "Article":
            # Indian statutes are divided into Sections; only the Constitution is
            # divided into Articles. So "Article 21" is the Constitution unless some
            # other Act is named right up against it. Without this rule, distance
            # alone mis-assigns "...of the BNSS and Article 21(1) of the Constitution"
            # to the BNSS, because the two Act mentions are near-equidistant.
            tail = text[m.end(): m.end() + 30]
            explicit = identify_act(tail)
            act_key = explicit if explicit and explicit != "constitution" else "constitution"
        if act_key is None:
            continue

        act = BY_KEY.get(act_key)
        if act and act.max_provision and (num == 0 or num > act.max_provision):
            continue                          # OCR noise or a misprint; do not store it

        pid = f"{act_key}:{kind.lower()}:{num}{suffix.lower()}"
        entry = found.get(pid)
        if entry:
            entry["mentions"] += 1
        else:
            label = f"{kind} {num}{suffix}, {act.name if act else act_key}"
            found[pid] = {"pid": pid, "act_key": act_key, "kind": kind, "number": num,
                          "suffix": suffix, "label": label, "mentions": 1}
    return list(found.values())


def extract_acts(text: str) -> list[dict]:
    """Acts named anywhere, with mention counts."""
    counts: Counter = Counter()
    for act in BY_KEY.values():
        pattern = re.compile("|".join(act.patterns), re.IGNORECASE)
        n = len(pattern.findall(text))
        if n:
            counts[act.key] = n
    return [{"act_key": k, "mentions": n} for k, n in counts.most_common()]


# Whole-word matchers, compiled once. Substring counting was catastrophically wrong:
# the term "cat" (for Central Administrative Tribunal) matched inside certifiCATe,
# classifiCATion, indiCATed, appliCATion and eduCATion, which filed a sales-tax appeal
# under Service Law with a score of 0.50. Any term list scored with `str.count` has this
# failure mode; \b is the whole fix.
# Each term is compiled separately so it can be weighted by specificity: a three-word
# phrase like "letters of administration" is far more diagnostic of an area than a bare
# noun, and scoring every term equally lets the vague ones outvote the precise ones.
_AREA_MATCHERS: dict[str, list[tuple[re.Pattern, float]]] = {
    key: [(re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE), float(len(t.split())))
          for t in terms]
    for key, (_, _, terms) in AREAS.items()
}
_DOCTRINE_MATCHERS: dict[str, re.Pattern] = {
    key: re.compile("|".join(rf"\b{re.escape(t)}\b" for t in terms), re.IGNORECASE)
    for key, terms in DOCTRINES.items()
}


def classify_areas(text: str, title: str = "", top_k: int = 4,
                   min_score: float = 0.08) -> list[dict]:
    """Score the judgment against every area of law; keep the strongest few.

    Title terms count triple: a judgment titled "... v. Commissioner of Income Tax" is
    a tax matter even if the body spends pages on limitation.
    """
    blob = (title + " ") * 3 + text
    raw: dict[str, float] = {}
    for key, matchers in _AREA_MATCHERS.items():
        score = sum(len(pat.findall(blob)) * weight for pat, weight in matchers)
        if score:
            raw[key] = score
    if not raw:
        return []
    total = sum(raw.values())
    ranked = sorted(raw.items(), key=lambda kv: -kv[1])[:top_k]
    return [{"area": k, "score": round(v / total, 4)}
            for k, v in ranked if v / total >= min_score]


def extract_doctrines(text: str) -> list[dict]:
    out = []
    for key, matcher in _DOCTRINE_MATCHERS.items():
        n = len(matcher.findall(text))
        if n:
            out.append({"doctrine": key, "mentions": n})
    return out


JUDGE_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|&|\n)\s*", re.IGNORECASE)
JUDGE_NOISE_RE = re.compile(
    r"\b(hon(?:'?ble)?|mr|mrs|ms|dr|justice|j\.?|cji|chief justice|the)\b\.?",
    re.IGNORECASE)


def parse_judges(raw: str | None) -> list[str]:
    """Split and clean the metadata judge string into individual names."""
    if not raw or not isinstance(raw, str):
        return []
    names = []
    for part in JUDGE_SPLIT_RE.split(raw):
        name = JUDGE_NOISE_RE.sub(" ", part)
        name = re.sub(r"[^\w\s.'-]", " ", name)
        name = re.sub(r"\s+", " ", name).strip(" .,-")
        if len(name) >= 4 and any(c.isalpha() for c in name):
            names.append(name.title())
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def extract_all(text: str, title: str = "", judges_raw: str | None = None) -> dict:
    """One pass, everything the graph builder needs."""
    text = text or ""
    return {
        "citations": extract_citations(text),
        "case_names": extract_case_names(text),
        "provisions": extract_provisions(text),
        "acts": extract_acts(text),
        "areas": classify_areas(text, title),
        "doctrines": extract_doctrines(text),
        "judges": parse_judges(judges_raw),
    }


def summarise(records: Iterable[dict]) -> dict:
    """Aggregate counts, for the build report."""
    agg: Counter = Counter()
    for r in records:
        for field in ("citations", "provisions", "areas", "doctrines", "judges", "acts"):
            agg[field] += len(r.get(field, []))
    return dict(agg)
