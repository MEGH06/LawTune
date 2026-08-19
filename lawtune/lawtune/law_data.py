"""Load and normalise the Kaggle Indian-law QA corpora into one schema.

Target schema per record:
    {"question": str, "answer": str, "source": str, "lang": "eng",
     "citation": str|None}

The loader is deliberately key-tolerant: Kaggle law dumps use question/Question/
query/instruction/prompt interchangeably and answer/Answer/response/output/text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Any

from .config import RAW

Q_KEYS = ("question", "Question", "query", "instruction", "prompt", "input", "q")
A_KEYS = ("answer", "Answer", "response", "output", "completion", "text", "a")

# Article 21 / Section 302 IPC / Sec. 154 CrPC / S. 66A IT Act
CITATION_RE = re.compile(
    r"\b(?:Article|Art\.?|Section|Sec\.?|S\.)\s*[-–]?\s*(\d+[A-Z]{0,2}(?:\(\d+\))?)",
    re.IGNORECASE,
)


def _first(d: Dict[str, Any], keys: Iterable[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _clean(t: str) -> str:
    t = t.replace("​", "").replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _iter_records(payload: Any) -> Iterable[Dict[str, Any]]:
    """Handle list-of-dicts, {"data": [...]}, and JSONL-ish nesting."""
    if isinstance(payload, list):
        yield from (r for r in payload if isinstance(r, dict))
    elif isinstance(payload, dict):
        for key in ("data", "records", "rows", "train", "examples"):
            if isinstance(payload.get(key), list):
                yield from (r for r in payload[key] if isinstance(r, dict))
                return
        # dict-of-QA {question: answer}
        for k, v in payload.items():
            if isinstance(k, str) and isinstance(v, str):
                yield {"question": k, "answer": v}


def load_file(path: Path, source: str | None = None) -> List[Dict[str, Any]]:
    source = source or path.stem
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
        records = list(_iter_records(payload))
    except json.JSONDecodeError:                       # assume JSONL
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)

    out = []
    for r in records:
        q, a = _clean(_first(r, Q_KEYS)), _clean(_first(r, A_KEYS))
        if len(q) < 8 or len(a) < 20:
            continue
        if q.lower() == a.lower():
            continue
        m = CITATION_RE.search(a) or CITATION_RE.search(q)
        out.append({
            "question": q,
            "answer": a,
            "source": source,
            "lang": "eng",
            "citation": m.group(0) if m else None,
        })
    return out


def load_all(raw_dir: Path = RAW) -> List[Dict[str, Any]]:
    """Read every .json/.jsonl under data/raw/ and dedupe on the question."""
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in {".json", ".jsonl"})
    if not files:
        raise FileNotFoundError(
            f"No .json/.jsonl found in {raw_dir}. Put your Kaggle law files there "
            f"(constitution_qa.json, crpc_qa.json, ipc_qa.json, ...) or run "
            f"`python scripts/00_fetch_kaggle.py --dataset <owner/slug>`."
        )
    seen, merged = set(), []
    for f in files:
        recs = load_file(f)
        kept = 0
        for r in recs:
            key = re.sub(r"\W+", "", r["question"].lower())[:160]
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
            kept += 1
        print(f"  {f.name:<32} {len(recs):>6} parsed  {kept:>6} kept after dedupe")
    print(f"  TOTAL unique English law QA pairs: {len(merged)}")
    return merged
