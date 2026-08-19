#!/usr/bin/env python
"""Sample ~200 MB from ai4bharat/sangraha WITHOUT downloading the corpus.

The problem with your notebook: `load_dataset("ai4bharat/sangraha", name="verified")`
resolves the whole config, and `.shard()` only helps *after* the download. Sangraha
verified is ~1.5 TB of parquet. You cannot delete your way out of that on a laptop.

The fix: list the parquet shard paths over the HTTP filesystem, pick a random handful
per language, and *stream* rows out of them, stopping the moment the per-language byte
budget is met. Nothing lands on disk except the sample you keep.

Budget is split EVENLY across the 23 languages (~8.7 MB each at 200 MB total) rather
than proportionally, so Hindi and Bengali do not bury Santali and Bodo. A language
with less data than its budget simply contributes what it has.

Output: data/interim/sangraha_sample/  (HF dataset on disk, columns: text, lang)
        data/interim/sangraha_sample_stats.json

    python scripts/01_sample_sangraha.py                # 200 MB, all 23 languages
    python scripts/01_sample_sangraha.py --total-mb 60  # quick smoke run
    python scripts/01_sample_sangraha.py --langs hin,tam,ben
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset, disable_caching  # noqa: E402

from lawtune.config import (  # noqa: E402
    HF_TOKEN, INTERIM, LANGUAGES, SANGRAHA_MAX_DOC_CHARS, SANGRAHA_MIN_DOC_CHARS,
    SANGRAHA_SHARDS_PER_LANG, SANGRAHA_STREAM_CAP, SANGRAHA_TOTAL_MB, SEED,
)
from lawtune.textutil import clean, fingerprint, looks_like_junk, script_ratio  # noqa: E402

REPO = "ai4bharat/sangraha"
SUBSET = "verified"          # human-verified split; cleanest of the three
OUT_DIR = INTERIM / "sangraha_sample"
STATS = INTERIM / "sangraha_sample_stats.json"


def list_shards(lang: str, token: str | None) -> list[str]:
    """Return hf:// URLs for the parquet shards of one language."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    patterns = [
        f"datasets/{REPO}/{SUBSET}/{lang}/*.parquet",
        f"datasets/{REPO}/{SUBSET}/{lang}/**/*.parquet",
        f"datasets/{REPO}/data/{SUBSET}/{lang}/*.parquet",
    ]
    for pat in patterns:
        try:
            hits = fs.glob(pat)
        except Exception:
            continue
        if hits:
            return ["hf://" + h for h in hits]
    return []


def stream_rows(urls: list[str], token: str | None):
    """Yield dicts from parquet shards, pulling row groups lazily over HTTP."""
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files={"train": urls},
        split="train",
        streaming=True,
        token=token,
    )
    yield from ds


def sample_language(lang: str, budget_bytes: int, args, rng: random.Random) -> tuple[list[dict], dict]:
    _, _, _, script = LANGUAGES[lang]
    t0 = time.time()
    urls = list_shards(lang, HF_TOKEN)
    if not urls:
        return [], {"lang": lang, "status": "no_shards_found", "docs": 0, "bytes": 0}

    rng.shuffle(urls)
    picked = urls[: max(1, args.shards_per_lang)]

    kept: list[dict] = []
    seen: set[str] = set()
    got = scanned = skipped_script = skipped_junk = skipped_dup = 0

    try:
        for row in stream_rows(picked, HF_TOKEN):
            scanned += 1
            if scanned > args.stream_cap or got >= budget_bytes:
                break

            text = row.get("text") or row.get("content") or ""
            if not isinstance(text, str):
                continue
            text = clean(text)
            if not (SANGRAHA_MIN_DOC_CHARS <= len(text)):
                continue
            if len(text) > SANGRAHA_MAX_DOC_CHARS:
                text = text[:SANGRAHA_MAX_DOC_CHARS].rsplit(" ", 1)[0]

            # A "Tamil" doc that is mostly English is a scrape artefact, not Tamil data.
            if script_ratio(text, script) < args.min_script_ratio:
                skipped_script += 1
                continue
            if looks_like_junk(text):
                skipped_junk += 1
                continue
            fp = fingerprint(text)
            if fp in seen:
                skipped_dup += 1
                continue

            seen.add(fp)
            kept.append({"text": text, "lang": lang})
            got += len(text.encode("utf-8"))
    except Exception as exc:                                   # noqa: BLE001
        return kept, {"lang": lang, "status": f"partial: {type(exc).__name__}: {exc}",
                      "docs": len(kept), "bytes": got, "scanned": scanned,
                      "seconds": round(time.time() - t0, 1)}

    return kept, {
        "lang": lang,
        "status": "ok" if got >= budget_bytes * 0.9 else "under_budget",
        "docs": len(kept), "bytes": got, "mb": round(got / 1e6, 2), "scanned": scanned,
        "shards_used": len(picked), "shards_available": len(urls),
        "skipped_script": skipped_script, "skipped_junk": skipped_junk,
        "skipped_dup": skipped_dup, "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-mb", type=int, default=SANGRAHA_TOTAL_MB)
    ap.add_argument("--langs", type=str, default="", help="comma list; default all 23")
    ap.add_argument("--shards-per-lang", type=int, default=SANGRAHA_SHARDS_PER_LANG)
    ap.add_argument("--stream-cap", type=int, default=SANGRAHA_STREAM_CAP)
    ap.add_argument("--min-script-ratio", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    disable_caching()
    rng = random.Random(args.seed)
    langs = [l.strip() for l in args.langs.split(",") if l.strip()] or list(LANGUAGES)
    unknown = [l for l in langs if l not in LANGUAGES]
    if unknown:
        print(f"unknown language codes: {unknown}", file=sys.stderr)
        return 2

    budget = int(args.total_mb * 1e6 / len(langs))
    print(f"Sangraha sample: {args.total_mb} MB over {len(langs)} languages "
          f"= {budget / 1e6:.2f} MB each\n")

    all_rows: list[dict] = []
    stats: list[dict] = []
    for i, lang in enumerate(langs, 1):
        name = LANGUAGES[lang][0]
        print(f"[{i:>2}/{len(langs)}] {lang} ({name}) ... ", end="", flush=True)
        rows, st = sample_language(lang, budget, args, rng)
        all_rows.extend(rows)
        stats.append(st)
        print(f"{st['docs']:>6} docs  {st.get('mb', 0):>6.2f} MB  "
              f"[{st['status']}]  {st.get('seconds', 0)}s")

    if not all_rows:
        print("\nNothing sampled. Most likely causes:\n"
              "  1. Not authenticated -- run `huggingface-cli login` or export HF_TOKEN\n"
              "  2. Sangraha directory layout changed -- check list_shards() patterns\n",
              file=sys.stderr)
        return 1

    rng.shuffle(all_rows)
    ds = Dataset.from_list(all_rows)
    ds.save_to_disk(str(OUT_DIR))

    total_mb = sum(s.get("bytes", 0) for s in stats) / 1e6
    STATS.write_text(json.dumps(
        {"total_mb": round(total_mb, 2), "total_docs": len(all_rows),
         "budget_mb": args.total_mb, "seed": args.seed, "per_language": stats},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved {len(all_rows)} docs / {total_mb:.1f} MB -> {OUT_DIR}")
    print(f"Per-language report -> {STATS}")
    thin = [s["lang"] for s in stats if s.get("docs", 0) < 50]
    if thin:
        print(f"\nWARNING: thin coverage for {thin}. Sangraha genuinely has little data "
              f"for some of these. Raise --shards-per-lang or accept the imbalance; the "
              f"SFT stage carries the real multilingual load anyway.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
