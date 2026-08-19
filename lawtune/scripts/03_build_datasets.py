#!/usr/bin/env python
"""Assemble the two training corpora.

  data/processed/cpt/    plain text, 23 languages   -> stage A (continued pretraining)
  data/processed/sft/    chat messages, 23 languages -> stage B (instruction tuning)
  data/processed/eval/   held-out slice, never trained on

Stage A and stage B are deliberately separate. Sangraha is raw web text; wrapping it in
"### Instruction:" the way the original notebook did teaches the model that the correct
response to a legal question is a paragraph of scraped Odia news. Keeping the objectives
apart is most of the difference between a model that works and one that does not.

    python scripts/03_build_datasets.py
    python scripts/03_build_datasets.py --max-english 20000 --eval-per-lang 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import Dataset, load_from_disk  # noqa: E402

from lawtune import guard_data, law_data  # noqa: E402
from lawtune.config import (  # noqa: E402
    INTERIM, LANGUAGES, PROCESSED, SEED, SYSTEM_PROMPT,
)

CPT_IN = INTERIM / "sangraha_sample"
TR_IN = INTERIM / "translated"
CPT_OUT = PROCESSED / "cpt"
SFT_OUT = PROCESSED / "sft"
EVAL_OUT = PROCESSED / "eval"
REPORT = PROCESSED / "build_report.json"


def to_messages(rec: dict) -> dict:
    """Gemma-2 has no system role, so the persona rides on the first user turn.

    chat.py builds the identical prefix at inference time. If you change one, change
    both, or the model sees a prompt shape it was never trained on.
    """
    user = SYSTEM_PROMPT + "\n\n---\n\n" + rec["question"].strip()
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": rec["answer"].strip()},
        ],
        "lang": rec.get("lang", "eng"),
        "source": rec.get("source", "unknown"),
    }


def build_cpt(args, rng: random.Random) -> Dataset | None:
    if not CPT_IN.exists():
        print(f"! {CPT_IN} missing -- run scripts/01_sample_sangraha.py first. "
              f"Skipping stage-A corpus.")
        return None
    ds = load_from_disk(str(CPT_IN))
    print(f"CPT: {len(ds)} Sangraha docs, "
          f"{sum(len(t.encode()) for t in ds['text']) / 1e6:.1f} MB")

    # A language tag in front of each document is a cheap, strong signal: it gives the
    # model an explicit handle on "which language am I in" instead of inferring it from
    # script alone, which matters for the four Devanagari-sharing languages.
    def tag(row):
        name = LANGUAGES.get(row["lang"], ("Unknown",))[0]
        return {"text": f"<<{name}>>\n{row['text']}"}

    ds = ds.map(tag, desc="tagging language")
    ds = ds.shuffle(seed=args.seed)
    ds.save_to_disk(str(CPT_OUT))
    print(f"  -> {CPT_OUT}")
    return ds


def build_sft(args, rng: random.Random) -> tuple[Dataset, Dataset, dict]:
    records: list[dict] = []

    # 1. English law QA -- the substantive core
    print("\nSFT: English law corpus")
    english = law_data.load_all()
    rng.shuffle(english)
    if args.max_english and len(english) > args.max_english:
        print(f"  capping English at {args.max_english} "
              f"(was {len(english)}) so it does not swamp the 22 languages")
        english = english[: args.max_english]
    records.extend(english)

    # 2. Guardrails in English, upsampled -- see lawtune/guard_data.py for why
    guards_en = guard_data.build(n_per_bucket=args.guard_per_bucket)
    records.extend(guards_en)
    print(f"  {len(guards_en)} English guardrail examples")

    # 3. Translated law + guardrails, 22 languages
    print("\nSFT: translated corpora")
    if not TR_IN.exists() or not any(TR_IN.glob("*.jsonl")):
        print("  ! no translations found -- run scripts/02_translate.py.")
        print("  ! Training now would give you an English-only model.")
        if not args.allow_english_only:
            raise SystemExit(
                "Refusing to build an English-only 'multilingual' dataset. "
                "Run 02_translate.py, or pass --allow-english-only if that is what "
                "you actually want."
            )
    else:
        for path in sorted(TR_IN.glob("*.jsonl")):
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            records.extend(rows)
            print(f"  {path.stem:<5} {len(rows):>6}")

    # 4. Held-out eval: strip N per language BEFORE any duplication
    by_lang: dict[str, list[dict]] = {}
    for r in records:
        by_lang.setdefault(r.get("lang", "eng"), []).append(r)

    eval_rows, train_rows = [], []
    for lang, group in by_lang.items():
        rng.shuffle(group)
        k = min(args.eval_per_lang, max(0, len(group) // 10))
        eval_rows.extend(group[:k])
        train_rows.extend(group[k:])

    # 5. Upsample the thin languages toward the median so the model does not simply
    #    learn "when unsure, answer in Hindi".
    counts = Counter(r.get("lang", "eng") for r in train_rows)
    if counts:
        target = int(sorted(counts.values())[len(counts) // 2] * args.balance)
        by_lang_train: dict[str, list[dict]] = {}
        for r in train_rows:
            by_lang_train.setdefault(r.get("lang", "eng"), []).append(r)
        balanced: list[dict] = []
        for lang, group in by_lang_train.items():
            balanced.extend(group)
            if 0 < len(group) < target:
                need = target - len(group)
                balanced.extend(rng.choice(group) for _ in range(need))
        train_rows = balanced

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    train_ds = Dataset.from_list([to_messages(r) for r in train_rows])
    eval_ds = Dataset.from_list([to_messages(r) for r in eval_rows])
    train_ds.save_to_disk(str(SFT_OUT))
    eval_ds.save_to_disk(str(EVAL_OUT))

    stats = {
        "train": len(train_ds), "eval": len(eval_ds),
        "per_language": dict(sorted(Counter(train_ds["lang"]).items())),
        "per_source": dict(sorted(Counter(
            s.split(":")[0] for s in train_ds["source"]).items())),
    }
    return train_ds, eval_ds, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-english", type=int, default=25000)
    ap.add_argument("--guard-per-bucket", type=int, default=250)
    ap.add_argument("--eval-per-lang", type=int, default=40)
    ap.add_argument("--balance", type=float, default=1.0,
                    help="upsample thin languages to balance*median count")
    ap.add_argument("--allow-english-only", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cpt = build_cpt(args, rng)
    train_ds, eval_ds, stats = build_sft(args, rng)

    print("\n" + "=" * 60)
    print(f"CPT docs      : {len(cpt) if cpt is not None else 0}")
    print(f"SFT train     : {stats['train']}")
    print(f"SFT eval      : {stats['eval']}")
    print("\nper language:")
    for lang, n in stats["per_language"].items():
        print(f"  {lang:<5} {LANGUAGES.get(lang, ('?',))[0]:<12} {n:>7}")
    print("\nper source:")
    for src, n in stats["per_source"].items():
        print(f"  {src:<20} {n:>7}")

    missing = [l for l in LANGUAGES if l not in stats["per_language"]]
    if missing:
        print(f"\nWARNING: no SFT data at all for {missing}. The model will not speak "
              f"these. Re-run 02_translate.py for them.")

    REPORT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {REPORT}")
    print("\nsample training example:")
    ex = train_ds[0]
    print(f"  lang={ex['lang']} source={ex['source']}")
    print("  user     :", ex["messages"][0]["content"][-220:].replace("\n", " "))
    print("  assistant:", ex["messages"][1]["content"][:220].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
