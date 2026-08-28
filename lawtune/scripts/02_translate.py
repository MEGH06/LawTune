#!/usr/bin/env python
"""Fan the English law QA + guardrail sets out to all 22 scheduled languages.

This is the step that actually makes LawTune multilingual. Sangraha teaches the model
what Odia *looks like*; it does not teach it to answer a bail question in Odia. Only
parallel instruction data does that, and none exists for Indian law, so we build it
with IndicTrans2.

Resumable by design -- one JSONL per language under data/interim/translated/. This is
the longest step in the pipeline; Ctrl-C is safe and re-running picks up the languages
it has not finished.

    python scripts/02_translate.py                     # all 22
    python scripts/02_translate.py --langs hin,tam     # subset
    python scripts/02_translate.py --pairs-per-lang 300 --model ai4bharat/indictrans2-en-indic-1B
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from lawtune import guard_data, law_data  # noqa: E402
from lawtune.config import (  # noqa: E402
    INDIC_LANGS, INTERIM, LANGUAGES, SEED, TRANSLATE_BATCH, TRANSLATE_BEAMS,
    TRANSLATE_MAX_SRC_CHARS, TRANSLATE_MODEL, TRANSLATE_PAIRS_PER_LANG,
)
from lawtune.textutil import script_ratio  # noqa: E402

OUT_DIR = INTERIM / "translated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IndicTrans2 is a sentence-level model. Feeding it a paragraph degrades badly, so we
# split on terminators including the Devanagari danda and the Urdu full stop.
_SENT_RE = re.compile(r"(?<=[.!?।॥۔])\s+")
# Sentinel that separates the question from the answer inside one translation batch,
# so both cross the GPU in a single call instead of two.
MARKER = "<<<SPLIT>>>"

def split_sentences(text: str, max_chars: int = 400) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            out.append("")            # preserve paragraph breaks as empty markers
            continue
        for sent in _SENT_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            while len(sent) > max_chars:            # hard-wrap runaway sentences
                cut = sent.rfind(",", 0, max_chars)
                cut = cut if cut > max_chars // 3 else max_chars
                out.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if sent:
                out.append(sent)
        out.append("")
    while out and out[-1] == "":
        out.pop()
    return out


def rejoin(sentences: list[str]) -> str:
    text = ""
    for s in sentences:
        if s == "":
            text += "\n"
        else:
            text += (" " if text and not text.endswith("\n") else "") + s
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def pick_device(preferred: str = "") -> str:
    """MPS if Metal is available, else CPU. IndicTrans2 runs fine on MPS."""
    if preferred:
        return preferred
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Translator:
    def __init__(self, model_name: str, device: str, beams: int):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            try:
                from IndicTransToolkit import IndicProcessor      # older layout
            except ImportError as exc:
                raise SystemExit(
                    "IndicTransToolkit is required:\n"
                    "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
                ) from exc

        self.ip = IndicProcessor(inference=True)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # float32 always: Metal's fp16 path produces NaNs in this model's encoder
        # attention, and a NaN here silently poisons a whole language's training data.
        dtype = torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype,
        ).to(device).eval()
        self.device = device
        self.beams = beams

    @torch.inference_mode()
    def translate(self, sentences: list[str], tgt: str, batch_size: int) -> list[str]:
        """Translate real sentences; empty strings and MARKER pass through untouched.

        Positions are preserved, so the caller can still index the result by the marker.
        """
        idx = [i for i, s in enumerate(sentences) if s and s != MARKER]
        payload = [sentences[i] for i in idx]
        results: list[str] = ["" if s != MARKER else MARKER for s in sentences]

        for start in range(0, len(payload), batch_size):
            chunk = payload[start:start + batch_size]
            pre = self.ip.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang=tgt)
            enc = self.tok(pre, truncation=True, padding="longest",
                           return_tensors="pt", max_length=256).to(self.device)
            gen = self.model.generate(
                **enc, num_beams=self.beams, num_return_sequences=1,
                # use_cache=False: IndicTrans2's remote modeling code predates the
                # transformers Cache API rewrite. From ~4.5x, generate() passes a cache
                # object instead of None on the first decode step, so its
                # `past_key_values[0][0].shape[2] if past_key_values is not None` guard
                # passes and then dereferences an empty entry -> AttributeError on every
                # row. Disabling the cache avoids that path. Costs ~2.2 sent/s vs faster
                # cached decoding, but it is correct.
                max_length=256, min_length=0, use_cache=False,
            )
            dec = self.tok.batch_decode(gen, skip_special_tokens=True)
            post = self.ip.postprocess_batch(dec, lang=tgt)
            for j, t in enumerate(post):
                results[idx[start + j]] = t
        return results


def load_source_pool(pairs_per_lang: int, seed: int) -> list[dict]:
    """English law QA (sampled) + the FULL guardrail set (never sampled down)."""
    print("Loading English law corpus...")
    law = law_data.load_all()
    guards = guard_data.build()

    rng = random.Random(seed)
    # Prefer records that carry a citation and are neither trivially short nor huge --
    # those translate cleanly and are the ones worth having in 22 languages.
    scored = sorted(
        law,
        key=lambda r: (r["citation"] is None, abs(len(r["answer"]) - 700)),
    )
    pool = scored[: pairs_per_lang * 3]
    rng.shuffle(pool)
    pool = pool[:pairs_per_lang]

    print(f"  {len(pool)} law pairs + {len(guards)} guardrail pairs per language")
    return pool + guards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", type=str, default="")
    ap.add_argument("--pairs-per-lang", type=int, default=TRANSLATE_PAIRS_PER_LANG)
    ap.add_argument("--model", type=str, default=TRANSLATE_MODEL)
    ap.add_argument("--batch-size", type=int, default=TRANSLATE_BATCH)
    ap.add_argument("--beams", type=int, default=TRANSLATE_BEAMS)
    ap.add_argument("--device", type=str, default="",
                    help="mps | cpu (default: mps when available)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="redo languages already done")
    args = ap.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()] or INDIC_LANGS
    bad = [l for l in langs if l not in LANGUAGES or l == "eng"]
    if bad:
        print(f"bad language codes: {bad}", file=sys.stderr)
        return 2

    todo = [l for l in langs if args.force or not (OUT_DIR / f"{l}.jsonl").exists()]
    if not todo:
        print("All requested languages already translated. Use --force to redo.")
        return 0
    print(f"To translate: {todo}")

    device = pick_device(args.device)
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: CPU only. Expect this to take many hours.")

    batch = args.batch_size if device != "mps" else min(args.batch_size, 16)
    pool = load_source_pool(args.pairs_per_lang, args.seed)
    print(f"Loading {args.model} on {device}...")
    tr = Translator(args.model, device, args.beams)

    for n, lang in enumerate(todo, 1):
        name, _, flores, script = LANGUAGES[lang]
        out_path = OUT_DIR / f"{lang}.jsonl"
        tmp_path = OUT_DIR / f"{lang}.jsonl.partial"
        t0 = time.time()
        print(f"\n[{n}/{len(todo)}] {lang} ({name}) -> {flores}")

        written = dropped = 0
        with tmp_path.open("w", encoding="utf-8") as fh:
            for i, rec in enumerate(pool):
                q = rec["question"][:TRANSLATE_MAX_SRC_CHARS]
                a = rec["answer"][:TRANSLATE_MAX_SRC_CHARS]
                q_sents, a_sents = split_sentences(q), split_sentences(a)
                merged = q_sents + [MARKER] + a_sents
                try:
                    out = tr.translate(merged, flores, batch)
                except Exception as exc:                       # noqa: BLE001
                    print(f"    row {i} failed ({type(exc).__name__}: {exc}); skipped")
                    dropped += 1
                    continue

                cut = merged.index(MARKER)
                q_t, a_t = rejoin(out[:cut]), rejoin(out[cut + 1:])
                if not q_t or not a_t:
                    dropped += 1
                    continue
                # Quality gate: if IndicTrans2 fell back to copying English, drop it.
                # A bad translation in training data is worse than no translation.
                if script_ratio(a_t, script) < 0.5:
                    dropped += 1
                    continue

                fh.write(json.dumps({
                    "question": q_t, "answer": a_t, "lang": lang,
                    "source": rec["source"], "citation": rec.get("citation"),
                    "question_en": rec["question"],
                }, ensure_ascii=False) + "\n")
                written += 1

                if (i + 1) % 100 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    eta = (len(pool) - i - 1) / max(rate, 1e-6) / 60
                    print(f"    {i + 1}/{len(pool)}  kept={written} dropped={dropped}  "
                          f"{rate:.1f} rec/s  ETA {eta:.0f} min")

        tmp_path.replace(out_path)
        print(f"  done: {written} kept, {dropped} dropped, "
              f"{(time.time() - t0) / 60:.1f} min -> {out_path}")

    print(f"\nTranslations in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
