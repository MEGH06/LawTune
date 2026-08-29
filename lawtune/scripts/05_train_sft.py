#!/usr/bin/env python
"""Stage B on Apple Silicon -- instruction tuning on Indian law in 23 languages.

LoRA on 4-bit weights via MLX, loss computed on the assistant turn only.

This is the stage that matters. If you run nothing else, run this.

    python scripts/05_train_sft.py                  # 6000 iters; 6 h on a Max, 25 h on an M1
    python scripts/05_train_sft.py --iters 20       # ~2 min; read the tok/s and ETA
    python scripts/05_train_sft.py --batch-size 4 --grad-accum 2   # same batch, ~1.5-2x faster
    python scripts/05_train_sft.py --no-cpt         # ignore stage A
    python scripts/05_train_sft.py --num-layers 12  # faster, lighter, a bit weaker
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk  # noqa: E402

from lawtune.config import (  # noqa: E402
    BASE_MODEL, CPT_FUSED, SFT, SFT_ADAPTER, PROCESSED, SEED,
)
from lawtune.mlx_train import train  # noqa: E402
from lawtune.mlx_utils import (  # noqa: E402
    apply_lora, encode_chat_example, load_model, require_mlx,
)

SFT_DATA = PROCESSED / "sft"
EVAL_DATA = PROCESSED / "eval"


def encode_split(tokenizer, ds, seq_len: int, limit: int = 0):
    examples, skipped = [], 0
    rows = ds if not limit else ds.select(range(min(limit, len(ds))))
    for messages in rows["messages"]:
        user = [m for m in messages if m["role"] == "user"]
        assistant = [m for m in messages if m["role"] == "assistant"]
        if not user or not assistant:
            skipped += 1
            continue
        ids, mask = encode_chat_example(tokenizer, user[:1], assistant[-1]["content"],
                                        seq_len)
        if sum(mask) < 8:
            skipped += 1
            continue
        examples.append((ids, mask))
    if skipped:
        print(f"  skipped {skipped} malformed/over-long examples")
    return examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--iters", type=int, default=SFT["iters"])
    ap.add_argument("--batch-size", type=int, default=SFT["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=SFT["grad_accum"])
    ap.add_argument("--lr", type=float, default=SFT["learning_rate"])
    ap.add_argument("--rank", type=int, default=SFT["rank"])
    ap.add_argument("--seq-len", type=int, default=SFT["seq_len"])
    ap.add_argument("--resume", action="store_true",
                    help="reload adapters.safetensors before training, so a crashed "
                         "or interrupted run continues instead of restarting at step 1")
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--no-cpt", action="store_true")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    require_mlx()

    if not SFT_DATA.exists():
        print(f"{SFT_DATA} missing. Run scripts/03_build_datasets.py first.",
              file=sys.stderr)
        return 1

    if args.model:
        base = args.model
    elif args.no_cpt or not CPT_FUSED.exists():
        base = BASE_MODEL
        if not args.no_cpt:
            print(f"[info] no fused stage-A model at {CPT_FUSED}; "
                  f"starting from {BASE_MODEL}.")
    else:
        base = str(CPT_FUSED)
    print(f"base model: {base}")

    train_ds = load_from_disk(str(SFT_DATA))
    eval_ds = load_from_disk(str(EVAL_DATA)) if EVAL_DATA.exists() else None
    print(f"train {len(train_ds)}   eval {len(eval_ds) if eval_ds else 0}")
    print("languages:", dict(sorted(Counter(train_ds["lang"]).items())))

    print(f"\nloading {base} ...")
    model, tokenizer = load_model(base)
    cfg = apply_lora(model, rank=args.rank, scale=SFT["scale"],
                     dropout=SFT["dropout"], keys=SFT["keys"],
                     num_layers=args.num_layers)
    # Stamp the base in, so inference reattaches this adapter to the right weights.
    cfg["base_model"] = base

    if args.resume:
        from lawtune.mlx_utils import load_adapters
        if load_adapters(model, SFT_ADAPTER):
            print(f"resumed adapter weights from {SFT_ADAPTER}")
        else:
            print(f"--resume given but no adapters.safetensors in {SFT_ADAPTER}; "
                  "starting fresh")

    print("tokenising ...")
    examples = encode_split(tokenizer, train_ds, args.seq_len, args.max_examples)
    val = encode_split(tokenizer, eval_ds, args.seq_len, 200) if eval_ds else None
    print(f"{len(examples)} training sequences")

    ids, mask = examples[0]
    first = next(i for i, m in enumerate(mask) if m)
    print("\nfirst example, supervised span begins at token "
          f"{first}/{len(ids)}:\n  ...{tokenizer.decode(ids[max(0, first - 12):first + 40])!r}")

    train(
        model, tokenizer, examples, cfg, SFT_ADAPTER,
        iters=args.iters, batch_size=args.batch_size, grad_accum=args.grad_accum,
        learning_rate=args.lr, warmup=SFT["warmup"],
        weight_decay=SFT["weight_decay"], save_every=SFT["save_every"],
        val_examples=val, seed=args.seed, label="sft",
    )

    print(f"\nadapter -> {SFT_ADAPTER}")
    print("next: python scripts/06_evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
