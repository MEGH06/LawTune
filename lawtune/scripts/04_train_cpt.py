#!/usr/bin/env python
"""Stage A on Apple Silicon -- continued pretraining on the multilingual sample.

LoRA on 4-bit quantized weights via MLX. Same idea as QLoRA on CUDA: the base stays
frozen and quantized, only the adapters are trained in float.

This stage is OPTIONAL on a Mac and off by default in run_all.sh. Gemma-2's
tokenizer is a 256k multilingual SentencePiece that already covers every Indic script,
so stage B alone gets you a long way. Stage A buys fluency in the low-resource
languages (Santali, Bodo, Dogri, Manipuri, Konkani) at the cost of a few hours.

    python scripts/04_train_cpt.py                      # 3000 iters; 3 h on a Max, 12 h on an M1
    python scripts/04_train_cpt.py --iters 300          # smoke test
    python scripts/04_train_cpt.py --train-embeddings   # better scripts, riskier fuse
    python scripts/04_train_cpt.py --no-fuse            # keep the adapter only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk  # noqa: E402

from lawtune.config import (  # noqa: E402
    BASE_MODEL, CPT, CPT_ADAPTER, CPT_FUSED, PROCESSED, SEED,
)
from lawtune.mlx_train import train  # noqa: E402
from lawtune.mlx_utils import (  # noqa: E402
    apply_lora, encode_plain_text, load_model, require_mlx,
)

CPT_DATA = PROCESSED / "cpt"


def fuse(base: str, adapter: Path, dest: Path) -> bool:
    """Bake the adapter into the weights so stage B can start from a plain model."""
    cmd = [sys.executable, "-m", "mlx_lm.fuse", "--model", base,
           "--adapter-path", str(adapter), "--save-path", str(dest)]
    print("\n$ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"\n[warn] fuse failed (exit {exc.returncode}).")
        print("[warn] Not fatal: run stage B with --no-cpt, or point it at the adapter\n"
              "       with --base-adapter. If you used --train-embeddings, that is the\n"
              "       likely cause -- some mlx-lm versions cannot fuse a LoRA'd embedding.")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=BASE_MODEL)
    ap.add_argument("--iters", type=int, default=CPT["iters"])
    ap.add_argument("--batch-size", type=int, default=CPT["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=CPT["grad_accum"])
    ap.add_argument("--lr", type=float, default=CPT["learning_rate"])
    ap.add_argument("--rank", type=int, default=CPT["rank"])
    ap.add_argument("--seq-len", type=int, default=CPT["seq_len"])
    ap.add_argument("--num-layers", type=int, default=None,
                    help="LoRA only the last N blocks; fewer = faster and lighter")
    ap.add_argument("--train-embeddings", action="store_true")
    ap.add_argument("--no-fuse", action="store_true")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = use all")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    require_mlx()

    if not CPT_DATA.exists():
        print(f"{CPT_DATA} missing. Run 01_sample_sangraha.py then 03_build_datasets.py.",
              file=sys.stderr)
        return 1

    ds = load_from_disk(str(CPT_DATA))
    if args.max_docs:
        ds = ds.select(range(min(args.max_docs, len(ds))))
    print(f"CPT corpus: {len(ds)} documents")

    print(f"loading {args.model} (4-bit MLX) ...")
    model, tokenizer = load_model(args.model)

    keys = list(CPT["keys"])
    if args.train_embeddings:
        keys += list(CPT["embedding_keys"])
        print("training the embedding matrix too (helps unseen scripts, "
              "may block fusing)")
    cfg = apply_lora(model, rank=args.rank, scale=CPT["scale"],
                     dropout=CPT["dropout"], keys=keys, num_layers=args.num_layers)
    # Stamp the base in, so inference reattaches this adapter to the right weights.
    cfg["base_model"] = args.model

    print("tokenising ...")
    examples = []
    for text in ds["text"]:
        ids, mask = encode_plain_text(tokenizer, text, args.seq_len)
        if len(ids) >= 32:
            examples.append((ids, mask))
    print(f"{len(examples)} training sequences, "
          f"{sum(len(i) for i, _ in examples) / 1e6:.1f}M tokens")

    train(
        model, tokenizer, examples, cfg, CPT_ADAPTER,
        iters=args.iters, batch_size=args.batch_size, grad_accum=args.grad_accum,
        learning_rate=args.lr, warmup=CPT["warmup"],
        weight_decay=CPT["weight_decay"], save_every=CPT["save_every"],
        seed=args.seed, label="cpt",
    )

    if not args.no_fuse:
        if fuse(args.model, CPT_ADAPTER, CPT_FUSED):
            print(f"\nfused model -> {CPT_FUSED}")
            print("stage B will pick this up automatically.")

    print("\nnext: python scripts/05_train_sft.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
