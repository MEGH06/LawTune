"""A small, explicit MLX training loop.

Written by hand rather than shelling out to `mlx_lm.lora` for one reason: loss masking.
The CLI's masking behaviour varies by version and dataset format, and getting it wrong
is silent -- you get a normal-looking loss curve and a model that restates questions.
Here the mask is constructed in lawtune/mlx_utils.py, asserted before the first step,
and visible in the logs.

Everything else (AdamW, cosine schedule with warmup, gradient accumulation, periodic
checkpointing) is ~120 lines and worth owning.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

from .mlx_utils import batch_iterator, require_mlx, save_adapters, trainable_report

Example = Tuple[List[int], List[int]]      # (token_ids, loss_mask)


def build_schedule(lr: float, total: int, warmup: int):
    """Linear warmup into cosine decay."""
    _, _, optim = require_mlx()
    warmup = max(1, min(warmup, max(total - 1, 1)))
    return optim.join_schedules(
        [optim.linear_schedule(0.0, lr, warmup),
         optim.cosine_decay(lr, max(total - warmup, 1))],
        [warmup],
    )


def make_loss_fn():
    mx, nn, _ = require_mlx()

    def loss_fn(model, inputs, targets, mask):
        # Cast to float32 for the softmax: Gemma-2's logit softcapping in bf16 loses
        # enough precision at 256k vocab to visibly bias the loss.
        logits = model(inputs).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, targets, reduction="none")
        n = mask.sum()
        return (ce * mask).sum() / mx.maximum(n, 1), n

    return loss_fn


def evaluate_loss(model, examples: Sequence[Example], batch_size: int, pad_id: int) -> float:
    mx, _, _ = require_mlx()
    loss_fn = make_loss_fn()
    total, tokens = 0.0, 0
    for ids, mask in batch_iterator(examples, batch_size, pad_id):
        arr = mx.array(ids)
        m = mx.array(mask)[:, 1:]
        loss, n = loss_fn(model, arr[:, :-1], arr[:, 1:], m)
        mx.eval(loss, n)
        total += float(loss) * float(n)
        tokens += float(n)
    return total / tokens if tokens else float("nan")


def train(
    model,
    tokenizer,
    train_examples: List[Example],
    adapter_config: dict,
    out_dir: Path,
    *,
    iters: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    warmup: int,
    weight_decay: float,
    save_every: int,
    val_examples: Sequence[Example] | None = None,
    seed: int = 3407,
    log_every: int = 10,
    label: str = "train",
) -> dict:
    mx, nn, optim = require_mlx()
    from mlx.utils import tree_map

    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0

    trainable, total = trainable_report(model)
    print(f"trainable {trainable:,} / {total:,}  ({100 * trainable / total:.3f}% "
          f"-- {100 - 100 * trainable / total:.2f}% fewer than full fine-tuning)")

    supervised = sum(sum(m) for _, m in train_examples[:64])
    tokens = sum(len(m) for _, m in train_examples[:64])
    pct = 100 * supervised / max(tokens, 1)
    print(f"loss mask: {supervised}/{tokens} tokens supervised over the first 64 "
          f"examples ({pct:.1f}%)")
    if supervised == 0:
        raise SystemExit("Every token is masked -- nothing would be learned.")
    if label == "sft" and pct > 95:
        print("!! almost nothing is masked. For chat data the prompt should be masked; "
              "check encode_chat_example() against this tokenizer's chat template.")

    schedule = build_schedule(learning_rate, iters, warmup)
    opt = optim.AdamW(learning_rate=schedule, weight_decay=weight_decay)
    loss_fn = make_loss_fn()
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    rng = random.Random(seed)
    order = list(range(len(train_examples)))
    rng.shuffle(order)
    cursor = 0

    def next_batch():
        nonlocal cursor, order
        picked = []
        while len(picked) < batch_size:
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            picked.append(train_examples[order[cursor]])
            cursor += 1
        return next(batch_iterator(picked, batch_size, pad_id))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    running, running_tokens, t0 = 0.0, 0.0, time.time()
    start = time.time()

    for step in range(1, iters + 1):
        accum_grads = None
        step_loss, step_tokens = 0.0, 0.0

        for _ in range(grad_accum):
            ids, mask = next_batch()
            arr = mx.array(ids)
            m = mx.array(mask)[:, 1:]
            (loss, n), grads = loss_and_grad(model, arr[:, :-1], arr[:, 1:], m)
            accum_grads = grads if accum_grads is None else tree_map(
                lambda a, b: a + b, accum_grads, grads)
            step_loss += float(loss) * float(n)
            step_tokens += float(n)

        accum_grads = tree_map(lambda g: g / grad_accum, accum_grads)
        opt.update(model, accum_grads)
        mx.eval(model.parameters(), opt.state)

        running += step_loss
        running_tokens += step_tokens

        if step % log_every == 0 or step == 1:
            avg = running / max(running_tokens, 1)
            elapsed = time.time() - t0
            eta = (iters - step) * (elapsed / log_every) / 60 if step > 1 else float("nan")
            lr_now = float(schedule(opt.step)) if callable(schedule) else learning_rate
            print(f"  step {step:>5}/{iters}  loss {avg:.4f}  ppl {math.exp(min(avg, 20)):>8.2f}  "
                  f"lr {lr_now:.2e}  {step_tokens / max(elapsed / log_every, 1e-9):>6.0f} tok/s  "
                  f"ETA {eta:.0f}m")
            history.append({"step": step, "loss": avg})
            running, running_tokens, t0 = 0.0, 0.0, time.time()

        if step % save_every == 0 or step == iters:
            save_adapters(model, out_dir, adapter_config)
            print(f"  checkpoint -> {out_dir} (step {step})")

    result = {"iters": iters, "minutes": round((time.time() - start) / 60, 1),
              "history": history}

    if val_examples:
        val = evaluate_loss(model, list(val_examples)[:200], batch_size, pad_id)
        result["val_loss"] = val
        result["val_perplexity"] = math.exp(min(val, 20))
        print(f"\nheld-out loss {val:.4f}  (perplexity {result['val_perplexity']:.2f})")

    save_adapters(model, out_dir, adapter_config)
    (out_dir / "train_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{label} finished in {result['minutes']} min -> {out_dir}")
    return result
