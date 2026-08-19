"""MLX helpers: load, LoRA-ify, save/load adapters, generate.

mlx-lm moves fast and renames things between minor versions (`temp` -> `sampler`,
`lora_layers` -> `num_layers`, `mlx_lm.tuner` -> `mlx_lm.tuner.utils`). Everything
version-sensitive is isolated here and resolved by introspection, so the training
scripts stay readable and do not break on `pip install -U mlx-lm`.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


# ------------------------------------------------------------------ imports
def require_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError as exc:                                    # noqa: BLE001
        raise SystemExit(
            "MLX is not installed. On Apple Silicon:\n"
            "    pip install -r requirements.txt\n"
            f"({type(exc).__name__}: {exc})"
        ) from exc
    return mx, nn, optim


def _lora_fn():
    """linear_to_lora_layers moved between mlx_lm.tuner and mlx_lm.tuner.utils."""
    for mod in ("mlx_lm.tuner.utils", "mlx_lm.tuner", "mlx_lm.utils"):
        try:
            m = __import__(mod, fromlist=["linear_to_lora_layers"])
            if hasattr(m, "linear_to_lora_layers"):
                return m.linear_to_lora_layers
        except ImportError:
            continue
    raise SystemExit("mlx_lm.tuner.linear_to_lora_layers not found -- "
                     "pip install -U mlx-lm")


def load_model(model_id: str, adapter_path: Optional[str] = None):
    """Load an MLX model (+ optional trained adapter). Returns (model, tokenizer)."""
    from mlx_lm import load

    kwargs: dict[str, Any] = {}
    if adapter_path:
        params = inspect.signature(load).parameters
        key = "adapter_path" if "adapter_path" in params else "adapter_file"
        kwargs[key] = str(adapter_path)
    return load(model_id, **kwargs)


def transformer_layers(model) -> list:
    """The list of decoder blocks, wherever this mlx-lm version keeps it."""
    for attr in ("layers", "model"):
        obj = getattr(model, attr, None)
        if isinstance(obj, list):
            return obj
        if obj is not None and isinstance(getattr(obj, "layers", None), list):
            return obj.layers
    raise AttributeError("could not locate the transformer layer list on this model")


# ------------------------------------------------------------------ LoRA
def apply_lora(model, *, rank: int, scale: float, dropout: float,
               keys: Sequence[str], num_layers: Optional[int] = None) -> dict:
    """Freeze the base and attach LoRA adapters. Returns the config to persist."""
    mx, nn, _ = require_mlx()
    linear_to_lora_layers = _lora_fn()

    n_layers = num_layers if num_layers is not None else len(transformer_layers(model))
    # Superset dict: different versions read different keys out of it, and the extras
    # are ignored rather than rejected.
    lora_cfg = {"rank": rank, "scale": scale, "dropout": dropout,
                "alpha": scale, "keys": list(keys)}

    model.freeze()
    sig = inspect.signature(linear_to_lora_layers).parameters
    try:
        if "config" in sig:
            linear_to_lora_layers(model, n_layers, lora_cfg)
        else:                                        # very old positional form
            linear_to_lora_layers(model, n_layers, rank)
    except (KeyError, TypeError) as exc:
        print(f"[mlx] retrying LoRA attach without 'keys' ({type(exc).__name__}: {exc})")
        lora_cfg.pop("keys")
        linear_to_lora_layers(model, n_layers, lora_cfg)

    return {"fine_tune_type": "lora", "num_layers": n_layers,
            "lora_layers": n_layers,                 # older mlx-lm reads this name
            "lora_parameters": lora_cfg}


def trainable_report(model) -> tuple[int, int]:
    from mlx.utils import tree_flatten

    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    return trainable, total


def save_adapters(model, out_dir: Path, adapter_config: dict) -> None:
    """Write adapters.safetensors + adapter_config.json in mlx-lm's expected layout."""
    mx, _, _ = require_mlx()
    from mlx.utils import tree_flatten

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), weights)
    (out_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ generation
def make_sampler(temperature: float, top_p: float = 0.9):
    """Return whatever the installed mlx-lm wants for sampling, or None."""
    try:
        from mlx_lm.sample_utils import make_sampler as _mk
    except ImportError:
        return None
    if temperature <= 0:
        return _mk(temp=0.0)
    params = inspect.signature(_mk).parameters
    kw = {"temp": temperature}
    if "top_p" in params:
        kw["top_p"] = top_p
    return _mk(**kw)


def generate_text(model, tokenizer, prompt: str, *, max_tokens: int = 512,
                  temperature: float = 0.0, top_p: float = 0.9,
                  verbose: bool = False) -> str:
    """mlx_lm.generate across versions: newer takes sampler=, older takes temp=."""
    from mlx_lm import generate

    params = inspect.signature(generate).parameters
    kw: dict[str, Any] = {"max_tokens": max_tokens, "verbose": verbose}
    if "sampler" in params:
        sampler = make_sampler(temperature, top_p)
        if sampler is not None:
            kw["sampler"] = sampler
    elif "temp" in params:
        kw["temp"] = temperature
        if "top_p" in params:
            kw["top_p"] = top_p

    out = generate(model, tokenizer, prompt=prompt, **kw)
    return out.text if hasattr(out, "text") else str(out)


# ------------------------------------------------------------------ tokenisation
def end_of_turn_id(tokenizer) -> int:
    """Gemma-2 ends an assistant turn with <end_of_turn>, not <eos>."""
    for tok in ("<end_of_turn>", "<eos>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:                                        # noqa: BLE001
            tid = None
        if isinstance(tid, int) and tid >= 0:
            return tid
    return tokenizer.eos_token_id


def encode_chat_example(tokenizer, messages: List[dict], answer: str,
                        max_len: int) -> tuple[list[int], list[int]]:
    """Return (token_ids, loss_mask).

    The mask is 0 over the prompt and 1 over the assistant turn. This is the whole
    ballgame: supervise the prompt too and a 2B model learns to restate the question
    instead of answering it.
    """
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    prompt_ids = list(prompt_ids)

    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_ids = list(answer_ids) + [end_of_turn_id(tokenizer)]

    ids = prompt_ids + answer_ids
    mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

    if len(ids) > max_len:
        # Truncate from the FRONT of the prompt, never the answer -- a clipped answer
        # teaches the model to stop mid-sentence.
        overflow = len(ids) - max_len
        if overflow < len(prompt_ids) - 8:
            ids, mask = ids[overflow:], mask[overflow:]
        else:
            ids, mask = ids[:max_len], mask[:max_len]
    return ids, mask


def encode_plain_text(tokenizer, text: str, max_len: int) -> tuple[list[int], list[int]]:
    """Continued-pretraining example: every token supervised."""
    ids = tokenizer.encode(text)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    ids = list(ids)[:max_len]
    return ids, [1] * len(ids)


def batch_iterator(examples: Iterable[tuple[list[int], list[int]]], batch_size: int,
                   pad_id: int = 0):
    """Yield (inputs, mask) numpy-shaped lists, right-padded to the batch max."""
    batch: list[tuple[list[int], list[int]]] = []
    for ex in examples:
        batch.append(ex)
        if len(batch) == batch_size:
            yield _pad(batch, pad_id)
            batch = []
    if batch:
        yield _pad(batch, pad_id)


def _pad(batch, pad_id: int):
    width = max(len(ids) for ids, _ in batch)
    ids_out, mask_out = [], []
    for ids, mask in batch:
        pad = width - len(ids)
        ids_out.append(ids + [pad_id] * pad)
        mask_out.append(mask + [0] * pad)
    return ids_out, mask_out
