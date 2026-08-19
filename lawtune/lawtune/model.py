"""Inference: load the model, build the prompt, generate, score.

The prompt construction here has to match what 03_build_datasets.py trained on, byte
for byte. That is why there is exactly one copy of it, and why chat.py and
06_evaluate.py both go through this module rather than rolling their own.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence

from .config import BASE_MODEL, MAX_SEQ_LEN, SFT_ADAPTER, SYSTEM_PROMPT


def build_messages(question: str, history: Optional[Sequence[dict]] = None) -> List[dict]:
    """Gemma-2 has no system role, so the persona rides on the first user turn.

    03_build_datasets.py builds the identical string. Change one, change both.
    """
    turns = list(history or []) + [{"role": "user", "content": question}]
    out = []
    for i, turn in enumerate(turns):
        content = turn["content"]
        if i == 0 and turn["role"] == "user":
            content = SYSTEM_PROMPT + "\n\n---\n\n" + content
        out.append({"role": turn["role"], "content": content})
    return out


def default_model_path() -> str:
    """The trained adapter if it exists, else the untuned base."""
    return str(SFT_ADAPTER) if SFT_ADAPTER.exists() else BASE_MODEL


class LawTuneModel:
    """A loaded model (base, fused, or base+adapter) ready to answer questions."""

    def __init__(self, model_path: str = "", seq_len: int = MAX_SEQ_LEN):
        from .mlx_utils import load_model, require_mlx

        require_mlx()
        self.seq_len = seq_len
        model_path = model_path or default_model_path()
        p = Path(model_path)

        if p.is_dir() and (p / "adapters.safetensors").exists():
            # An adapter directory, not a model: load its base and attach it. The base
            # comes from adapter_config.json -- assuming the stock one would be wrong
            # whenever stage A ran, since that adapter was trained against the fused
            # CPT model, and pairing it with stock weights produces quiet garbage.
            base = BASE_MODEL
            cfg = p / "adapter_config.json"
            if cfg.exists():
                base = json.loads(cfg.read_text(encoding="utf-8")).get("base_model") or base
            if base != BASE_MODEL:
                print(f"[mlx] adapter was trained on {base}")
            self.model_id = f"{base} + {p.name}"
            self.model, self.tokenizer = load_model(base, adapter_path=str(p))
        else:
            self.model_id = model_path
            self.model, self.tokenizer = load_model(model_path)

    # ---------------------------------------------------------------- prompting
    def render(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def ask(self, question: str, history=None, max_new_tokens: int = 512,
            temperature: float = 0.0, stream: bool = False) -> str:
        from .mlx_utils import generate_text

        prompt = self.render(build_messages(question, history))
        return generate_text(self.model, self.tokenizer, prompt,
                             max_tokens=max_new_tokens, temperature=temperature,
                             verbose=stream).strip()

    # ---------------------------------------------------------------- scoring
    def token_loss(self, text: str) -> tuple[float, int]:
        """Summed NLL and token count for a fully-supervised string."""
        import mlx.core as mx
        import mlx.nn as nn

        ids = self.tokenizer.encode(text)
        ids = list(ids.tolist() if hasattr(ids, "tolist") else ids)[: self.seq_len]
        if len(ids) < 8:
            return 0.0, 0
        arr = mx.array([ids])
        logits = self.model(arr[:, :-1]).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, arr[:, 1:], reduction="sum")
        mx.eval(ce)
        return float(ce), len(ids) - 1

    def perplexity(self, texts: Sequence[str]) -> float:
        total, tokens = 0.0, 0
        for t in texts:
            nll, n = self.token_loss(t)
            total += nll
            tokens += n
        return math.exp(min(total / tokens, 20)) if tokens else float("nan")
