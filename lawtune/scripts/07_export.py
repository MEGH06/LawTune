#!/usr/bin/env python
"""Export the Mac-trained model: fused MLX weights, and optionally GGUF for llama.cpp.

Three artefacts, three purposes:
  * fused MLX 4-bit  -- fastest inference on this Mac, ~1.6 GB. Ship this for a Mac app.
  * fused float16    -- the portable checkpoint (--dequantize). Feed it to llama.cpp,
                        Ollama, or an iOS/Android build.
  * GGUF q4_k_m      -- what llama.cpp and Ollama actually load, ~1.7 GB.

    python scripts/07_export.py                     # fused 4-bit MLX
    python scripts/07_export.py --gguf              # + f16 fuse + GGUF q4_k_m
    python scripts/07_export.py --push you/lawtune-gemma2-2b-mlx
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lawtune.config import (  # noqa: E402
    HF_TOKEN, BASE_MODEL, FUSED, SFT_ADAPTER, OUT, SYSTEM_PROMPT,
)

LLAMA_CPP = OUT / "llama.cpp"
MODELFILE = OUT / "Modelfile"


def run(cmd: list[str]) -> bool:
    print("\n$ " + " ".join(str(c) for c in cmd))
    try:
        subprocess.run([str(c) for c in cmd], check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[fail] {type(exc).__name__}: {exc}")
        return False


def adapter_base(adapter: Path) -> str:
    """The model this adapter was trained against -- the stock base, or a fused stage A."""
    cfg = adapter / "adapter_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get("base_model") or BASE_MODEL
    return BASE_MODEL


def fuse(adapter: Path, dest: Path, dequantize: bool, upload: str = "") -> bool:
    cmd = [sys.executable, "-m", "mlx_lm.fuse",
           "--model", adapter_base(adapter), "--adapter-path", str(adapter),
           "--save-path", str(dest)]
    if dequantize:
        cmd.append("--de-quantize")
    if upload:
        cmd += ["--upload-repo", upload]
    return run(cmd)


def build_gguf(f16_dir: Path, quant: str) -> Path | None:
    """Convert the de-quantized fuse to GGUF via llama.cpp, then quantize."""
    if not LLAMA_CPP.exists():
        if not run(["git", "clone", "--depth", "1",
                    "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP)]):
            return None
        run([sys.executable, "-m", "pip", "install", "-q", "-r",
             str(LLAMA_CPP / "requirements.txt")])

    f16_gguf = OUT / "lawtune-f16.gguf"
    if not run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(f16_dir),
                "--outfile", str(f16_gguf), "--outtype", "f16"]):
        return None

    quantize = LLAMA_CPP / "build" / "bin" / "llama-quantize"
    if not quantize.exists():
        # Metal is on by default in recent llama.cpp on macOS; no extra flags needed.
        if not run(["cmake", "-B", str(LLAMA_CPP / "build"), str(LLAMA_CPP)]):
            return None
        if not run(["cmake", "--build", str(LLAMA_CPP / "build"),
                    "--config", "Release", "-j"]):
            return None

    out = OUT / f"lawtune-{quant}.gguf"
    if not run([str(quantize), str(f16_gguf), str(out), quant.upper()]):
        return None
    return out


def write_modelfile(gguf: Path) -> None:
    text = f'''FROM ./{gguf.name}

TEMPLATE """<start_of_turn>user
{{{{ .System }}}}

---

{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
{{{{ .Response }}}}<end_of_turn>
"""

SYSTEM """{SYSTEM_PROMPT}"""

PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 2048
'''
    MODELFILE.write_text(text, encoding="utf-8")
    print(f"\nModelfile -> {MODELFILE}")
    print(f"  cd {OUT} && ollama create lawtune -f Modelfile && ollama run lawtune")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=str(SFT_ADAPTER))
    ap.add_argument("--gguf", action="store_true", help="also build a GGUF for llama.cpp")
    ap.add_argument("--quant", type=str, default="q4_k_m")
    ap.add_argument("--push", type=str, default="", help="HF repo id for the MLX fuse")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    if not (adapter / "adapters.safetensors").exists():
        print(f"No adapter at {adapter}. Train first: scripts/05_train_sft.py",
              file=sys.stderr)
        return 1

    print(f"fusing {adapter.name} into 4-bit MLX weights "
          f"(base: {adapter_base(adapter)}) ...")
    if not fuse(adapter, FUSED, dequantize=False, upload=args.push if HF_TOKEN else ""):
        return 1
    print(f"\nfused MLX model -> {FUSED}")
    print(f"  python -m mlx_lm.generate --model {FUSED} --prompt 'What is Article 21?'")

    if args.gguf:
        f16_dir = OUT / "mlx_lawtune_fused_f16"
        print("\nfusing again at float16 for llama.cpp ...")
        if not fuse(adapter, f16_dir, dequantize=True):
            return 1
        gguf = build_gguf(f16_dir, args.quant)
        if gguf:
            print(f"\nGGUF -> {gguf}  ({gguf.stat().st_size / 1e9:.2f} GB)")
            write_modelfile(gguf)
        else:
            print("\n[warn] GGUF build failed. The float16 fuse at "
                  f"{f16_dir} is still valid -- convert it manually with "
                  "llama.cpp/convert_hf_to_gguf.py.")

    print("\ndone.")
    print("On-device note: this GGUF is the artefact you ship to iOS/Android via "
          "llama.cpp or MLC-LLM. TensorRT is NVIDIA-only and has no role on a handset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
