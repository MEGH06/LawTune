# LawTune

Gemma-2-2B, LoRA fine-tuned on 4-bit weights for Indian law across English + the 22
scheduled languages, with guardrails built so that abstention is the default rather
than an afterthought. Runs entirely on Apple Silicon via MLX.

**The pipeline lives in [`lawtune/`](lawtune/) — start with its
[README](lawtune/README.md).**

```bash
cd lawtune
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
SMOKE=1 bash run_all.sh      # ~25 min sanity check
bash run_all.sh              # the real run
```

`gemma_2_2b_fine_tuning_law.ipynb` is the original exploratory notebook, kept for
reference. The `lawtune/` README documents what it got wrong and how each issue is fixed.
