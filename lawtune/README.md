# LawTune

Gemma-2-2B, LoRA fine-tuned on 4-bit weights for Indian law across **English + the 22
scheduled languages**, with guardrails designed to make abstention the default rather
than an afterthought.

Runs entirely on **Apple Silicon**.

---

## Quickstart

```bash
cd lawtune
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login                    # Sangraha needs this

# drop your Kaggle law JSONs into data/raw/, then:
SMOKE=1 bash run_all.sh                  # ~25 min, proves the pipeline end to end
bash run_all.sh                          # the real run
python scripts/chat.py
```

Requires Apple Silicon (M1 or later) and macOS 13.5+. An Intel Mac cannot run this —
MLX is Metal-only. 16 GB unified memory is comfortable; 8 GB works with
`--num-layers 12 --seq-len 512`.

### How QLoRA works here

`bitsandbytes` — the usual QLoRA backend — is CUDA-only and has no Metal build. The
Apple equivalent is **MLX**: the base loads from
[`mlx-community/gemma-2-2b-it-4bit`](https://huggingface.co/mlx-community/gemma-2-2b-it-4bit)
(~1.5 GB, already 4-bit), stays frozen and quantized, and only LoRA adapters train in
float. Same technique, Apple's kernels. At the default rank 16 that is **20.8M trainable
parameters — 0.79% of the model**, a 42 MB adapter.

### Timings

**Measure yours before planning around any of this.** The training loop prints tok/s and
a live ETA from step 10:

```bash
python scripts/05_train_sft.py --iters 20      # ~2 min, read the tok/s and ETA
```

The estimates below are arithmetic, not measurements. At the defaults, stage B does
6000 iters × 8 grad-accum = **48,000 sequences ≈ 22M tokens**; stage A does 24,000
sequences ≈ 22M tokens. Divide by your measured throughput:

| Chip | ~tok/s (2B, 4-bit, LoRA) | Stage B at defaults |
|---|---|---|
| M1 / M2 (8-10 GPU cores) | ~250 | **~25 h** |
| M1 / M2 / M3 Pro | ~550 | ~11 h |
| M3 / M4 Max | ~1100 | ~6 h |
| M2 / M3 Ultra | ~1800 | ~3.5 h |

Non-training stages: sampling ~20 min (network-bound), translating 22 languages ~3-7 h
depending on chip.

**On a base M1/M2 the defaults are an overnight-plus job.** Cut it with any of:

| Change | Effect on time | Cost |
|---|---|---|
| `--batch-size 4 --grad-accum 2` | **~1.5-2× faster** | none — same effective batch, better GPU utilisation |
| `--iters 2500` | 2.4× faster | roughly half an epoch; usually still coherent |
| `--seq-len 512` | ~1.4× faster | truncates the longest ~20% of answers |
| `--num-layers 12` | ~1.25× faster | LoRA on the last 12 of 26 blocks; slightly weaker |
| `--max-english 8000` (step 03) | shrinks the corpus | less English, unchanged multilingual coverage |

Stacking the first two gets a base M1 to roughly 5-6 h. Start there, read the eval
numbers, and spend more time only where they are weak.

Skip stage A on the first pass — `run_all.sh` does. Everything checkpoints every 250
steps and every stage resumes, so Ctrl-C is safe and a long run can be split across
several sittings.

Back up `data/interim/translated/` before anything else. It is the most expensive
artefact and the only one you cannot cheaply regenerate.

### Which machine

Train on Apple Silicon. There is no second option in this repo, and the alternatives are
worse anyway:

| Machine | Path | Rough throughput |
|---|---|---|
| M1 Pro / M2 Pro | MLX (this repo) | ~550 tok/s |
| M1 / M2 base | MLX (this repo) | ~250 tok/s |
| Intel Arc iGPU (Core Ultra) | IPEX-LLM — not supported here | ~100-150 tok/s, fragile stack |
| Any laptop CPU | very slow | ~20-40 tok/s |

An Intel Arc integrated GPU has no CUDA and no Metal. Training on it needs Intel's
IPEX-LLM stack (oneAPI toolkit, pinned torch build, mostly documented for discrete Arc on
Linux) and would be a third backend to write and maintain — for something still ~4x
slower than an M1 Pro. Use the Mac.

---

## Keeping the repo light for GitHub

The tracked tree is **~320 KB of code**. Nothing heavy is committed, and `.gitignore`
enforces it: `data/`, `outputs/`, `*.gguf`, adapters, and `kaggle.json` are all excluded.
Verify before a push:

```bash
git add -An .                        # exactly what would be committed
git check-ignore -v outputs/lawtune_adapter/adapters.safetensors   # should print a rule
```

**Do not commit weights.** GitHub warns at 50 MB per file and hard-blocks at 100 MB.

| Artefact | Size | Where it goes |
|---|---|---|
| LoRA adapter, rank 16 (default) | 42 MB | Hugging Face Hub |
| LoRA adapter, rank 32 | 83 MB | Hub — over GitHub's warning threshold |
| Fused 4-bit MLX model | ~1.6 GB | Hub only |
| GGUF q4_k_m | ~1.7 GB | Hub only |

Push weights to the Hub and keep the git repo as code:

```bash
python scripts/07_export.py --push YOUR-HF-USERNAME/lawtune-gemma2-2b-mlx
```

If you really want the adapter in the repo, rank 16 at 42 MB will go in without Git LFS
— but the Hub is the right home for it, and it keeps `git clone` fast.

---

## Pipeline

```
00_fetch_kaggle.py     your law dataset  -> data/raw/
01_sample_sangraha.py  200 MB × 23 langs -> data/interim/sangraha_sample/
02_translate.py        law+guards × 22   -> data/interim/translated/*.jsonl  [the slow one]
03_build_datasets.py   everything        -> data/processed/{cpt,sft,eval}/
04_train_cpt.py        stage A, optional -> outputs/cpt_fused/
05_train_sft.py        stage B           -> outputs/lawtune_adapter/
06_evaluate.py         the scorecard     -> outputs/eval_report.json
07_export.py           fused MLX + GGUF  -> outputs/
chat.py                guarded REPL
```

---

## What this fixes in the original notebook

| Original | Problem | Here |
|---|---|---|
| `load_dataset("ai4bharat/sangraha", name="verified")` | Resolves ~1.5 TB. `.shard()` only helps *after* the download. | `01_sample_sangraha.py` streams parquet row-groups over HTTP and stops at a byte budget. Nothing hits disk but the sample. |
| `x["instruction"]` on Sangraha rows | Sangraha has only a `text` column, so **every language fell into the `except` block**. The "multilingual" run trained on the English law JSONs alone. | Sangraha is treated as what it is: a pretraining corpus, in its own stage. |
| Raw web text in an Alpaca template | Teaches the model that a legal question deserves a paragraph of scraped Odia news. | Two objectives, two stages: CPT on raw text, SFT on chat data. |
| Loss over the whole Alpaca string | The model spends its capacity learning to reproduce the prompt → restates the question, then invents. | Loss on the assistant turn only, asserted before the first step. |
| `max_steps = 60` | ~1000 examples seen. Nothing is learned. | Iteration-based, checkpointed, resumable. |
| Falcon tokenizer used for Gemma's EOS (cell 8) | Wrong EOS. The model never learns to stop. | Gemma-2 chat template throughout, `<end_of_turn>` terminated. |
| No evaluation | No way to know it works. | `06_evaluate.py` scores language fidelity, abstention, and impossible citations. |

---

## Why two stages

**Stage A — continued pretraining** on the 200 MB multilingual sample. Loads scripts and
vocabulary for languages the model barely knows. `--train-embeddings` extends LoRA to the
embedding matrix, which is what actually helps an unseen script; it is opt-in because not
every mlx-lm version can fuse a LoRA'd embedding back into the base.

**Stage B — instruction tuning** on chat-formatted law QA in 23 languages, loss on the
assistant turn only.

Stage A is fused into the weights and stage B starts from the fuse, rather than stacking
adapters. Each adapter records its true base in `adapter_config.json`, so inference
always reattaches to the right weights.

Stage A is optional because Gemma-2's tokenizer is a 256k multilingual SentencePiece that
already covers every Indic script — stage B alone gets you a long way. Run stage B first
and decide from the evaluation numbers.

MLX has no separate embedding learning rate, so stage A trains everything at one
conservative rate (3e-5) instead of the usual 10×-lower-for-embeddings split.

---

## The 200 MB sample

Split **evenly across the 23 languages** (~8.7 MB each), not proportionally to what
Sangraha holds. Proportional sampling gives you a model that speaks Hindi and pretends
the other 21 don't exist.

Each document passes a length check, a **script-ratio check** (a "Tamil" document that is
70% English is a scrape artefact, not Tamil data), a boilerplate detector, and a
normalised-hash dedupe.

```bash
python scripts/01_sample_sangraha.py --total-mb 200
python scripts/01_sample_sangraha.py --total-mb 20 --langs hin,tam,ben   # quick check
```

Per-language yield lands in `data/interim/sangraha_sample_stats.json`.

---

## The part that actually makes it multilingual

Sangraha teaches the model what Odia *looks like*. It does not teach it to answer a bail
question in Odia. No parallel Indian-legal instruction corpus exists, so
`02_translate.py` builds one with **IndicTrans2**: your English law QA plus the entire
guardrail set, fanned out to all 22 languages, sentence by sentence, with a script-ratio
gate that drops any "translation" that came back as copied English.

Runs on **MPS** in float32 — Metal's fp16 path NaNs in this model's encoder attention,
and one NaN silently poisons a whole language's training data. Defaults to the distilled
200M translator; `--model ai4bharat/indictrans2-en-indic-1B` is better and much slower.

---

## Guardrails

Three layers, because a 2B model needs all three.

**1. Training data** (`lawtune/guard_data.py`) — six buckets, upsampled, translated into
every language: out-of-domain, hallucination bait (fake sections and fake case names),
unsafe requests, genuine uncertainty, the advice/information boundary, and foreign
jurisdictions. A model abstains because abstention is the high-probability continuation
for a *shape* of question it has seen abstained on. So we manufacture that shape.

**2. Runtime validation** (`lawtune/guardrails.py`) — the piece that earns its keep is the
citation validator. Every Indian statute has a known highest provision number:

```python
>>> check_output("Under Section 812 of the Indian Penal Code, theft is punishable.")
Verdict(allowed=False, flags=['bad_citation:Section 812'], ...)
#   "Indian Penal Code, 1860 has no provision numbered 812 (max 511)"
```

Arithmetic, not hope, and no retrieval index required. Plus unsafe-intent screening on
input, foreign-jurisdiction detection, a language-match check, and a disclaimer.

**3. Retrieval** — the real fix for factual grounding, and the next piece of the project
(GraphRAG over Supreme Court judgments in Neo4j). A 2B model should be *reading* the
statute, not recalling it. Everything here is built to sit under that: the abstention
behaviour is what makes a RAG system say "not in the retrieved context" instead of
filling the gap.

### Honest limits

Nothing here gets you to zero hallucination, and you should not claim it does.
Gemma-2-2B has ~2.6 B parameters holding every language and domain it knows; the fraction
allocated to the exact text of Section 138 of the NI Act is small. What this pipeline
buys you: it abstains instead of inventing far more often, and the citation validator
catches a large share of what slips through. **Verified grounding requires retrieval.**

---

## Evaluate before you believe it

```bash
python scripts/06_evaluate.py
python scripts/06_evaluate.py --model mlx-community/gemma-2-2b-it-4bit \
    --out outputs/eval_baseline.json
```

Three numbers, and the delta against the untuned baseline is your evidence:

- **Language fidelity** — a real legal question in each of the 23 languages, checking the
  reply's dominant Unicode script. The number most multilingual projects never measure,
  and the one that catches "it answers everything in Hindi".
- **Abstention** — fake-section, out-of-domain, and unsafe probes. Scored by similarity to
  the refusal text the model was *actually trained on*, so a Tamil refusal counts.
- **Impossible-citation rate** — of every provision emitted, how many cannot exist.

---

## Deploy

```bash
python scripts/07_export.py            # fused 4-bit MLX, ~1.6 GB
python scripts/07_export.py --gguf     # + GGUF q4_k_m + an Ollama Modelfile
```

- **Fused MLX** — fastest on this Mac. Ship it in a Mac app, or serve with
  `mlx_lm.server`.
- **GGUF q4_k_m** (~1.7 GB) — what llama.cpp, Ollama, and mobile builds load.

One correction on the deployment plan: **TensorRT is NVIDIA-only.** It covers a Jetson or
a GPU server; it does not run on an Android or iOS handset. For genuine on-device
inference the path is **llama.cpp** (Android NDK / iOS) or **MLC-LLM**, both of which
consume this GGUF. Quote TensorRT numbers for your server tier and llama.cpp numbers for
mobile — don't merge them into one figure.

---

## Layout

```
lawtune/
├── lawtune/
│   ├── config.py       every knob; edit this, not the scripts
│   ├── model.py        load / prompt / generate / score
│   ├── mlx_utils.py    MLX load, LoRA, adapters, chat-example masking
│   ├── mlx_train.py    the training loop
│   ├── law_data.py     key-tolerant Kaggle JSON loader
│   ├── guard_data.py   synthetic abstention/refusal corpus
│   ├── guardrails.py   runtime input + output validation
│   └── textutil.py     script detection, dedupe, junk filter
├── scripts/            00 … 07 + chat.py
└── run_all.sh
```

### Knobs worth knowing

| Flag | Where | Effect |
|---|---|---|
| `--total-mb` | 01 | size of the Sangraha sample |
| `--pairs-per-lang` | 02 | translated law pairs per language; the main cost driver |
| `--device` | 02 | force `mps` or `cpu` |
| `--max-english` | 03 | caps English so it does not swamp the other 22 (default 25000) |
| `--guard-per-bucket` | 03 | more abstention data → fewer fabrications |
| `--balance` | 03 | upsamples thin languages toward the median |
| `--iters` | 04/05 | the wall-clock knob; 200 for a smoke test |
| `--num-layers` | 04/05 | LoRA only the last N blocks — faster, lighter, slightly weaker |
| `--seq-len` | 04/05 | drop to 512 on an 8 GB Mac |
| `--train-embeddings` | 04 | better unseen scripts, may block fusing |
| `--no-cpt` | 05 | train straight from the stock base |
