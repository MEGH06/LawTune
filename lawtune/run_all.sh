#!/usr/bin/env bash
# End-to-end LawTune pipeline on Apple Silicon. Run from the lawtune/ directory.
#
#   bash run_all.sh           # hours; see README timings, and measure yours first
#   SMOKE=1 bash run_all.sh   # ~25 min end-to-end sanity check
#   CPT=1 bash run_all.sh     # also run the optional stage A (adds 2-4 h)
#
# Everything is resumable. Ctrl-C and re-run the same command: the sampler and the
# translator skip completed work, and the trainers checkpoint every 250 steps.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "LawTune trains on MLX, which requires Apple Silicon (M1 or later)." >&2
  exit 1
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  TOTAL_MB=6; PAIRS=25; SFT_ITERS=60; CPT_ITERS=40; GUARD=30; EVAL_PER_LANG=4
  echo "### SMOKE MODE -- tiny run, proves the pipeline, produces a bad model ###"
else
  TOTAL_MB=200; PAIRS=600; SFT_ITERS=6000; CPT_ITERS=3000; GUARD=250; EVAL_PER_LANG=40
fi

step () { echo; echo "=============== $* ==============="; }

step "0/7  inspect the Kaggle law corpus"
python scripts/00_fetch_kaggle.py --inspect-only

step "1/7  sample ${TOTAL_MB} MB of Sangraha across 23 languages"
python scripts/01_sample_sangraha.py --total-mb "${TOTAL_MB}"

step "2/7  translate law + guardrails into 22 languages (resumable, the slow one)"
python scripts/02_translate.py --pairs-per-lang "${PAIRS}"

step "3/7  build CPT + SFT corpora"
python scripts/03_build_datasets.py \
  --guard-per-bucket "${GUARD}" --eval-per-lang "${EVAL_PER_LANG}"

if [[ "${CPT:-0}" == "1" ]]; then
  step "4/7  stage A: continued pretraining (MLX LoRA on 4-bit)"
  python scripts/04_train_cpt.py --iters "${CPT_ITERS}"
else
  step "4/7  stage A SKIPPED (set CPT=1 to enable; adds 2-4 h)"
fi

step "5/7  stage B: instruction tuning (MLX LoRA on 4-bit)"
python scripts/05_train_sft.py --iters "${SFT_ITERS}"

step "6/7  evaluate"
python scripts/06_evaluate.py

step "7/7  fuse for deployment"
python scripts/07_export.py

echo
echo "Done. Try it:  python scripts/chat.py"
