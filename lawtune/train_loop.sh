#!/usr/bin/env bash
# Train through the Metal watchdog crashes: resume from the last checkpoint each time.
# Checkpoints land every 50 steps, so a crash costs <=50 steps.
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

for attempt in $(seq 1 80); do
  echo "=================== attempt $attempt  $(date '+%H:%M:%S') ==================="
  python scripts/05_train_sft.py --iters 2500 --batch-size 4 --grad-accum 2 \
      --seq-len 512 --num-layers 12 --resume && {
    echo "=== TRAINING COMPLETE on attempt $attempt ==="
    python scripts/06_evaluate.py && python scripts/07_export.py
    exit 0
  }
  echo "--- crashed; resuming in 15s ---"
  sleep 15
done
echo "=== gave up after 80 attempts ==="
exit 1
