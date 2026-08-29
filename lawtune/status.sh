#!/usr/bin/env bash
# LawTune pipeline status.
#   bash status.sh [path-to-training-log]
cd "$(dirname "$0")"
LOG="${1:-}"

echo "── STAGE ─────────────────────────────────────"
if   pgrep -f "scripts/02_translate" >/dev/null; then echo "  02 translate       RUNNING"
elif pgrep -f "scripts/03_build"     >/dev/null; then echo "  03 build datasets  RUNNING"
elif pgrep -f "scripts/05_train"     >/dev/null; then echo "  05 train SFT       RUNNING"
elif pgrep -f "scripts/0[67]_"       >/dev/null; then echo "  06/07 eval+export  RUNNING"
else echo "  nothing running"; fi

echo
echo "── TRANSLATION ───────────────────────────────"
n=$(ls data/interim/translated/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "  languages done: ${n}/22"
[ "$n" -gt 0 ] && ls -1 data/interim/translated/*.jsonl | xargs -n1 basename | sed 's/.jsonl//' \
  | tr '\n' ' ' | fold -sw 46 | sed 's/^/  /'

echo
echo "── ARTEFACTS ─────────────────────────────────"
for p in outputs/lawtune_adapter/adapters.safetensors data/processed/sft \
         data/processed/legal_graph.kuzu data/processed/faiss_index; do
  [ -e "$p" ] && printf "  %-42s %s\n" "$p" "$(du -sh "$p" | cut -f1)"
done

echo
echo "── TRAINING ──────────────────────────────────"
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  grep -aE "^  step |checkpoint ->" "$LOG" | tail -3 | sed 's/^/  /'
else
  echo "  (pass a log path to see live steps)"
fi
