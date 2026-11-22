#!/usr/bin/env bash
# Make results/ committable: gzip raw JSONL over 1 MB (generation records, judged rows, load and
# trace logs); span logs and warm-up logs stay out of git (.gitignore). Summaries stay plain JSON.
# Idempotent: rerun after every `scripts/explorer.sh pull`.
set -euo pipefail
cd "$(dirname "$0")/../results"
find . -type f \( -name 'spans*.jsonl' -o -name 'warm.jsonl' -o -name 'warm.summary.json' \) -delete
# superseded M4 experiments (documented in docs/RESULTS_LOG.md) keep summaries only; the final run
# (explorer/m4/main_fp8) keeps its per-request logs
if [ -d explorer/m4 ]; then
  find explorer/m4 -mindepth 1 -maxdepth 1 -type d ! -name main_fp8 -print0 | while IFS= read -r -d '' d; do
    find "$d" -type f \( -name 'load.jsonl*' -o -name 'trace*.jsonl*' \) -delete
  done
fi
find . -type f -name '*.jsonl' -size +1M -print0 | while IFS= read -r -d '' f; do
  gzip -9 -f "$f"
done
find . -type f -name '*.json' -size +5M -print0 | while IFS= read -r -d '' f; do
  gzip -9 -f "$f"
done
du -sh . | cut -f1
