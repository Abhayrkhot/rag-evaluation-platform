#!/usr/bin/env bash
# Make results/ committable: gzip raw JSONL over 1 MB (generation records, judged rows, load and
# trace logs); span logs and warm-up logs stay out of git (.gitignore). Summaries stay plain JSON.
# Idempotent: rerun after every `scripts/explorer.sh pull`.
set -euo pipefail
cd "$(dirname "$0")/../results"
find . -type f \( -name 'spans*.jsonl' -o -name 'warm.jsonl' -o -name 'warm.summary.json' \) -delete
find . -type f -name '*.jsonl' -size +1M -print0 | while IFS= read -r -d '' f; do
  gzip -9 -f "$f"
done
find . -type f -name '*.json' -size +5M -print0 | while IFS= read -r -d '' f; do
  gzip -9 -f "$f"
done
du -sh . | cut -f1
