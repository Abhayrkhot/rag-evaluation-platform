#!/usr/bin/env bash
# Quality of answers served under load: test-suite questions answered during the 50 QPS runs,
# judged with the same TRUE (citations) and gpt-oss-120b (claims, correctness) judges, by route.
#   scripts/m4_quality.sh <run dir, e.g. results/m4/main_bf16>
set -euo pipefail
source slurm/serve.sh
R=$1; Q=$R/quality; mkdir -p "$Q"
cli() { in_image python3 -m rageval.cli "$@"; }
: > "$Q/records.jsonl"
for d in "$R"/q50_*_s*; do
  tag=$(basename "$d" | sed -E 's/_s[0-9]+$//')   # q50_static / q50_adaptive (seeds pooled)
  cli loadtest to-records --rows "$d/load.jsonl" --suite data/hotpot100k/test.jsonl --out "$Q/$(basename "$d").jsonl" --tag "$tag" > /dev/null
  cat "$Q/$(basename "$d").jsonl" >> "$Q/records.jsonl"
done
cli gen judge-nli --records "$Q/records.jsonl" --out "$Q/nli.json"
start_vllm openai/gpt-oss-120b 8010 0 --max-model-len 16384 --gpu-memory-utilization 0.85
cli gen judge-llm --records "$Q/records.jsonl" --out "$Q/llm.json" --llm-url http://127.0.0.1:8010
stop_vllms
cli gen merge --records "$Q/records.jsonl" --nli "$Q/nli.json" --llm "$Q/llm.json" --out "$Q/judged.jsonl"
cli gen report --judged "$Q/judged.jsonl" --out "$Q/report.json"
