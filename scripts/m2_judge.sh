#!/usr/bin/env bash
# Judge (and report) existing generation records without regenerating:  scripts/m2_judge.sh <tag>
set -euo pipefail
source slurm/serve.sh
tag=$1; R=results/m2
cli() { in_image python3 -m rageval.cli "$@"; }
cli gen judge-nli --records "$R/${tag}_gen.jsonl" --out "$R/${tag}_nli.json"
start_vllm openai/gpt-oss-120b 8010 0 --max-model-len 16384 --gpu-memory-utilization 0.85
cli gen judge-llm --records "$R/${tag}_gen.jsonl" --out "$R/${tag}_llm.json" --llm-url http://127.0.0.1:8010
stop_vllms
cli gen merge --records "$R/${tag}_gen.jsonl" --nli "$R/${tag}_nli.json" --llm "$R/${tag}_llm.json" --out "$R/${tag}_judged.jsonl"
PAIRS=()
if [ "$tag" != dev ]; then
  PAIRS=(--pairs zero_shot:few_shot --pairs zero_shot:cot --pairs s0_baseline:s4_verified --pairs zero_shot:s4_verified
         --pairs s1a_rerank:s1b_2hop --pairs s0_baseline:s1a_rerank)
fi
cli gen report --judged "$R/${tag}_judged.jsonl" --out "$R/${tag}_report.json" "${PAIRS[@]}"
if [ "$tag" = dev ]; then cli gen choose --dev-report "$R/dev_report.json" --out "$R/test_choices.json"; fi
