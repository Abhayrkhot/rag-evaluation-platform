#!/usr/bin/env bash
# M2/M3: generate answers for a suite with every pipeline, then judge them.
#   scripts/m2_run.sh <suite.jsonl> <pipelines.yaml> <tag> [extra gen args]
# 2 GPUs: GPU0 serves the generator then the gpt-oss judge; GPU1 runs retrieval encoders and the
# NLI verifier, then the TRUE NLI judge in parallel with the LLM judge. 1 GPU: all sequential.
set -euo pipefail
source slurm/serve.sh
suite=$1; pipes=$2; tag=$3; shift 3
R=results/m2; mkdir -p "$R"
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -gt 1 ]; then GEN=0; AUX=1; UTIL=0.9; else GEN=0; AUX=0; UTIL=0.55; fi
cli() { in_image python3 -m rageval.cli "$@"; }

start_vllm Qwen/Qwen3-8B 8000 "$GEN" --max-model-len 8192 --gpu-memory-utilization "$UTIL" --seed 0
CUDA_VISIBLE_DEVICES=$AUX cli gen run --suite "$suite" --pipelines "$pipes" --models configs/models.yaml \
  --out "$R/${tag}_gen.jsonl" --concurrency 192 "$@"
stop_vllms

if [ "$NGPU" -gt 1 ]; then
  CUDA_VISIBLE_DEVICES=$AUX cli gen judge-nli --records "$R/${tag}_gen.jsonl" --out "$R/${tag}_nli.json" &
  nli_pid=$!
  start_vllm openai/gpt-oss-120b 8010 "$GEN" --max-model-len 16384 --gpu-memory-utilization 0.9
  cli gen judge-llm --records "$R/${tag}_gen.jsonl" --out "$R/${tag}_llm.json" --llm-url http://127.0.0.1:8010
  stop_vllms
  wait "$nli_pid"
else
  CUDA_VISIBLE_DEVICES=$AUX cli gen judge-nli --records "$R/${tag}_gen.jsonl" --out "$R/${tag}_nli.json"
  start_vllm openai/gpt-oss-120b 8010 "$GEN" --max-model-len 16384 --gpu-memory-utilization 0.9
  cli gen judge-llm --records "$R/${tag}_gen.jsonl" --out "$R/${tag}_llm.json" --llm-url http://127.0.0.1:8010
  stop_vllms
fi
cli gen merge --records "$R/${tag}_gen.jsonl" --nli "$R/${tag}_nli.json" --llm "$R/${tag}_llm.json" --out "$R/${tag}_judged.jsonl"
cli gen report --judged "$R/${tag}_judged.jsonl" --out "$R/${tag}_report.json"
