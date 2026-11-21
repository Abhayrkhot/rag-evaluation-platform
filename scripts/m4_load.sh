#!/usr/bin/env bash
# M4: serving latency under open-loop load. Two GPUs (gpu-short allows two per job):
#   GPU0  primary Qwen3-8B (vLLM; PRECISION=bf16|fp8)
#   GPU1  fallback Qwen3-1.7B (vLLM) + the agent's encoders (embedder, reranker, NLI verifier)
#   CPU   the FastAPI/LangGraph agent and the load generator
# PHASE=probe : static-routing capacity sweep only (for sizing), tag prefix probe_<precision>
# PHASE=main  : capacity sweep, then 50 QPS static vs adaptive over three seeds, alternating order
# AGENT (JSON) and RERANKER pick the serving profile; see scripts/m4_prep.sh and README.
set -euo pipefail
source slurm/serve.sh
PHASE=${PHASE:-main}; PRECISION=${PRECISION:-bf16}
R=results/m4/${PHASE}_${PRECISION}; mkdir -p "$R"
cli() { in_image python3 -m rageval.cli "$@"; }
EMB_SLUG=$(in_image python3 -c "import json; print(json.load(open('results/m1/selected.json'))['selected']['embedder_slug'])")
EMB=$(in_image python3 -c "from rageval.retrieval.dense import FORMATS; from rageval.retrieval.experiments import slug; print({slug(m): m for m in FORMATS}['$EMB_SLUG'])")
RERANKER=${RERANKER:-Alibaba-NLP/gte-reranker-modernbert-base}
AGENT=${AGENT:-'{"retrieval": {"fusion": "weighted", "alpha": 0.7, "rerank": true, "rerank_depth": 50, "ef_search": 256}, "context_k": 5, "prompt": "zero_shot", "verify": "repair", "verify_tau": 0.5}'}
cli serve-materialize --template configs/serve/explorer.yaml --out "$R/serve.yaml" --embedder "$EMB" --reranker "$RERANKER" --agent "$AGENT"
POOL=(--suites data/hotpot100k/test.jsonl --suites data/hotpot100k/load_pool.jsonl)
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$R/gpus.csv"
echo "phase=$PHASE precision=$PRECISION reranker=$RERANKER agent=$AGENT" > "$R/layout.txt"

QARGS=(); [ "$PRECISION" = fp8 ] && QARGS=(--quantization fp8)
start_vllm Qwen/Qwen3-8B 8000 0 --max-model-len 4096 --gpu-memory-utilization 0.9 --seed 0 "${QARGS[@]}"
start_vllm Qwen/Qwen3-1.7B 8001 1 --max-model-len 4096 --gpu-memory-utilization 0.4 --seed 0

API_PID=
start_api() {
  local tag=$1; shift
  mkdir -p "$R/$tag"
  CUDA_VISIBLE_DEVICES=1 in_image python3 -m rageval.cli serve --config "$R/serve.yaml" --port 8080 \
    --set "telemetry.trace_log=$R/$tag/trace.jsonl" --set "telemetry.span_log=$R/$tag/spans.jsonl" "$@" \
    > "$R/$tag/api.log" 2>&1 &
  API_PID=$!
  wait_http http://127.0.0.1:8080/health 900
  # warm the encoders and both LLM endpoints (not measured)
  cli loadtest run --url http://127.0.0.1:8080/v1/answer "${POOL[@]}" --qps 5 --duration 20 --warmup 20 --seed 999 \
    --out "$R/$tag/warm.jsonl" > /dev/null
}
stop_api() {
  curl -sf http://127.0.0.1:8080/v1/stats > "$R/$1/stats.json" || true
  kill "$API_PID" 2>/dev/null || true; wait "$API_PID" 2>/dev/null || true
}
load() {  # tag qps duration seed
  cli loadtest run --url http://127.0.0.1:8080/v1/answer "${POOL[@]}" --qps "$2" --duration "$3" --warmup 30 \
    --seed "$4" --out "$R/$1/load.jsonl" --label "$1"
}

SWEEP=${SWEEP_QPS:-20 30 40 50 60 70}
for q in $SWEEP; do
  start_api "sweep_q$q" --set router=static
  load "sweep_q$q" "$q" "${SWEEP_S:-120}" "$((100 + q))"
  stop_api "sweep_q$q"
done
[ "$PHASE" = probe ] && { stop_vllms; exit 0; }

for seed in 1 2 3; do
  order=(static adaptive); [ $((seed % 2)) -eq 0 ] && order=(adaptive static)
  for mode in "${order[@]}"; do
    tag="q50_${mode}_s$seed"
    if [ "$mode" = static ]; then
      start_api "$tag" --set router=static
    else
      start_api "$tag" --set "router=${ROUTER:-adaptive}" --set agent.adaptive=true --set "agent.budget_ms=${BUDGET_MS:-6000}"
    fi
    load "$tag" 50 "${DURATION:-240}" "$seed"
    stop_api "$tag"
  done
done
stop_vllms
b=(); t=(); for s in 1 2 3; do b+=(--base "$R/q50_static_s$s"); t+=(--treat "$R/q50_adaptive_s$s"); done
sw=(); for q in $SWEEP; do sw+=(--sweep "$R/sweep_q$q"); done
cli loadtest compare "${b[@]}" "${t[@]}" "${sw[@]}" --out "$R/compare_q50.json"
