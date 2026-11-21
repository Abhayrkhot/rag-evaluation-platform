#!/usr/bin/env bash
# M4: serving latency under open-loop load (4 GPUs on one node).
#   GPU0-1  primary Qwen3-8B (vLLM, data parallel 2)
#   GPU2    fallback Qwen3-1.7B (vLLM)
#   GPU3    the agent's in-process encoders (embedder, reranker, NLI verifier)
#   CPU     the FastAPI/LangGraph agent and the load generator
# Phases: capacity sweep (static routing), then 50 QPS static vs adaptive, three seeds each,
# alternating order. Every run restarts the API so router state starts cold.
set -euo pipefail
source slurm/serve.sh
R=results/m4; mkdir -p "$R"
SERVE=${SERVE:-configs/serve/explorer.yaml}
POOL=(--suites data/hotpot100k/test.jsonl --suites data/hotpot100k/load_pool.jsonl)
cli() { in_image python3 -m rageval.cli "$@"; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$R/gpus.csv"

# GPU layout (overridable): primary data-parallel over PRIMARY_GPUS, fallback on FALLBACK_GPU,
# encoders on AUX_GPU. FALLBACK_UTIL < 0.9 when the fallback shares its GPU with the encoders.
PRIMARY_GPUS=${PRIMARY_GPUS:-0,1}; FALLBACK_GPU=${FALLBACK_GPU:-2}; AUX_GPU=${AUX_GPU:-3}
FALLBACK_UTIL=${FALLBACK_UTIL:-0.9}
DP=$(echo "$PRIMARY_GPUS" | tr ',' '\n' | wc -l)
echo "primary gpus=$PRIMARY_GPUS dp=$DP fallback=$FALLBACK_GPU aux=$AUX_GPU" | tee "$R/layout.txt"
start_vllm Qwen/Qwen3-8B 8000 "$PRIMARY_GPUS" --data-parallel-size "$DP" --max-model-len 8192 --gpu-memory-utilization 0.9 --seed 0
start_vllm Qwen/Qwen3-1.7B 8001 "$FALLBACK_GPU" --max-model-len 8192 --gpu-memory-utilization "$FALLBACK_UTIL" --seed 0

API_PID=
start_api() {
  local tag=$1; shift
  mkdir -p "$R/$tag"
  CUDA_VISIBLE_DEVICES=$AUX_GPU in_image python3 -m rageval.cli serve --config "$SERVE" --port 8080 \
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

# 1) capacity sweep with static routing (primary only)
for q in ${SWEEP_QPS:-10 20 30 40 50 60}; do
  start_api "sweep_q$q" --set router=static
  load "sweep_q$q" "$q" 150 "$((100 + q))"
  stop_api "sweep_q$q"
done

# 2) 50 QPS: static vs adaptive, three seeds, alternating order
for seed in 1 2 3; do
  order=(static adaptive); [ $((seed % 2)) -eq 0 ] && order=(adaptive static)
  for mode in "${order[@]}"; do
    tag="q50_${mode}_s$seed"
    if [ "$mode" = static ]; then
      start_api "$tag" --set router=static
    else
      start_api "$tag" --set router=adaptive --set agent.adaptive=true --set "agent.budget_ms=${BUDGET_MS:-6000}"
    fi
    load "$tag" 50 "${DURATION:-300}" "$seed"
    stop_api "$tag"
  done
done
stop_vllms
b=(); t=(); for s in 1 2 3; do b+=(--base "$R/q50_static_s$s"); t+=(--treat "$R/q50_adaptive_s$s"); done
sw=(); for q in ${SWEEP_QPS:-10 20 30 40 50 60}; do sw+=(--sweep "$R/sweep_q$q"); done
cli loadtest compare "${b[@]}" "${t[@]}" "${sw[@]}" --out "$R/compare_q50.json"
