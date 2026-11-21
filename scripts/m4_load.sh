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
# serving profile (latency): single hop, ModernBERT reranker depth 30, K=5, rules prompt, NLI repair
AGENT=${AGENT:-'{"retrieval": {"fusion": "weighted", "alpha": 0.7, "rerank": true, "rerank_depth": 30, "ef_search": 256}, "context_k": 5, "prompt": "zero_shot", "verify": "repair", "verify_tau": 0.5}'}
cli serve-materialize --template configs/serve/explorer.yaml --out "$R/serve.yaml" --embedder "$EMB" --reranker "$RERANKER" --agent "$AGENT"
POOL=(--suites data/hotpot100k/test.jsonl --suites data/hotpot100k/load_pool.jsonl)
nvidia-smi --query-gpu=index,name,memory.total --format=csv > "$R/gpus.csv"
echo "phase=$PHASE precision=$PRECISION reranker=$RERANKER agent=$AGENT" > "$R/layout.txt"

QARGS=(); [ "$PRECISION" = fp8 ] && QARGS=(--quantization fp8)
start_vllm Qwen/Qwen3-8B 8000 0 --max-model-len 4096 --gpu-memory-utilization 0.82 --seed 0 "${QARGS[@]}"
start_vllm Qwen/Qwen3-1.7B 8001 1 --max-model-len 4096 --gpu-memory-utilization 0.3 --seed 0

# encoders on GPU1: embedder and reranker served by vLLM's pooling runner (CUDA graphs, unpadded
# batches); the NLI verifier (DeBERTa, not supported by vLLM) in its own micro-batched process.
start_vllm "$EMB" 8101 1 --runner pooling --max-model-len 512 --gpu-memory-utilization 0.06
start_vllm "$RERANKER" 8102 1 --runner pooling --max-model-len 4096 --gpu-memory-utilization 0.08 \
  --max-num-batched-tokens 131072 --max-num-seqs 1024
# NLI verifier replicas (DeBERTa-v3-large is compute-heavy): two on GPU0 beside the primary, one on GPU1
ENC_PIDS=()
NLI_URLS=""
for spec in 0:8103 0:8104 1:8105; do
  g=${spec%%:*}; port=${spec##*:}
  ENC_PIDS+=($(start_bg "enc-nli-$port" "$g" "http://127.0.0.1:$port/health" python3 -m rageval.cli encoders serve --kind nli \
    --model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --port "$port" --max-wait-ms "${NLI_WAIT_MS:-6}"))
  NLI_URLS="${NLI_URLS:+$NLI_URLS,}http://127.0.0.1:$port"
done
trap 'for p in "${ENC_PIDS[@]}"; do stop_bg "$p"; done; stop_vllms' EXIT
CUDA_VISIBLE_DEVICES=1 cli bench parity --out "$R/parity.json" --embed-url http://127.0.0.1:8101 --embedder "$EMB" \
  --rerank-url http://127.0.0.1:8102 --reranker "$RERANKER" --candidates "artifacts/cands/dev__$EMB_SLUG.jsonl"
URLS=(--set models.embedder_url=http://127.0.0.1:8101 --set models.reranker_url=http://127.0.0.1:8102 --set "models.verifier_url=$NLI_URLS")
API_WORKERS=${API_WORKERS:-4}

API_PID=
start_api() {
  local tag=$1; shift
  mkdir -p "$R/$tag"
  API_PID=$(start_bg "api-$tag" "" http://127.0.0.1:8080/health python3 -m rageval.cli serve --config "$R/serve.yaml" --port 8080 \
    --workers "$API_WORKERS" "${URLS[@]}" --set "telemetry.trace_log=$R/$tag/trace.jsonl" --set "telemetry.span_log=$R/$tag/spans.jsonl" "$@")
  # warm the encoders and both LLM endpoints (not measured)
  cli loadtest run --url http://127.0.0.1:8080/v1/answer "${POOL[@]}" --qps 5 --duration 20 --warmup 20 --seed 999 \
    --out "$R/$tag/warm.jsonl" > /dev/null
}
stop_api() {
  curl -sf http://127.0.0.1:8080/v1/stats > "$R/$1/stats.json" || true
  for k in 8103 8104 8105; do curl -sf "http://127.0.0.1:$k/stats"; echo; done > "$R/$1/encoders.json" || true
  stop_bg "$API_PID"
  # the port must be free before the next run starts its API
  for _ in $(seq 30); do curl -sf -o /dev/null http://127.0.0.1:8080/health || break; sleep 1; done
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
[ "$PHASE" = probe ] && exit 0

for seed in 1 2 3; do
  order=(static adaptive); [ $((seed % 2)) -eq 0 ] && order=(adaptive static)
  for mode in "${order[@]}"; do
    tag="q50_${mode}_s$seed"
    if [ "$mode" = static ]; then
      start_api "$tag" --set router=static
    else
      # fallback routing + graceful degradation; thresholds are per API worker (API_WORKERS of them)
      start_api "$tag" --set "router.max_inflight=${MAX_INFLIGHT:-12}" --set "router.hedge_ms=${HEDGE_MS:-1500}" \
        --set agent.adaptive=true --set "agent.budget_ms=${BUDGET_MS:-4000}" \
        --set "agent.rerank_max_inflight=${RERANK_CAP:-24}" --set "agent.verify_max_inflight=${VERIFY_CAP:-24}"
    fi
    load "$tag" 50 "${DURATION:-240}" "$seed"
    stop_api "$tag"
  done
done
b=(); t=(); for s in 1 2 3; do b+=(--base "$R/q50_static_s$s"); t+=(--treat "$R/q50_adaptive_s$s"); done
sw=(); for q in $SWEEP; do sw+=(--sweep "$R/sweep_q$q"); done
cli loadtest compare "${b[@]}" "${t[@]}" "${sw[@]}" --out "$R/compare_q50.json"
