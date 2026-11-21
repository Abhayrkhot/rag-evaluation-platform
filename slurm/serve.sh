# Sourced by stage scripts (after slurm/env.sh): start/stop vLLM servers inside the image.
#   start_vllm <model> <port> <gpu-list> [extra vllm args...]   -> waits for /health, records the PID
#   stop_vllms                                                   -> stops every server this script started
VLLM_PIDS=()
start_vllm() {
  local model=$1 port=$2 gpus=$3; shift 3
  local log="$RAG_ROOT/logs/vllm-${SLURM_JOB_ID:-local}-$(echo "${model##*/}" | tr '[:upper:]' '[:lower:]')-$port.log"
  CUDA_VISIBLE_DEVICES=$gpus in_image vllm serve "$model" --host 127.0.0.1 --port "$port" \
    --disable-log-requests --uvicorn-log-level warning "$@" > "$log" 2>&1 &
  VLLM_PIDS+=($!)
  echo "vllm $model on :$port (gpu $gpus) pid $! log $log"
  if ! wait_http "http://127.0.0.1:$port/health" 1800; then
    tail -50 "$log" >&2; return 1
  fi
}
stop_vllms() {
  for p in "${VLLM_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  for p in "${VLLM_PIDS[@]}"; do wait "$p" 2>/dev/null || true; done
  VLLM_PIDS=()
  sleep 5
}
trap stop_vllms EXIT
