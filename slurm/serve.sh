# Sourced by stage scripts (after slurm/env.sh): start/stop vLLM servers inside the image.
#   start_vllm <model> <port> <gpu-list> [extra vllm args...]   -> waits for /health, records the PID
#   stop_vllms                                                   -> stops every server this script started
VLLM_PIDS=()
start_vllm() {
  local model=$1 port=$2 gpus=$3; shift 3
  local log="$RAG_ROOT/logs/vllm-${SLURM_JOB_ID:-local}-$(echo "${model##*/}" | tr '[:upper:]' '[:lower:]')-$port.log"
  # own session/process group, so stop_vllms can kill vLLM's engine workers too, not just the wrapper
  CUDA_VISIBLE_DEVICES=$gpus setsid bash -c 'in_image vllm serve "$@"' _ "$model" --host 127.0.0.1 --port "$port" \
    --disable-log-requests --uvicorn-log-level warning "$@" > "$log" 2>&1 &
  local pid=$!
  VLLM_PIDS+=($pid)
  echo "vllm $model on :$port (gpu $gpus) pid $pid log $log"
  local deadline=$(( $(date +%s) + 1800 ))
  until curl -sf -o /dev/null "http://127.0.0.1:$port/health"; do
    if ! kill -0 "$pid" 2>/dev/null; then echo "vllm $model exited during startup" >&2; tail -40 "$log" >&2; return 1; fi
    [ "$(date +%s)" -gt "$deadline" ] && { echo "vllm $model startup timeout" >&2; tail -40 "$log" >&2; return 1; }
    sleep 5
  done
}
stop_vllms() {
  for p in "${VLLM_PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true; done
  for p in "${VLLM_PIDS[@]}"; do
    for _ in $(seq 60); do kill -0 -- "-$p" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$p" 2>/dev/null || true
  done
  VLLM_PIDS=()
  # wait until this job's GPUs are actually free (engine workers release memory last)
  for _ in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt 2000 ] && break
    sleep 2
  done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}
trap stop_vllms EXIT

# start_bg <name> <gpu-list> <health-url> <command...>: run a command in its own process group inside
# the image, wait for its health URL, and print the group id (use stop_bg <pgid> to stop it all).
start_bg() {
  local name=$1 gpus=$2 health=$3; shift 3
  local log="$RAG_ROOT/logs/${name}-${SLURM_JOB_ID:-local}.log"
  CUDA_VISIBLE_DEVICES=$gpus setsid bash -c 'in_image "$@"' _ "$@" > "$log" 2>&1 &
  local pid=$!
  local deadline=$(( $(date +%s) + 900 ))
  until curl -sf -o /dev/null "$health"; do
    if ! kill -0 "$pid" 2>/dev/null; then echo "$name exited during startup" >&2; tail -40 "$log" >&2; return 1; fi
    [ "$(date +%s)" -gt "$deadline" ] && { echo "$name startup timeout" >&2; tail -40 "$log" >&2; return 1; }
    sleep 2
  done
  echo "$pid"
}
stop_bg() {
  local pid=$1
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 30); do kill -0 -- "-$pid" 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- "-$pid" 2>/dev/null || true
}
