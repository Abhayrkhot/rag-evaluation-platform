# Sourced by every Explorer job: paths, caches, and `in_image` (run a command inside the vLLM
# image with this project's venv on top of the image's torch / transformers / vLLM).
export RAG_ROOT=${RAG_ROOT:-/scratch/$USER/rag}
export RAG_SIF=${RAG_SIF:-/scratch/$USER/icp/vllm.sif}
export HF_HOME=$RAG_ROOT/hf HF_HUB_DISABLE_TELEMETRY=1
# jobs only read the pre-downloaded cache (slurm/setup.sbatch sets RAG_ONLINE=1 to download)
[ "${RAG_ONLINE:-0}" = 1 ] || export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=$RAG_ROOT/cache TRITON_CACHE_DIR=$RAG_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$RAG_ROOT/cache/inductor PIP_CACHE_DIR=$RAG_ROOT/pip-cache
export VLLM_CACHE_ROOT=$RAG_ROOT/cache/vllm
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1  # ~/.local has an old torch
export TRANSFORMERS_VERBOSITY=error HF_DATASETS_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=true
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy} NO_PROXY=localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}
export RAGEVAL_HOST_LABEL="explorer-$(hostname -s)"
# The scratch copy is a `git archive` of one commit, recorded in GIT_SHA next to the code.
export RAGEVAL_GIT_SHA=${RAGEVAL_GIT_SHA:-$(cat "${SLURM_SUBMIT_DIR:-.}/GIT_SHA" 2>/dev/null || echo unknown)}
mkdir -p "$RAG_ROOT/logs" "$HF_HOME/hub" "$XDG_CACHE_HOME"
# The image's CUDA 13 user-mode driver needs driver >= 580; older nodes use its compat libs.
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)
export RAG_CUDA_COMPAT=0
if [ -n "$DRIVER" ] && [ "$DRIVER" -lt 580 ]; then RAG_CUDA_COMPAT=1; fi

# in_image <command...>
in_image() {
  apptainer exec --nv --bind "/scratch/$USER" --pwd "$PWD" "$RAG_SIF" bash -c '
    if [ "$RAG_CUDA_COMPAT" = 1 ]; then export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; fi
    source "$RAG_ROOT/venv/bin/activate"
    export PYTHONPATH=$PWD/src
    exec "$@"' _ "$@"
}

# wait_http <url> <timeout_s>: poll until a server answers 200
wait_http() {
  local url=$1 deadline=$(( $(date +%s) + ${2:-900} ))
  until curl -sf -o /dev/null "$url"; do
    [ "$(date +%s)" -gt "$deadline" ] && { echo "timeout waiting for $url" >&2; return 1; }
    sleep 5
  done
}
