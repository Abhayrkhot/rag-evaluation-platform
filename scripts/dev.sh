#!/usr/bin/env bash
# Run a command inside the ragdev container with the repo mounted at /src.
#   scripts/dev.sh 'uv run pytest -q'
# Env: RAG_CPUS (docker --cpus), RAG_NET (default host), RAG_GPU=1 (--gpus all).
set -euo pipefail
cd "$(dirname "$0")/.."
image=ragdev:py3.12
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build -q -t "$image" -f docker/dev.Dockerfile docker >/dev/null
fi
mkdir -p "$HOME/.cache/ragdev"
args=(--rm -i -v "$PWD:/src" -v "$HOME/.cache/ragdev:/cache" -w /src
      --user "$(id -u):$(id -g)" -e HOME=/cache --network "${RAG_NET:-host}"
      -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}")
[ -n "${RAG_CPUS:-}" ] && args+=(--cpus "$RAG_CPUS")
[ "${RAG_GPU:-0}" = 1 ] && args+=(--gpus all)
exec docker run "${args[@]}" "$image" bash -lc "$*"
