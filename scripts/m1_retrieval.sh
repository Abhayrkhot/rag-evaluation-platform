#!/usr/bin/env bash
# M1: retrieval experiments (one GPU). Sourced env: slurm/env.sh (in_image).
set -euo pipefail
R=results/m1; mkdir -p "$R"
EMBS=(Qwen/Qwen3-Embedding-0.6B BAAI/bge-base-en-v1.5 BAAI/bge-large-en-v1.5)
RRS=(BAAI/bge-reranker-v2-m3 Alibaba-NLP/gte-reranker-modernbert-base cross-encoder/ms-marco-MiniLM-L6-v2)
cli() { in_image python3 -m rageval.cli "$@"; }
slug() { echo "${1##*/}" | tr '[:upper:]' '[:lower:]'; }

args=(); for e in "${EMBS[@]}"; do args+=(--embedder "$e"); done
cli index build "${args[@]}" --batch-size 256 > "$R/index_build.json"
for e in "${EMBS[@]}"; do
  for s in dev test full; do
    cli retrieval candidates --suite "data/hotpot100k/$s.jsonl" --embedder "$e" --batch-size 256 > "$R/cands_${s}__$(slug "$e").json"
  done
done
sweeps=()
for e in "${EMBS[@]}"; do
  for r in "${RRS[@]}"; do
    f="$R/dev__$(slug "$e")__$(slug "$r").json"
    cli retrieval sweep --suite data/hotpot100k/dev.jsonl --candidates "artifacts/cands/dev__$(slug "$e").jsonl" \
      --grid configs/retrieval/dev_grid.yaml --reranker "$r" --rerank-batch 256 --out "$f" > /dev/null
    sweeps+=(--sweeps "$f")
  done
done
cli retrieval select "${sweeps[@]}" --out "$R/selected.json"
e=$(in_image python3 -c "import json;print(json.load(open('$R/selected.json'))['selected']['embedder_slug'])")
r=$(in_image python3 -c "import json;print(json.load(open('$R/selected.json'))['selected']['reranker'])")
for s in test full; do
  cli retrieval sweep --suite "data/hotpot100k/$s.jsonl" --candidates "artifacts/cands/${s}__${e}.jsonl" \
    --grid "$R/selected.grid.yaml" --reranker "$r" --rerank-batch 256 --out "$R/final_${s}.json" \
    --rows-out "$R/final_${s}.rows.jsonl"
done
