#!/usr/bin/env bash
# M4 prep (one GPU): encoder throughput on the GPU, and the recall of cheaper serving profiles
# (rerank depth x reranker) on dev, test and full from the cached candidates.
set -euo pipefail
R=results/m4; mkdir -p "$R/prep"
cli() { in_image python3 -m rageval.cli "$@"; }
EMB=$(in_image python3 -c "import json; print(json.load(open('results/m1/selected.json'))['selected']['embedder_slug'])")
cli bench encoders --out "$R/prep/encoders.json" --candidates "artifacts/cands/dev__$EMB.jsonl" \
  --reranker BAAI/bge-reranker-v2-m3 --reranker Alibaba-NLP/gte-reranker-modernbert-base --reranker cross-encoder/ms-marco-MiniLM-L6-v2 \
  --embedder BAAI/bge-large-en-v1.5 --verifier MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
for rr in Alibaba-NLP/gte-reranker-modernbert-base BAAI/bge-reranker-v2-m3; do
  for s in dev test full; do
    cli retrieval sweep --suite "data/hotpot100k/$s.jsonl" --candidates "artifacts/cands/${s}__$EMB.jsonl" \
      --grid configs/retrieval/serving_grid.yaml --reranker "$rr" --rerank-batch 256 \
      --out "$R/prep/serving_${s}__$(echo "${rr##*/}" | tr '[:upper:]' '[:lower:]').json" > /dev/null
  done
done
