#!/usr/bin/env bash
# CPU smoke test of the offline job chain (box dev container): mini corpus -> indexes ->
# candidates/sweep -> gen run (fake LLM) -> judges (small NLI + fake LLM judge) -> merge -> report.
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
S=artifacts/smoke; rm -rf "$S"; mkdir -p "$S"
uv run python - <<'PY'
import random
from rageval.io import read_jsonl, write_jsonl
qs = list(read_jsonl("data/hotpot100k/dev.jsonl"))[:16]
gold = {g for q in qs for g in q["gold_doc_ids"]}
docs = list(read_jsonl("data/hotpot100k/corpus.jsonl"))
rng = random.Random(0)
keep = [d for d in docs if d["doc_id"] in gold] + rng.sample(docs, 1500)
write_jsonl("artifacts/smoke/corpus.jsonl", {d["doc_id"]: d for d in keep}.values())
write_jsonl("artifacts/smoke/dev.jsonl", qs)
PY
cli() { uv run rageval "$@"; }
cli index build --corpus $S/corpus.jsonl --out $S/index --embedder BAAI/bge-small-en-v1.5 --batch-size 64 > /dev/null
cli retrieval candidates --suite $S/dev.jsonl --embedder BAAI/bge-small-en-v1.5 --index-dir $S/index --out-dir $S/cands > /dev/null
cli retrieval sweep --suite $S/dev.jsonl --candidates $S/cands/dev__bge-small-en-v1.5.jsonl --grid configs/retrieval/smoke_grid.yaml \
  --corpus $S/corpus.jsonl --reranker cross-encoder/ms-marco-MiniLM-L6-v2 --rerank-cache-dir $S/rerank --out $S/sweep.json > /dev/null
cli retrieval select --sweeps $S/sweep.json --out $S/selected.json > /dev/null
cat > $S/models.yaml <<YML
corpus: $S/corpus.jsonl
index_dir: $S/index
embedder: BAAI/bge-small-en-v1.5
dense_index: hnsw
reranker: cross-encoder/ms-marco-MiniLM-L6-v2
verifier: cross-encoder/nli-deberta-v3-xsmall
generator: fake/generator
generator_url: http://127.0.0.1:18000
YML
cat > $S/pipes.yaml <<YML
base: {retrieval: {fusion: rrf, rerank: true, rerank_depth: 20}, context_k: 5, prompt: zero_shot}
pipelines:
  naive: {retrieval: {rerank: false}, context_k: 10, prompt: naive}
  zero: {}
  cot: {prompt: cot}
  zero_repair: {verify: repair}
  zero_prune: {verify: prune}
YML
uv run python scripts/fake_openai.py --port 18000 & srv=$!
trap 'kill $srv' EXIT
until curl -sf http://127.0.0.1:18000/health > /dev/null; do sleep 1; done
cli gen run --suite $S/dev.jsonl --pipelines $S/pipes.yaml --models $S/models.yaml --out $S/gen.jsonl
cli gen judge-nli --records $S/gen.jsonl --corpus $S/corpus.jsonl --out $S/nli.json --kind class --model cross-encoder/nli-deberta-v3-xsmall
cli gen judge-llm --records $S/gen.jsonl --corpus $S/corpus.jsonl --out $S/llm.json --llm-url http://127.0.0.1:18000 --llm-model fake/judge
cli gen merge --records $S/gen.jsonl --nli $S/nli.json --llm $S/llm.json --out $S/judged.jsonl
cli gen report --judged $S/judged.jsonl --out $S/report.json --pairs naive:zero --pairs zero:cot --pairs zero:zero_prune
