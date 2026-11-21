#!/usr/bin/env bash
# CPU smoke test of the serving path (box dev container): agent API + fake OpenAI server + open-loop
# load generator + compare. Needs artifacts/smoke from scripts/smoke_pipeline.sh.
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
S=artifacts/smoke; R=$S/serve; rm -rf "$R"; mkdir -p "$R"
cat > $R/serve.yaml <<YML
models:
  corpus: $S/corpus.jsonl
  index_dir: $S/index
  embedder: BAAI/bge-small-en-v1.5
  dense_index: hnsw
  reranker: cross-encoder/ms-marco-MiniLM-L6-v2
  verifier: cross-encoder/nli-deberta-v3-xsmall
  generator: fake/primary
  generator_url: http://127.0.0.1:18000
  fallback: fake/fallback
  fallback_url: http://127.0.0.1:18001
agent: {retrieval: {fusion: weighted, alpha: 0.7, rerank: true, rerank_depth: 20}, context_k: 5, prompt: zero_shot, verify: repair}
router: static
telemetry: {service: rag-agent-smoke}
YML
uv run python scripts/fake_openai.py --port 18000 & p1=$!
uv run python scripts/fake_openai.py --port 18001 & p2=$!
trap 'kill $p1 $p2 $api 2>/dev/null' EXIT
for mode in static adaptive; do
  mkdir -p $R/$mode
  uv run rageval serve --config $R/serve.yaml --port 18080 --set router=$mode --set agent.adaptive=true --set agent.budget_ms=5000 \
    --set telemetry.trace_log=$R/$mode/trace.jsonl --set telemetry.span_log=$R/$mode/spans.jsonl > $R/$mode/api.log 2>&1 & api=$!
  until curl -sf http://127.0.0.1:18080/health > /dev/null; do sleep 1; done
  uv run rageval loadtest run --url http://127.0.0.1:18080/v1/answer --suites $S/dev.jsonl --qps 5 --duration 12 --warmup 2 --seed 1 --out $R/$mode/load.jsonl --label $mode
  curl -sf http://127.0.0.1:18080/v1/stats; echo
  curl -sf http://127.0.0.1:18080/metrics | grep -E "^rag_(route_total|request_seconds_count)" | head -5
  kill $api; wait $api 2>/dev/null || true
done
uv run rageval loadtest compare --base $R/static --treat $R/adaptive --out $R/compare.json > /dev/null
wc -l $R/*/trace.jsonl $R/*/spans.jsonl
head -c 600 $R/adaptive/trace.jsonl; echo
