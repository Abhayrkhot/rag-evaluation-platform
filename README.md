# rag-evaluation-platform

A production-style retrieval-augmented generation service and the evaluation platform used to
measure it: hybrid retrieval (FAISS + BM25 + cross-encoder reranking) over a 100,000-document
Wikipedia corpus, grounded answers with sentence-level citations that are verified by an NLI
model, a LangGraph agent served by FastAPI with OpenTelemetry tracing and deadline-aware fallback
routing, and a 500-question evaluation suite scored with ALCE citation metrics, an LLM judge and
paired significance tests.

Every number below is generated from files in [`results/`](results) by
[`scripts/gen_readme.py`](scripts/gen_readme.py), and CI fails if the README drifts from them.
Methodology: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Architecture:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Headline results

<!-- BEGIN:headline -->
_Pending: results not generated yet._
<!-- END:headline -->

## Retrieval: hybrid FAISS + BM25 + reranking over 100K documents

<!-- BEGIN:retrieval -->
Embedder `bge-large-en-v1.5`, reranker `bge-reranker-v2-m3`, selected on dev (dev Recall@10 92.5%) with config `{"alpha": 0.7, "dense_index": "hnsw", "fusion": "weighted", "rerank_blend": 0.0, "rerank_depth": 100}`.

| Pipeline | Test Recall@10 [95% CI] | Test both-hops@10 | Test nDCG@10 | Full Recall@10 [95% CI] | Full both-hops@10 |
|---|---|---|---|---|---|
| bm25 | 75.2% [72.5, 78.0] | 53.2% | 0.701 | 76.6% [75.9, 77.2] | 55.9% |
| dense | 87.8% [85.5, 89.9] | 76.8% | 0.826 | 86.7% [86.1, 87.3] | 75.2% |
| hybrid | 89.5% [87.5, 91.5] | 79.2% | 0.851 | 88.6% [88.1, 89.1] | 78.4% |
| hybrid+rerank | 90.9% [89.0, 92.8] | 82.0% | 0.891 | 92.0% [91.6, 92.5] | 84.3% |

Test: 400 answerable questions. Full: 7018 answerable questions. HNSW vs exact search top-10 overlap: 99.6%.

Index build: Qwen3-Embedding-0.6B 655 docs/s; bge-base-en-v1.5 1928 docs/s; bge-large-en-v1.5 1540 docs/s; BM25 6 s.
<!-- END:retrieval -->

### Two-hop retrieval (LangGraph planner node)

<!-- BEGIN:multihop -->
Second hop for bridge questions: the generator (Qwen3-8B) reads the top 5 first-hop passages and names the Wikipedia article still missing; a second hybrid + rerank pass runs on that title and its top 1 hit is spliced in after the first 2 first-hop hits (planner context and splice chosen on dev; follow-up issued = planner output parsed as a title).

| Suite | Questions | Single hop Recall@10 | Two hop Recall@10 | Bridge Recall@10 | Both hops@10 | Follow-up issued |
|---|---|---|---|---|---|---|
| test | 400 | 90.9% [89.0, 92.8] | 93.9% [92.2, 95.5] | 88.6% -> 92.3% | 82.0% -> 88.0% | 92% |
| full | 7018 | 92.0% [91.6, 92.5] | 95.0% [94.7, 95.4] | 90.1% -> 93.8% | 84.3% -> 90.3% | 95% |
<!-- END:multihop -->

## Grounding: citation precision, stage by stage

<!-- BEGIN:grounding -->
_Pending: results not generated yet._
<!-- END:grounding -->

## Prompt strategies: zero-shot vs few-shot vs chain of thought (paired)

<!-- BEGIN:prompts -->
_Pending: results not generated yet._
<!-- END:prompts -->

## Serving: LangGraph agent at 50 QPS, static vs fallback routing

<!-- BEGIN:serving -->
_Pending: results not generated yet._
<!-- END:serving -->

## Reproduce

```bash
uv sync --all-extras
uv run rageval data build                    # 100K corpus + test/dev/full suites (pinned HotpotQA revision)
uv run rageval index build --embedder Qwen/Qwen3-Embedding-0.6B
uv run pytest -q                             # unit + agent/API tests (fake models)
bash scripts/smoke_pipeline.sh               # CPU smoke test of the whole offline job chain
```

GPU stages (Slurm on Explorer; each job runs the tests first, then one stage):

```bash
sbatch slurm/gpu.sbatch scripts/m1_retrieval.sh                                   # retrieval sweep + selection
sbatch --gres=gpu:h200:2 slurm/gpu.sbatch scripts/m2_run.sh data/hotpot100k/test.jsonl configs/pipelines/m2_test.yaml test
sbatch --gres=gpu:h200:4 slurm/gpu.sbatch scripts/m4_load.sh                      # 50 QPS load tests
```

Serving stack on one host: `docker compose -f deploy/docker-compose.yml up -d` (vLLM primary and
fallback, the agent API, Jaeger, Prometheus, Grafana).

## Limitations

<!-- BEGIN:limitations -->
_Pending: results not generated yet._
<!-- END:limitations -->

## Layout

```
src/rageval/
  data/hotpotqa.py        corpus + suite construction
  retrieval/              bm25, dense (FAISS), fusion, rerank, hybrid, cached experiments
  generation/             prompts, citation parsing, NLI citation verifier
  judges/                 TRUE NLI, classifier NLI, LLM judge
  eval/                   retrieval/answer/grounding metrics, statistics, runners, reports
  llm/                    OpenAI-compatible client, fallback router
  agent/graph.py          LangGraph agent
  serve/                  FastAPI app, micro-batcher, telemetry
  loadtest/loadgen.py     open-loop Poisson load generator
configs/                  retrieval grids, pipelines, models, serving
slurm/, scripts/          Explorer jobs and stage scripts
deploy/                   docker compose, Prometheus, Grafana
results/                  every number in this README
```
