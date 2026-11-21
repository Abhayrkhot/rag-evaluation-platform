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
_Pending: results not generated yet._
<!-- END:retrieval -->

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
