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
| Claim | Target | Measured |
|---|---|---|
| Recall@10, hybrid FAISS + BM25 + reranking, 100K docs | 91% | 95.0% two-hop / 92.0% single-hop (7,018 questions); test suite 93.9% / 90.9% |
| Citation precision, naive RAG -> final pipeline (500-question suite) | 76% -> 94% | 57.3% -> 85.4% (balanced, correct 81.8%) / 93.0% (strict, correct 73.8%); naive correct 75.8% |
| Unsupported claims, zero-shot vs cot (paired) | -32% | -57% (198 -> 85), Wilcoxon p (Holm) 4.5e-13; per-claim rate on questions both answered 16.6% -> 14.1% (not significant) |
| p95 latency at 50 QPS, static vs fallback routing | -28% | -35% (1.97 s -> 1.29 s; 95% CI 30 to 39%) |
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
Test suite, 500 questions (400 answerable, 100 unanswerable). Citation precision and recall are ALCE's, judged by TRUE (T5-11B NLI) over answered questions; correctness is gpt-oss-120b against the HotpotQA answer on answerable questions; abstention is on unanswerable questions. Generator: Qwen3-8B, greedy.

| Stage | Citation precision | Citation recall | Correct (answerable) | Abstains (unanswerable) | Gold docs in context |
|---|---|---|---|---|---|
| S0 naive RAG: hybrid top-10, no reranking, plain "cite [n]" prompt | 57.3% [53.3, 61.3] | 35.1% [32.2, 38.0] | 75.8% [71.5, 80.0] | 6.0% | 89.5% |
| S1a + cross-encoder reranking | 56.5% [52.5, 60.6] | 35.1% [32.2, 38.0] | 76.2% [72.0, 80.5] | 4.0% | 90.9% |
| S1b + two-hop retrieval | 59.0% [55.0, 62.9] | 36.4% [33.6, 39.4] | 78.2% [74.2, 82.2] | 4.0% | 93.9% |
| S2 + citation and grounding rules (zero-shot) | 71.5% [68.2, 74.8] | 68.6% [65.3, 71.8] | 80.5% [76.5, 84.2] | 31.0% | 93.9% |
| S3 + best prompt strategy (picked on dev) (`cot`) | 75.8% [72.3, 79.2] | 78.9% [75.5, 82.2] | 77.8% [73.8, 81.8] | 70.0% | 93.9% |
| S4 + NLI citation repair and one revision round (balanced) | 85.4% [82.9, 87.7] | 87.6% [85.3, 89.7] | 81.8% [78.0, 85.5] | 45.0% | 93.9% |
| S4 strict: NLI and MiniCheck must both accept | 93.0% [91.3, 94.6] | 95.9% [94.5, 97.1] | 73.8% [69.5, 78.0] | 52.0% | 93.9% |
<!-- END:grounding -->

## Prompt strategies: zero-shot vs few-shot vs chain of thought (paired)

<!-- BEGIN:prompts -->
Same 500 questions and identical retrieved contexts for every strategy; unsupported claims are statements gpt-oss-120b judges not fully supported by the documents the generator saw.

| Strategy | Unsupported claims | Claims | Unsupported rate [95% CI] | Answers with one+ | Citation precision | Correct | Abstains (unanswerable) |
|---|---|---|---|---|---|---|---|
| zero_shot | 198 | 904 | 21.9% [19.1, 24.8] | 36.6% | 71.5% [68.2, 74.8] | 80.5% [76.5, 84.2] | 31.0% |
| few_shot | 297 | 1232 | 24.1% [21.3, 27.0] | 41.2% | 70.7% [67.7, 73.7] | 82.2% [78.5, 86.0] | 20.0% |
| cot | 85 | 660 | 12.9% [10.3, 15.6] | 19.4% | 75.8% [72.3, 79.2] | 77.8% [73.8, 81.8] | 70.0% |

Unsupported claims, `zero_shot` -> `cot`: 198 -> 85 (57% fewer over the suite). Part of that is fewer and more cautious claims: on the questions both strategies answered, the per-claim unsupported rate falls less and the difference is not significant (paired `unsupported_rate` row below), and CoT abstains more, on unanswerable questions (good) and on answerable ones (bad).

Paired tests (bootstrap CI on the per-question difference; McNemar for binary metrics, Wilcoxon otherwise; Holm across the family):

| Comparison | Metric | n | Mean | Delta [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|
| zero_shot vs few_shot | unsupported_count | 500 | 0.396 -> 0.594 | +0.198 [+0.134, +0.264] | 3.9e-09 | 1.1e-07 |
| zero_shot vs few_shot | has_unsupported | 500 | 0.342 -> 0.396 | +0.054 [+0.014, +0.094] | 0.01 | 0.22 |
| zero_shot vs few_shot | unsupported_rate | 465 | 0.235 -> 0.219 | -0.016 [-0.043, +0.011] | 0.065 | 1 |
| zero_shot vs few_shot | citation_precision | 465 | 0.718 -> 0.720 | +0.002 [-0.030, +0.033] | 0.97 | 1 |
| zero_shot vs few_shot | correct | 400 | 0.805 -> 0.823 | +0.018 [-0.013, +0.048] | 0.32 | 1 |
| zero_shot vs few_shot | abstain_unanswerable | 100 | 0.310 -> 0.200 | -0.110 [-0.190, -0.040] | 0.0074 | 0.18 |
| zero_shot vs cot | unsupported_count | 500 | 0.396 -> 0.170 | -0.226 [-0.280, -0.174] | 1.2e-14 | 4.5e-13 |
| zero_shot vs cot | has_unsupported | 500 | 0.342 -> 0.152 | -0.190 [-0.232, -0.148] | 1.1e-17 | 4.3e-16 |
| zero_shot vs cot | unsupported_rate | 390 | 0.166 -> 0.141 | -0.025 [-0.055, +0.004] | 0.15 | 1 |
| zero_shot vs cot | citation_precision | 390 | 0.783 -> 0.757 | -0.025 [-0.065, +0.015] | 0.22 | 1 |
| zero_shot vs cot | correct | 400 | 0.805 -> 0.777 | -0.028 [-0.060, +0.005] | 0.14 | 1 |
| zero_shot vs cot | abstain_unanswerable | 100 | 0.310 -> 0.700 | +0.390 [+0.290, +0.490] | 3.8e-11 | 1.3e-09 |
<!-- END:prompts -->

## Serving: LangGraph agent at 50 QPS, static vs fallback routing

<!-- BEGIN:serving -->
Open-loop Poisson arrivals at 50 QPS, 2 seeds per mode, alternating order, 30 s warm-up excluded (`main_fp8`). Static: every request takes the full serving pipeline on the primary model. Fallback routing: stage fallbacks (rerank 15 instead of 30 candidates, or verify only the cited documents and prune what fails, skipping the repair search) when that stage is backed up or the request is already late, plus model failover to Qwen3-1.7B through the circuit breaker and retry. Thresholds are calibrated from the 40 QPS sweep point, not from these runs (`calibration.json`). Load-based overflow, deadline routing and hedging to the small model were tried and made p95 worse here; see docs/RESULTS_LOG.md.

| Latency | Static | Fallback routing | Reduction [95% CI, block bootstrap] | Per seed (static / routed), s |
|---|---|---|---|---|
| p50 | 0.84 s | 0.65 s | 22% [17, 26] | 0.81, 0.86 / 0.65, 0.66 |
| p90 | 1.66 s | 1.16 s | 30% [26, 34] | 1.64, 1.69 / 1.14, 1.18 |
| p95 | 1.97 s | 1.29 s | 35% [30, 39] | 1.95, 1.99 / 1.27, 1.31 |
| p99 | 2.78 s | 1.58 s | 43% [36, 49] | 2.80, 2.73 / 1.56, 1.62 |

Requests per mode: 21166 static, 21166 routed; errors 0.00% / 0.00%. Routed traffic: primary 100.0%. Degraded stages: rerank 0.5%, rerank_shallow 27.8%, verify 0.0%, verify_fast 19.0%.

Serving profile retrieval (single hop, gte-reranker-modernbert-base at depth 30): Recall@10 test 91.4%, full 91.2%. vLLM-served encoders vs the offline HF models: embedding cosine >= 0.9999, rerank Spearman 0.998, top-10 overlap 99%.

Quality of the answers served during these runs (test-suite questions in the request pool, judged like the offline results; the FP8 primary and the 5-document serving profile make these lower than the offline pipeline's):

| Mode | Answers | Citation precision | Correct (answerable) | Unsupported rate |
|---|---|---|---|---|
| Static | 539 | 78.1% [75.4, 80.7] | 77.3% [73.2, 81.2] | 21.2% |
| Fallback routing | 539 | 82.0% [79.3, 84.5] | 74.8% [70.6, 78.7] | 17.8% |

Routed answers by path (paths with 20+ answers): full: n=304, citation precision 78.7%, correct 78.0%; rerank_shallow: n=100, citation precision 77.8%, correct 76.2%; rerank_shallow+verify_fast: n=41, citation precision 93.6%, correct 62.5%; verify_fast: n=89, citation precision 93.3%, correct 67.6%. The shallow rerank path costs little; the fast verification path prunes instead of repairing, which raises citation precision and lowers correctness.

A third seed per mode was started and aborted when the reranker's vLLM engine died with a CUDA launch failure (error 719) after 32 minutes of sustained load; its requests are excluded (`aborted_seed3/README.txt`). The agent now degrades on encoder errors (fused order, unverified answer) instead of failing the request.

Capacity sweep with static routing (sizing):

| Offered QPS | p50 | p95 | p99 | Errors |
|---|---|---|---|---|
| 29.5 | 0.40 s | 0.76 s | 0.97 s | 0.0% |
| 39.5 | 0.51 s | 1.06 s | 1.47 s | 0.0% |
| 50.1 | 0.81 s | 1.88 s | 2.55 s | 0.0% |
| 60.4 | 18.02 s | 37.47 s | 42.45 s | 24.5% |
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
- One dataset. Every number is on a 100K-paragraph corpus built from HotpotQA (Wikipedia intro paragraphs). It contains each validation question's hard negatives but is far smaller than full Wikipedia, and Qwen3 has likely seen HotpotQA, so parametric knowledge can leak into answers. The unsupported-claims judge sees only the retrieved documents, so leaked facts count against the pipeline rather than for it.
- Model judges, no human labels. Citation metrics use TRUE (T5-11B NLI), as ALCE does; unsupported claims and correctness use gpt-oss-120b. Both are models with their own errors. The pipeline's verifier (DeBERTa NLI, optionally with MiniCheck) is a different model from both judges, and the strict operating point's gain also shows up in the gpt-oss judge, but no human agreement study was run.
- Small dev set. Every choice (fusion, depth, splice, K, prompt, verifier threshold) was made on 250 dev questions (200 answerable), so the dev picks are noisy; the test suite and the 7,018-question full set were each scored once with the frozen choices.
- Precision vs correctness. The strict citation operating point prunes more aggressively and answers fewer questions correctly than the balanced one; both are reported.
- Serving profile differs from the best offline pipeline. Two-hop retrieval, 10-document contexts and the revision round cost extra LLM calls, so the 50 QPS deployment serves a latency profile (single hop, ModernBERT reranker at depth 30, 5 documents, NLI repair without revision) whose retrieval recall is reported in its own section.
- Hardware-specific latency. The load tests run on two H200s (GPU0: FP8 Qwen3-8B primary and the Qwen3-1.7B failover model; GPU1: vLLM embedder and reranker plus three NLI replicas) with a single-node API. The FP8 primary is not the BF16 model the offline evaluation used; answer quality under load is judged separately. Results will differ on other hardware.
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
