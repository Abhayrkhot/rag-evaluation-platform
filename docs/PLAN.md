# Build plan

Target resume entry (numbers to beat, all must be measured and reproducible from `results/`):

| Claim | Target to beat |
|---|---|
| Hybrid FAISS + BM25 + reranking over 100K documents | Recall@10 > 91% |
| 500-question grounding and retrieval suite | citation precision lifted from a naive baseline to > 94% |
| Zero-shot vs few-shot vs chain-of-thought, paired tests | unsupported claims cut by > 32% |
| LangGraph agent with trace logs and fallback routing | p95 latency cut by > 28% at 50 QPS |

## Data

- Corpus: exactly 100,000 Wikipedia intro paragraphs. All 66,581 paragraphs from the HotpotQA
  validation (distractor) contexts, which include the TF-IDF hard negatives HotpotQA mined for each
  question, plus paragraphs sampled from the train contexts, minus the supporting paragraphs of the
  unanswerable questions.
- Test suite (500): stratified by HotpotQA type in its natural bridge/comparison ratio for the 400
  answerable questions, plus 100 unanswerable questions whose supporting paragraphs are removed from
  the corpus. Each question keeps its gold paragraphs, gold supporting sentences and answer.
- Dev split (250, disjoint) is the only data used to tune fusion weights, depths and prompts.
- Retrieval-full: every remaining answerable validation question, for tight retrieval CIs.

## Models (all open weights, $0, served with vLLM on Explorer H200s)

- Embedder and reranker candidates compared on dev (Qwen3-Embedding-0.6B / bge-base,
  bge-reranker-v2-m3 / Qwen3-Reranker-0.6B).
- Generator Qwen3-8B (non-thinking), fallback Qwen3-1.7B.
- In-pipeline citation verifier: DeBERTa-v3-large NLI (small, fast).
- Evaluators are different models from anything in the pipeline: TRUE T5-11B NLI (the ALCE
  citation metric) and gpt-oss-120b as an LLM judge (different family from the generator).

## Milestones

- M0: repo, CI, corpus and suite builder, BM25 + FAISS + fusion + reranker, retrieval metrics.
- M1: retrieval experiments on Explorer (embedders, rerankers, fusion), Recall@10 on suite and full.
- M2: grounded generation with citations, ALCE-style citation precision/recall, staged ablation
  (baseline, reranked context, citation prompt, NLI citation verification).
- M3: zero-shot / few-shot / CoT prompts, paired bootstrap + Wilcoxon + McNemar with Holm correction.
- M4: LangGraph agent served by FastAPI, OpenTelemetry spans + JSONL trace logs + Prometheus,
  deadline-aware fallback routing and circuit breaker, open-loop 50 QPS load tests.
- M5: README generated from results files (CI `--check`), docs, limitations.
