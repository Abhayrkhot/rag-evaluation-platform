# Results log

Decisions and dead ends, in order, so the final configuration can be traced.

## 2026-11-21

- **M1 retrieval (dev, 200 answerable).** bge-large beat Qwen3-Embedding-0.6B as the dense leg
  (dense-only Recall@10 87.5% vs 79.3%; hybrid candidate ceiling at depth 100 95.5% vs 92.5%).
  Selected: bge-large + bge-reranker-v2-m3, weighted fusion alpha 0.7, rerank depth 100
  (dev 92.5%). Held out: test 90.9%, full 92.0%.
- **Two-hop retrieval, first planner prompt.** The planner could answer NONE; Qwen3-8B did so on
  85% of dev questions, including clear bridge misses (first article retrieved, second missing),
  so dev Recall@10 only moved 92.50% -> 92.75%. Replaced with a prompt that always names the
  missing (or answer-bearing) article; the splice (keep, take) limits displacement and is tuned on dev.
- **Serving reranker.** bge-reranker-v2-m3 is GPU-bound near 1.5-2K pairs/s on an H200, so depth 100
  at 50 QPS (5K pairs/s) cannot be served on one shared GPU. gte-reranker-modernbert-base at depth
  50 keeps full-set Recall@10 at 91.7% (test 91.4%) and is the serving profile. Encoder throughput was
  CPU-bound by single-threaded tokenization; fixed by one parallel tokenizer call per request batch.
- **Infrastructure.** The `gpu` QOS allows one GPU per job and four per user; `gpu-short` two per job.
  M2 runs as single-GPU sequential jobs; M4 uses a two-GPU `gpu-short` job.
