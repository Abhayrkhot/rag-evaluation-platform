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
- **M2 dev iteration 1** (rules v1; archived in `results/explorer/m2/dev_iter1`). TRUE-judged citation
  precision was 52-75% across pipelines (naive baseline 57%). Reading the failures: two-hop
  sentences cited one of their two sources, about 11% of sentences carried no citation, and the
  sentence splitter broke "Vol. 2" titles. Pruning unsupported sentences raised precision but cut
  correctness from 82% to 70% (answers emptied into abstentions).
  Iteration 2: rules v2 (one fact per sentence, cite every source, no pronouns), citation repair
  that first completes a citation with the missing second source and minimises the result, and a
  `revise` policy that sends flagged sentences back to the generator once and prunes what still fails.
- **M2 dev iteration 2** (archived in `results/explorer/m2/dev_iter2`). Rules v2 alone lifted
  TRUE citation precision from 52% to 73% (zero-shot, K=10); revise reached 83-85% at unchanged
  correctness (about 0.84). Remaining errors included NLI false positives found by the repair
  search (DeBERTa gave 0.98 entailment for "The series aired on Dragon TV, where Li Yitong made
  her television debut" from The CW's paragraph) and a sentence split at "Roth v. United States".
  Iteration 3: re-citation candidates must share an entity token with the statement, "v." added
  to the abbreviations, revise threshold swept at K=10.
- **M4 capacity probes.** In-process encoders saturated near 25 QPS (GIL and kernel-launch bound;
  the old API process also survived restarts because `kill` hit the apptainer wrapper). Separate
  encoder processes: the reranker saturated near 30 QPS. vLLM pooling servers for the embedder and
  reranker (parity vs HF: embedding cosine >= 0.9999, rerank Spearman 0.998): reranking at 30 QPS
  dropped from 10.6 s to 0.16 s p50, and the DeBERTa verifier became the bottleneck, so it now
  runs as three replicas across both GPUs.
- **M2 dev iteration 3** (archived in `results/explorer/m2/dev_iter3`). Revise at tau 0.9 reached
  86.9% TRUE citation precision (zero-shot) at correctness 0.825. Of the 63 remaining imprecise
  statements, 35 were judged supported by gpt-oss: meta sentences ("... is not directly mentioned
  in the documents") counted as claims, redundant second citations kept because minimisation used
  the strict 0.9 threshold, and DeBERTa false positives even at 0.9. Iteration 4: wider abstention
  patterns, minimisation at the NLI decision threshold (0.5), and an ensemble gate with MiniCheck
  (DeBERTa-v3-large, Tang et al. 2024) that must agree with the NLI model.
- **M4 main, BF16 primary** (archived as `results/explorer/m4/main_bf16_overloaded`, not used for
  claims). With the encoders fixed, the BF16 Qwen3-8B primary (sharing GPU0 with two NLI replicas)
  saturated between 30 and 40 QPS: static routing at 50 QPS had a growing queue (p95 165 s), and
  adaptive routing with hand-set thresholds sent 89% of traffic to the 1.7B fallback (p95 5.8 s).
  Neither number is meaningful. Next: FP8 primary alone on GPU0, NLI replicas on GPU1, and router
  thresholds calibrated from the 40 QPS sweep point (`rageval loadtest calibrate`).
- **M4 main, FP8 primary, fallback on the encoder GPU** (archived as
  `results/explorer/m4/main_fp8_fallback_on_encoder_gpu`, not used for claims). Static routing was
  stable at 50 QPS (p95 1.97 s). Adaptive routing was worse (p95 5.9-6.3 s): requests shed to the
  1.7B fallback and hedge duplicates ran on GPU1 next to the reranker and NLI replicas, verification
  rose from 220 ms to 1.8 s p50, and the contended fallback generated slower (874 ms p50) than the
  primary (332 ms). Hedges almost never won. Next: fallback co-located with the primary on GPU0
  (GPU1 encoder-only) and hedging off (the tail here is load-induced, where duplicates add load).
- **M4 main, model overflow with a co-located fallback** (archived as
  `results/explorer/m4/main_fp8_model_overflow`, not used for claims). Still worse than static
  (p95 about 5.7 s vs 2.0 s): the 1.7B on GPU0 slowed the primary (generation p50 409 -> 821 ms),
  so 45% of requests overflowed, and the fallback's weaker answers failed the cited-document check
  more often, which triggered the expensive repair search and backed up the verifier (p50 1.8 s).
  The static trace logs showed where the tail actually is: the slowest 10% spent 915 ms in
  verification (12.8 NLI pairs vs 3.8 for the fastest half) and 847 ms in generation.
  Next: stage-level fallbacks (fast verification path and shallower reranking when a stage is
  backed up or the request is late) with model overflow only above 2.5x normal concurrency.
- **M4 main, stage fallbacks plus model overflow** (archived as
  `results/explorer/m4/main_fp8_stage_plus_overflow`, not used for claims). Stage fallbacks cut
  reranking p95 476 -> 109 ms and verification p95 1179 -> 252 ms, but 30% of requests still
  overflowed to the co-located 1.7B and generation p95 doubled (970 -> 1962 ms), so end-to-end p95
  was flat (2.0 vs 2.1-2.2 s). Final policy: stage fallbacks for latency, the model fallback only
  as failover (breaker + retry on primary errors and timeouts).
