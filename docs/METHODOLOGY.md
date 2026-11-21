# Methodology

How every number in the README is produced. Code paths are given so each definition can be
checked against the implementation.

## Corpus and suites (`src/rageval/data/hotpotqa.py`)

- Source: HotpotQA distractor setting, `hotpotqa/hotpot_qa` pinned to revision
  `1908d6afbbead072334abe2965f91bd2709910ab`.
- Corpus: exactly 100,000 Wikipedia intro paragraphs, keyed by title. It holds every paragraph
  from the validation contexts (each question's 2 supporting paragraphs plus the 8 distractors
  HotpotQA mined from full Wikipedia with bigram TF-IDF, so hard negatives are present), topped up
  with paragraphs sampled from the train contexts, minus the supporting paragraphs of the
  unanswerable questions. 66,281 paragraphs come from validation, 33,719 from train.
- Suites, all from validation and disjoint:
  - `test` (500): 400 answerable questions stratified in HotpotQA's natural bridge/comparison ratio
    (320/80) and 100 unanswerable questions (80/20) whose supporting paragraphs were removed from
    the corpus. This is the evaluation suite.
  - `dev` (250, same composition): the only data used for any tuning decision.
  - `full` (7,018 answerable): every remaining answerable validation question, used for
    retrieval-only metrics with tight confidence intervals.
- 61 validation paragraphs appear with slightly different text in different questions; all
  differences checked are whitespace (non-breaking spaces). The first occurrence is kept.
- Load tests draw from a separate pool of 20,000 unique HotpotQA train questions plus the test
  suite, so repeated prompts cannot hit vLLM's prefix cache.

## Retrieval metrics (`src/rageval/eval/retrieval_metrics.py`)

Over answerable questions, with the HotpotQA supporting paragraphs as the relevant set:

- **Recall@k**: fraction of a question's supporting paragraphs in the top k, averaged over
  questions (the BEIR/TREC definition). Headline metric, k = 10.
- **All-recall@k**: 1 if every supporting paragraph is in the top k (both hops present).
- nDCG@10 and MRR@10 with binary relevance.
- 95% CIs: percentile bootstrap over questions, 10,000 resamples.

Every fusion weight, candidate depth, embedder and reranker choice is made on `dev`; the selected
configuration is then scored once on `test` and `full` (`rageval retrieval select`).

## Grounding metrics

Answers are split into sentence-level statements; each statement keeps the documents it cites
(`src/rageval/generation/citations.py`). Abstention sentences are not claims.

- **Citation recall / precision** (`src/rageval/eval/grounding.py`): the ALCE definitions
  (Gao et al., 2023) with ALCE's judge, the TRUE NLI model (`google/t5_xxl_true_nli_mixture`,
  T5-11B). A statement is supported when the concatenation of its cited documents entails it.
  A citation is precise unless the statement is unsupported, or the statement has several
  citations and this one alone does not entail it while the others still do (it is unnecessary).
  Scores are averaged per answer, then over answered questions.
- **Unsupported claims** (`src/rageval/judges/llm_judge.py`): gpt-oss-120b judges every statement
  against all documents the generator was shown; partly supported counts as unsupported.
  Reported as the total count over the suite, the rate per claim, and the share of answers with
  at least one.
- **Correctness**: gpt-oss-120b compares the response to the HotpotQA reference answer.
  A normalised string match is reported alongside as a deterministic cross-check.
- **Abstention**: on unanswerable questions, the share of responses that decline; on answerable
  questions, the false-abstention rate.

The judges are deliberately different models from anything inside the pipeline: the pipeline's
citation verifier is DeBERTa-v3-large NLI, the evaluator is TRUE T5-11B, and the LLM judge
(gpt-oss-120b) is from a different model family than the Qwen3 generator.

## Paired tests (`src/rageval/eval/stats.py`, `src/rageval/eval/report.py`)

Every compared configuration answers the same questions with the same retrieved contexts.

- Paired bootstrap on per-question differences (10,000 resamples): mean delta, 95% CI, p-value.
- Wilcoxon signed-rank on per-question differences (continuous metrics).
- Exact McNemar on paired binary outcomes (correct, has-unsupported-claim, abstained).
- Holm-Bonferroni correction across all tests reported in one comparison family.

## Load tests (`src/rageval/loadtest/loadgen.py`)

- Open loop: requests are released on a precomputed Poisson schedule at the target rate, whatever
  the server's speed, and latency is measured from each request's scheduled release time, so
  queueing delay is never hidden (no coordinated omission).
- 30 s warm-up excluded; each condition runs on a freshly started API process.
- Conditions are interleaved across seeds so cache and thermal state cannot favour one side.
