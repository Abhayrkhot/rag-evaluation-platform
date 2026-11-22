"""Render the README's result sections from results/ so every number is traceable.

    python scripts/gen_readme.py          rewrite the generated blocks in README.md
    python scripts/gen_readme.py --check  exit 1 if README.md is out of date (CI)

Generated blocks sit between `<!-- BEGIN:<name> -->` and `<!-- END:<name> -->` markers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
README = ROOT / "README.md"


def load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def pct(ci: dict[str, float] | None, digits: int = 1) -> str:
    if not ci or ci.get("mean") is None:
        return "n/a"
    return f"{100 * ci['mean']:.{digits}f}%"


def pct_ci(ci: dict[str, float] | None, digits: int = 1) -> str:
    if not ci or ci.get("mean") is None:
        return "n/a"
    return (
        f"{100 * ci['mean']:.{digits}f}% [{100 * ci['lo']:.{digits}f}, {100 * ci['hi']:.{digits}f}]"
    )


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def model_short(m: str | None) -> str:
    return (m or "").split("/")[-1]


# ---------------------------------------------------------------- retrieval


def retrieval_block() -> str | None:
    sel = load(RES / "explorer/m1/selected.json")
    test = load(RES / "explorer/m1/final_test.json")
    full = load(RES / "explorer/m1/final_full.json")
    build = load(RES / "explorer/m1/index_build.json")
    if not (sel and test and full):
        return None
    s = sel["selected"]
    rows = []
    for r_test, r_full in zip(test["results"], full["results"], strict=True):
        fam = r_test["config"]["family"]
        rows.append(
            [
                fam,
                pct_ci(r_test["recall@10"]),
                pct(r_test["all_recall@10"]),
                f"{r_test['ndcg@10']['mean']:.3f}",
                pct_ci(r_full["recall@10"]),
                pct(r_full["all_recall@10"]),
            ]
        )
    cfg = {k: v for k, v in s["config"].items() if k != "family"}
    lines = [
        f"Embedder `{s['embedder_slug']}`, reranker `{model_short(s['reranker'])}`, selected on dev "
        f"(dev Recall@10 {100 * s['recall@10']:.1f}%) with config `{json.dumps(cfg, sort_keys=True)}`.",
        "",
        table(
            [
                "Pipeline",
                "Test Recall@10 [95% CI]",
                "Test both-hops@10",
                "Test nDCG@10",
                "Full Recall@10 [95% CI]",
                "Full both-hops@10",
            ],
            rows,
        ),
        "",
        f"Test: {test['results'][0]['n']} answerable questions. Full: {full['results'][0]['n']} answerable questions. "
        f"HNSW vs exact search top-10 overlap: {100 * test['ann_recall@10']:.1f}%.",
    ]
    if build:
        dense = [v for k, v in build.items() if isinstance(v, dict) and "embedder" in v]
        if dense:
            lines += [
                "",
                "Index build: "
                + "; ".join(
                    f"{model_short(d['embedder'])} {d['docs_per_second']:.0f} docs/s" for d in dense
                )
                + f"; BM25 {build['bm25']['bm25_build_seconds']:.0f} s.",
            ]
    return "\n".join(lines)


# ---------------------------------------------------------------- two-hop retrieval


def multihop_block() -> str | None:
    t = load(RES / "explorer/m1/multihop_test.json")
    f = load(RES / "explorer/m1/multihop_full.json")
    d = load(RES / "explorer/m1/multihop_dev.json")
    if not (t and f and d):
        return None
    b = d["best"]
    rows = []
    for name, x in (("test", t), ("full", f)):
        one, two = x["baseline_hop1"], x["best"]
        rows.append(
            [
                name,
                str(one["recall@10"].get("n", "")),
                pct_ci(one["recall@10"]),
                pct_ci(two["recall@10"]),
                f"{pct(one['recall@10/bridge'])} -> {pct(two['recall@10/bridge'])}",
                f"{pct(one['all_recall@10'])} -> {pct(two['all_recall@10'])}",
                f"{100 * two['followup_rate']:.0f}%",
            ]
        )
    return "\n".join(
        [
            "Second hop for bridge questions: the generator (Qwen3-8B) reads the top "
            f"{b['plan_k']} first-hop passages and names the Wikipedia article still missing; a second "
            f"hybrid + rerank pass runs on that title and its top {b['take']} "
            f"{'hit is' if b['take'] == 1 else 'hits are'} spliced in after the first {b['keep']} "
            "first-hop hits (planner context and splice chosen on dev; follow-up issued = planner "
            "output parsed as a title).",
            "",
            table(
                [
                    "Suite",
                    "Questions",
                    "Single hop Recall@10",
                    "Two hop Recall@10",
                    "Bridge Recall@10",
                    "Both hops@10",
                    "Follow-up issued",
                ],
                rows,
            ),
        ]
    )


# ---------------------------------------------------------------- grounding / prompts


STAGES = [
    ("s0_baseline", 'S0 naive RAG: hybrid top-10, no reranking, plain "cite [n]" prompt'),
    ("s1a_rerank", "S1a + cross-encoder reranking"),
    ("s1b_2hop", "S1b + two-hop retrieval"),
    ("zero_shot", "S2 + citation and grounding rules (zero-shot)"),
    ("__best__", "S3 + best prompt strategy (picked on dev)"),
    ("s4_verified", "S4 + NLI citation repair and one revision round (balanced)"),
    ("s4_strict", "S4 strict: NLI and MiniCheck must both accept"),
]


def _m(s: dict[str, Any], k: str) -> str:
    return pct_ci(s.get(k)) if s.get(k) else "n/a"


def grounding_block() -> str | None:
    rep = load(RES / "explorer/m2/test_report.json")
    choice = load(RES / "explorer/m2/test_choices.json")
    if not (rep and choice):
        return None
    S = rep["summary"]
    rows = []
    for key, label in STAGES:
        name = choice["best_prompt"] if key == "__best__" else key
        if name not in S:
            continue
        x = S[name]
        rows.append(
            [
                label + (f" (`{name}`)" if key == "__best__" else ""),
                _m(x, "citation_precision"),
                _m(x, "citation_recall"),
                _m(x, "correct"),
                pct(x.get("abstain_unanswerable")),
                pct(x.get("context_recall")),
            ]
        )
    return "\n".join(
        [
            f"Test suite, {S['s0_baseline']['n']} questions ({S['s0_baseline']['n_answerable']} answerable, "
            f"{S['s0_baseline']['n_unanswerable']} unanswerable). Citation precision and recall are ALCE's, judged by "
            "TRUE (T5-11B NLI) over answered questions; correctness is gpt-oss-120b against the HotpotQA answer on "
            "answerable questions; abstention is on unanswerable questions. Generator: Qwen3-8B, greedy.",
            "",
            table(
                [
                    "Stage",
                    "Citation precision",
                    "Citation recall",
                    "Correct (answerable)",
                    "Abstains (unanswerable)",
                    "Gold docs in context",
                ],
                rows,
            ),
        ]
    )


def prompts_block() -> str | None:
    rep = load(RES / "explorer/m2/test_report.json")
    choice = load(RES / "explorer/m2/test_choices.json")
    if not (rep and choice):
        return None
    S = rep["summary"]
    rows = []
    for name in ("zero_shot", "few_shot", "cot"):
        x = S[name]
        rows.append(
            [
                name,
                str(x.get("unsupported_claims_total")),
                str(x.get("claims_total")),
                pct_ci(x.get("unsupported_rate")),
                pct(x.get("answers_with_unsupported")),
                _m(x, "citation_precision"),
                _m(x, "correct"),
                pct(x.get("abstain_unanswerable")),
            ]
        )
    tests = []
    for r in rep["paired"]:
        if (
            r["a"] == "zero_shot"
            and r["b"] in ("few_shot", "cot")
            and r["metric"]
            in (
                "unsupported_count",
                "has_unsupported",
                "unsupported_rate",
                "correct",
                "citation_precision",
                "abstain_unanswerable",
            )
        ):
            p = r["p_mcnemar"] if r["p_mcnemar"] is not None else r["p_wilcoxon"]
            tests.append(
                [
                    f"{r['a']} vs {r['b']}",
                    r["metric"],
                    str(r["n"]),
                    f"{r['mean_a']:.3f} -> {r['mean_b']:.3f}",
                    f"{r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}]",
                    f"{p:.2g}",
                    f"{r['p_holm']:.2g}",
                ]
            )
    z, b = S["zero_shot"], S[choice["best_prompt"]]
    cut = 1 - b["unsupported_claims_total"] / max(z["unsupported_claims_total"], 1)
    return "\n".join(
        [
            "Same 500 questions and identical retrieved contexts for every strategy; unsupported claims are "
            "statements gpt-oss-120b judges not fully supported by the documents the generator saw.",
            "",
            table(
                [
                    "Strategy",
                    "Unsupported claims",
                    "Claims",
                    "Unsupported rate [95% CI]",
                    "Answers with one+",
                    "Citation precision",
                    "Correct",
                    "Abstains (unanswerable)",
                ],
                rows,
            ),
            "",
            f"Unsupported claims, `zero_shot` -> `{choice['best_prompt']}`: {z['unsupported_claims_total']} -> "
            f"{b['unsupported_claims_total']} ({100 * cut:.0f}% fewer over the suite). Part of that is fewer and more "
            "cautious claims: on the questions both strategies answered, the per-claim unsupported rate falls less "
            "and the difference is not significant (paired `unsupported_rate` row below), and CoT abstains more, "
            "on unanswerable questions (good) and on answerable ones (bad).",
            "",
            "Paired tests (bootstrap CI on the per-question difference; McNemar for binary metrics, Wilcoxon otherwise; Holm across the family):",
            "",
            table(["Comparison", "Metric", "n", "Mean", "Delta [95% CI]", "p", "p (Holm)"], tests),
        ]
    )


# ---------------------------------------------------------------- serving


def serving_block() -> str | None:
    cands = (
        sorted((RES / "explorer/m4").glob("main_*/compare_q50.json"))
        if (RES / "explorer/m4").exists()
        else []
    )
    if not cands:
        return None
    c = load(cands[-1])
    run = cands[-1].parent.name
    rows = []
    for q in ("p50", "p90", "p95", "p99"):
        x = c[q]
        rows.append(
            [
                q,
                f"{x['base_ms'] / 1e3:.2f} s",
                f"{x['treat_ms'] / 1e3:.2f} s",
                f"{100 * x['reduction']:.0f}% [{100 * x['reduction_lo']:.0f}, {100 * x['reduction_hi']:.0f}]",
                ", ".join(f"{v / 1e3:.2f}" for v in x["base_per_seed_ms"])
                + " / "
                + ", ".join(f"{v / 1e3:.2f}" for v in x["treat_per_seed_ms"]),
            ]
        )
    tr = c["treat"]
    share = ", ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(tr["route_share"].items()))
    deg = (
        ", ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(tr["degraded_share"].items())) or "none"
    )
    lines = [
        f"Open-loop Poisson arrivals at 50 QPS, {c['seeds']} seeds per mode, alternating order, 30 s warm-up excluded (`{run}`). "
        "Static: every request takes the full serving pipeline on the primary model. Fallback routing: "
        "stage fallbacks (rerank 15 instead of 30 candidates, or verify only the cited documents and prune "
        "what fails) when that stage is backed up or the request is already late, model overflow to Qwen3-1.7B "
        "above 2.5x normal primary concurrency, and the circuit breaker / retry; every threshold calibrated "
        "from the 40 QPS sweep point, not from these runs (`calibration.json`).",
        "",
        table(
            [
                "Latency",
                "Static",
                "Fallback routing",
                "Reduction [95% CI, block bootstrap]",
                "Per seed (static / routed), s",
            ],
            rows,
        ),
        "",
        f"Requests per mode: {c['base']['requests']} static, {tr['requests']} routed; errors {100 * c['base']['error_rate']:.2f}% / "
        f"{100 * tr['error_rate']:.2f}%. Routed traffic: {share}. Degraded stages: {deg}.",
    ]
    prof = []
    for suite in ("test", "full"):
        sp = load(RES / f"explorer/m4/prep/serving_{suite}__gte-reranker-modernbert-base.json")
        if sp:
            r = next(
                (x for x in sp["results"] if x["config"].get("family") == "serve_w07_d30"), None
            )
            if r:
                prof.append(f"{suite} {pct(r['recall@10'])}")
    par = load(cands[-1].parent / "parity.json")
    if prof or par:
        lines += [
            "",
            "Serving profile retrieval (single hop, gte-reranker-modernbert-base at depth 30): Recall@10 "
            + ", ".join(prof)
            + (
                f". vLLM-served encoders vs the offline HF models: embedding cosine >= {par['embed_cosine_min']:.4f}, "
                f"rerank Spearman {par['rerank_spearman_mean']:.3f}, top-10 overlap {100 * par['rerank_top10_overlap_mean']:.0f}%."
                if par
                else "."
            ),
        ]
    if c.get("sweep"):
        lines += [
            "",
            "Capacity sweep with static routing (sizing):",
            "",
            table(
                ["Offered QPS", "p50", "p95", "p99", "Errors"],
                [
                    [
                        f"{x['offered_qps']:.1f}",
                        f"{x['p50_ms'] / 1e3:.2f} s",
                        f"{x['p95_ms'] / 1e3:.2f} s",
                        f"{x['p99_ms'] / 1e3:.2f} s",
                        f"{100 * x['error_rate']:.1f}%",
                    ]
                    for x in c["sweep"]
                ],
            ),
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------- headline


def headline_block() -> str | None:
    rows: list[list[str]] = []
    mt, mf = (
        load(RES / "explorer/m1/multihop_test.json"),
        load(RES / "explorer/m1/multihop_full.json"),
    )
    if mt and mf:
        rows.append(
            [
                "Recall@10, hybrid FAISS + BM25 + reranking, 100K docs",
                "91%",
                f"{pct(mf['best']['recall@10'])} two-hop / {pct(mf['baseline_hop1']['recall@10'])} single-hop "
                f"(7,018 questions); test suite {pct(mt['best']['recall@10'])} / {pct(mt['baseline_hop1']['recall@10'])}",
            ]
        )
    rep, ch = (
        load(RES / "explorer/m2/test_report.json"),
        load(RES / "explorer/m2/test_choices.json"),
    )
    if rep and ch:
        S = rep["summary"]
        rows.append(
            [
                "Citation precision, naive RAG -> final pipeline (500-question suite)",
                "76% -> 94%",
                f"{pct(S['s0_baseline']['citation_precision'])} -> {pct(S['s4_verified']['citation_precision'])} "
                f"(balanced, correct {pct(S['s4_verified']['correct'])}) / {pct(S['s4_strict']['citation_precision'])} "
                f"(strict, correct {pct(S['s4_strict']['correct'])}); naive correct {pct(S['s0_baseline']['correct'])}",
            ]
        )
        z, b = S["zero_shot"], S[ch["best_prompt"]]
        cut = 1 - b["unsupported_claims_total"] / max(z["unsupported_claims_total"], 1)
        p = next(
            (
                r
                for r in rep["paired"]
                if r["a"] == "zero_shot"
                and r["b"] == ch["best_prompt"]
                and r["metric"] == "unsupported_count"
            ),
            None,
        )
        pr = next(
            (
                r
                for r in rep["paired"]
                if r["a"] == "zero_shot"
                and r["b"] == ch["best_prompt"]
                and r["metric"] == "unsupported_rate"
            ),
            None,
        )
        rows.append(
            [
                f"Unsupported claims, zero-shot vs {ch['best_prompt']} (paired)",
                "-32%",
                f"-{100 * cut:.0f}% ({z['unsupported_claims_total']} -> {b['unsupported_claims_total']})"
                + (f", Wilcoxon p (Holm) {p['p_holm']:.2g}" if p else "")
                + (
                    f"; per-claim rate on questions both answered {100 * pr['mean_a']:.1f}% -> "
                    f"{100 * pr['mean_b']:.1f}% (p Holm {pr['p_holm']:.2g})"
                    if pr
                    else ""
                ),
            ]
        )
    cands = (
        sorted((RES / "explorer/m4").glob("main_*/compare_q50.json"))
        if (RES / "explorer/m4").exists()
        else []
    )
    if cands:
        c = load(cands[-1])
        x = c["p95"]
        rows.append(
            [
                "p95 latency at 50 QPS, static vs fallback routing",
                "-28%",
                f"-{100 * x['reduction']:.0f}% ({x['base_ms'] / 1e3:.2f} s -> {x['treat_ms'] / 1e3:.2f} s; "
                f"95% CI {100 * x['reduction_lo']:.0f} to {100 * x['reduction_hi']:.0f}%)",
            ]
        )
    if not rows:
        return None
    return table(["Claim", "Target", "Measured"], rows)


# ---------------------------------------------------------------- limitations


def limitations_block() -> str | None:
    items = [
        "One dataset. Every number is on a 100K-paragraph corpus built from HotpotQA (Wikipedia intro "
        "paragraphs). It contains each validation question's hard negatives but is far smaller than full "
        "Wikipedia, and Qwen3 has likely seen HotpotQA, so parametric knowledge can leak into answers. The "
        "unsupported-claims judge sees only the retrieved documents, so leaked facts count against the "
        "pipeline rather than for it.",
        "Model judges, no human labels. Citation metrics use TRUE (T5-11B NLI), as ALCE does; unsupported claims "
        "and correctness use gpt-oss-120b. Both are models with their own errors. The pipeline's verifier "
        "(DeBERTa NLI, optionally with MiniCheck) is a different model from both judges, and the strict "
        "operating point's gain also shows up in the gpt-oss judge, but no human agreement study was run.",
        "Small dev set. Every choice (fusion, depth, splice, K, prompt, verifier threshold) was made on 250 dev "
        "questions (200 answerable), so the dev picks are noisy; the test suite and the 7,018-question full set "
        "were each scored once with the frozen choices.",
        "Precision vs correctness. The strict citation operating point prunes more aggressively and answers "
        "fewer questions correctly than the balanced one; both are reported.",
        "Serving profile differs from the best offline pipeline. Two-hop retrieval, 10-document contexts and "
        "the revision round cost extra LLM calls, so the 50 QPS deployment serves a latency profile (single "
        "hop, ModernBERT reranker at depth 30, 5 documents, NLI repair without revision) whose retrieval recall "
        "is reported in its own section.",
        "Hardware-specific latency. The load tests run on two H200s (one primary Qwen3-8B, one GPU shared by the "
        "fallback model and the encoders) with a single-node API; the fallback GPU is idle under static routing, "
        "which is part of what routing exploits. Results will differ on other hardware.",
    ]
    return "\n".join(f"- {x}" for x in items)


# ---------------------------------------------------------------- render


BLOCKS = {
    "headline": headline_block,
    "retrieval": retrieval_block,
    "multihop": multihop_block,
    "grounding": grounding_block,
    "prompts": prompts_block,
    "serving": serving_block,
    "limitations": limitations_block,
}


def render(text: str) -> str:
    for name, fn in BLOCKS.items():
        body = fn() or "_Pending: results not generated yet._"
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(\n<!-- END:{name} -->)", re.S)
        text = pat.sub(lambda m, b=body: m.group(1) + b + m.group(2), text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cur = README.read_text()
    new = render(cur)
    if args.check:
        if new != cur:
            print("README.md is out of date: run python scripts/gen_readme.py", file=sys.stderr)
            return 1
        return 0
    README.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
