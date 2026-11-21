"""Aggregate judged rows into per-pipeline metrics with CIs, and paired comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from rageval.eval.stats import bootstrap_ci, compare, holm, ratio_bootstrap_ci

NAN = float("nan")


def _mean_ci(vals: Sequence[float]) -> dict[str, float]:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return bootstrap_ci(v).to_dict() if v else {"mean": NAN, "lo": NAN, "hi": NAN, "n": 0}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["abstained"] and not r.get("error")]
    s: dict[str, Any] = {
        "n": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "errors": sum(1 for r in rows if r.get("error")),
        "answered": len(answered),
        "correct": _mean_ci(
            [float(bool(r.get("judge_correct"))) for r in ans if r.get("judge_correct") is not None]
        ),
        "str_match": _mean_ci([float(r["str_match"]) for r in ans]),
        "abstain_unanswerable": _mean_ci([float(r["abstained"]) for r in unans]),
        "abstain_answerable": _mean_ci([float(r["abstained"]) for r in ans]),
        "context_recall": _mean_ci(
            [r["context_recall"] for r in ans if r.get("context_recall") is not None]
        ),
        "claims_per_answer": _mean_ci([float(r["n_claims"]) for r in answered]),
    }
    if any("cit_precision" in r for r in answered):
        s["citation_precision"] = _mean_ci(
            [r["cit_precision"] for r in answered if "cit_precision" in r]
        )
        s["citation_recall"] = _mean_ci([r["cit_recall"] for r in answered if "cit_recall" in r])
    judged = [r for r in answered if "n_unsupported" in r]
    if judged:
        s["unsupported_claims_total"] = int(sum(r["n_unsupported"] for r in judged))
        s["claims_total"] = int(sum(r["n_claims"] for r in judged))
        s["unparsed_claim_verdicts"] = int(sum(r.get("n_unparsed", 0) for r in judged))
        s["unsupported_rate"] = ratio_bootstrap_ci(
            [r["n_unsupported"] for r in judged], [r["n_claims"] for r in judged]
        ).to_dict()
        s["answers_with_unsupported"] = _mean_ci([float(r["n_unsupported"] > 0) for r in judged])
        # per question over the whole suite (abstentions contribute 0 claims)
        s["unsupported_per_question"] = _mean_ci(
            [float(r.get("n_unsupported", 0)) for r in rows if not r.get("error")]
        )
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    if lat:
        s["gen_latency_ms_p50"] = float(np.percentile(lat, 50))
        s["gen_latency_ms_p95"] = float(np.percentile(lat, 95))
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    if toks:
        s["completion_tokens_mean"] = float(np.mean(toks))
    for t in ("bridge", "comparison"):
        sub = [r for r in ans if r["qtype"] == t]
        if sub:
            s[f"correct/{t}"] = _mean_ci(
                [
                    float(bool(r.get("judge_correct")))
                    for r in sub
                    if r.get("judge_correct") is not None
                ]
            )
    return s


def by_pipeline(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r["pipeline"]].append(r)
    return dict(out)


def _vector(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        if r.get("error"):
            continue
        q = r["qid"]
        if metric == "unsupported_count":
            out[q] = float(r.get("n_unsupported", 0)) if not r["abstained"] else 0.0
        elif metric == "has_unsupported":
            out[q] = float(r.get("n_unsupported", 0) > 0)
        elif metric == "unsupported_rate":
            if not r["abstained"] and r.get("n_claims"):
                out[q] = r.get("n_unsupported", 0) / r["n_claims"]
        elif metric == "citation_precision":
            if "cit_precision" in r:
                out[q] = r["cit_precision"]
        elif metric == "citation_recall":
            if "cit_recall" in r:
                out[q] = r["cit_recall"]
        elif metric == "correct":
            if r["answerable"] and r.get("judge_correct") is not None:
                out[q] = float(bool(r["judge_correct"]))
        elif metric == "abstain_unanswerable":
            if not r["answerable"]:
                out[q] = float(r["abstained"])
        else:
            raise ValueError(metric)
    return out


BINARY = {"has_unsupported", "correct", "abstain_unanswerable"}


def paired(
    groups: dict[str, list[dict[str, Any]]],
    pairs: Sequence[tuple[str, str]],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    results = []
    for a, b in pairs:
        for m in metrics:
            va, vb = _vector(groups[a], m), _vector(groups[b], m)
            keys = sorted(set(va) & set(vb))
            if not keys:
                continue
            res = compare(m, a, [va[k] for k in keys], b, [vb[k] for k in keys], binary=m in BINARY)
            results.append(res.to_dict())
    # Holm correction across every test in this family (one family per call)
    pvals = [float(r["p_wilcoxon"] if r["p_mcnemar"] is None else r["p_mcnemar"]) for r in results]  # type: ignore[arg-type]
    for r, p in zip(results, holm(pvals) if pvals else [], strict=True):
        r["p_holm"] = p
    return results
