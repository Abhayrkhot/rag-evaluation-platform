"""Command line entry point: `rageval <group> <command>`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from rageval.io import read_json, run_meta, write_json, write_jsonl

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
data_app = typer.Typer(no_args_is_help=True, help="Corpus and suite construction")
index_app = typer.Typer(no_args_is_help=True, help="BM25 and FAISS indexes")
ret_app = typer.Typer(no_args_is_help=True, help="Offline retrieval experiments")
app.add_typer(data_app, name="data")
app.add_typer(index_app, name="index")
app.add_typer(ret_app, name="retrieval")


def _echo(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------- data


@data_app.command("build")
def data_build(
    out: Annotated[Path, typer.Option()] = Path("data/hotpot100k"),
    n_docs: int = 100_000,
    seed: int = 13,
    revision: Annotated[str | None, typer.Option(help="HF dataset revision to pin")] = None,
) -> None:
    """Build the 100K-document corpus and the test/dev/full suites from HotpotQA."""
    from rageval.data.hotpotqa import BuildConfig, build

    manifest = build(out, BuildConfig(n_docs=n_docs, seed=seed, revision=revision))
    _echo(manifest)


@data_app.command("load-pool")
def data_load_pool(
    out: Annotated[Path, typer.Option()] = Path("data/hotpot100k/load_pool.jsonl"),
    n: int = 20_000,
    seed: int = 13,
    revision: str | None = "1908d6afbbead072334abe2965f91bd2709910ab",
) -> None:
    """Unique questions for load tests (HotpotQA train), so repeated prompts cannot hit vLLM's
    prefix cache and flatter latency."""
    import random

    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", revision=revision)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    rows = [
        {"qid": ds[i]["id"], "question": ds[i]["question"].strip(), "source": "hotpotqa-train"}
        for i in idx[:n]
    ]
    write_jsonl(out, rows)
    _echo({"out": str(out), "n": len(rows)})


# ---------------------------------------------------------------- index


@index_app.command("build")
def index_build(
    corpus: Annotated[Path, typer.Option()] = Path("data/hotpot100k/corpus.jsonl"),
    out: Annotated[Path, typer.Option()] = Path("artifacts/index/hotpot100k"),
    embedder: Annotated[list[str] | None, typer.Option(help="embedding model(s)")] = None,
    base_url: Annotated[
        str | None, typer.Option(help="OpenAI-compatible embeddings server")
    ] = None,
    batch_size: int = 128,
    bm25: bool = True,
    k1: float = 0.9,
    b: float = 0.4,
) -> None:
    from rageval.corpus import Corpus
    from rageval.retrieval.dense import make_embedder
    from rageval.retrieval.experiments import build_bm25, build_dense, slug

    c = Corpus.load(corpus)
    report: dict[str, Any] = {"n_docs": len(c)}
    if bm25:
        report["bm25"] = build_bm25(c, out / "bm25", k1=k1, b=b)
    for model in embedder or []:
        emb = make_embedder(model, base_url, batch_size=batch_size)
        report[slug(model)] = build_dense(c, emb, out / "dense" / slug(model))
    _echo(report)


# ---------------------------------------------------------------- retrieval


@ret_app.command("candidates")
def ret_candidates(
    suite: Annotated[Path, typer.Option()],
    embedder: Annotated[str, typer.Option()],
    index_dir: Annotated[Path, typer.Option()] = Path("artifacts/index/hotpot100k"),
    out_dir: Annotated[Path, typer.Option()] = Path("artifacts/cands"),
    base_url: str | None = None,
    k: int = 200,
    ef_search: int = 256,
    batch_size: int = 128,
) -> None:
    from rageval.corpus import load_suite
    from rageval.retrieval.dense import make_embedder
    from rageval.retrieval.experiments import compute_candidates, slug

    qs = load_suite(suite)
    emb = make_embedder(embedder, base_url, batch_size=batch_size)
    rows = compute_candidates(qs, index_dir, emb, k=k, ef_search=ef_search)
    path = out_dir / f"{suite.stem}__{slug(embedder)}.jsonl"
    write_jsonl(path, rows)
    _echo({"out": str(path), "n": len(rows), "timing": rows[0].get("_timing") if rows else {}})


@ret_app.command("sweep")
def ret_sweep(
    suite: Annotated[Path, typer.Option()],
    candidates: Annotated[Path, typer.Option()],
    grid: Annotated[
        Path, typer.Option(help="YAML: {name: {param: [values]}} or a list of configs")
    ],
    out: Annotated[Path, typer.Option()],
    corpus: Annotated[Path, typer.Option()] = Path("data/hotpot100k/corpus.jsonl"),
    reranker: Annotated[str | None, typer.Option()] = None,
    rerank_base_url: str | None = None,
    rerank_cache_dir: Annotated[Path, typer.Option()] = Path("artifacts/rerank"),
    rerank_batch: int = 128,
    rows_out: Annotated[
        Path | None, typer.Option(help="per-question rows of the best config")
    ] = None,
) -> None:
    """Score a grid of fusion/rerank configs from cached candidates."""
    from rageval.corpus import Corpus, load_suite
    from rageval.retrieval.experiments import (
        RerankCache,
        ann_recall,
        candidate_timing,
        expand_grid,
        load_candidates,
        needed_pairs,
        score_config,
        slug,
    )

    qs = load_suite(suite)
    cands = load_candidates(candidates)
    spec = yaml.safe_load(grid.read_text())
    configs: list[dict[str, Any]] = []
    if isinstance(spec, list):
        configs = [dict(c) for c in spec]
    else:
        for name, g in spec.items():
            for c in expand_grid(g):
                c["family"] = name
                configs.append(c)
    rr: dict[tuple[str, str], float] | None = None
    rerank_meta: dict[str, Any] = {}
    if any(int(c.get("rerank_depth", 0)) > 0 for c in configs):
        if not reranker:
            raise typer.BadParameter("grid needs reranker scores: pass --reranker")
        from rageval.retrieval.rerank import make_reranker

        cache = RerankCache(
            rerank_cache_dir
            / f"{suite.stem}__{candidates.stem.split('__')[-1]}__{slug(reranker)}.jsonl"
        )
        need = needed_pairs(qs, cands, configs)
        missing = cache.missing(need)
        if missing:
            corp = Corpus.load(corpus)
            kw: dict[str, Any] = {} if rerank_base_url else {"batch_size": rerank_batch}
            model = make_reranker(reranker, rerank_base_url, **kw)
            secs = cache.fill(missing, model, {q.qid: q.question for q in qs}, corp)
            rerank_meta = {
                "pairs_scored": len(missing),
                "seconds": round(secs, 2),
                "pairs_per_second": round(len(missing) / max(secs, 1e-9), 1),
            }
        rr = cache.scores
        rerank_meta["reranker"] = reranker
    results = []
    best: tuple[float, float, dict[str, Any], list[dict[str, Any]]] | None = None
    for cfg in configs:
        summary, rows = score_config(qs, cands, cfg, rr)
        results.append(summary)
        key = (summary["recall@10"]["mean"], summary["ndcg@10"]["mean"])
        if best is None or key > best[:2]:
            best = (key[0], key[1], summary, rows)
    payload = {
        "meta": run_meta(
            suite=str(suite), candidates=str(candidates), grid=str(grid), rerank=rerank_meta
        ),
        "ann_recall@10": ann_recall(cands, 10),
        "candidate_timing": candidate_timing(candidates),
        "results": results,
        "best": best[2] if best else None,
    }
    write_json(out, payload)
    if rows_out and best:
        write_jsonl(rows_out, best[3])
    top = sorted(results, key=lambda r: -r["recall@10"]["mean"])[:8]
    _echo(
        [
            {
                "config": r["config"],
                "recall@10": round(r["recall@10"]["mean"], 4),
                "all@10": round(r["all_recall@10"]["mean"], 4),
                "ndcg@10": round(r["ndcg@10"]["mean"], 4),
            }
            for r in top
        ]
    )


@ret_app.command("score")
def ret_score(
    suite: Annotated[Path, typer.Option()],
    candidates: Annotated[Path, typer.Option()],
    config: Annotated[str, typer.Option(help="JSON config dict")],
    out: Annotated[Path, typer.Option()],
    reranker: Annotated[str | None, typer.Option()] = None,
    rerank_cache_dir: Annotated[Path, typer.Option()] = Path("artifacts/rerank"),
) -> None:
    """Score one config (e.g. the dev-selected one) on another suite. Rerank scores must be cached."""
    from rageval.corpus import load_suite
    from rageval.retrieval.experiments import RerankCache, load_candidates, score_config, slug

    qs = load_suite(suite)
    cands = load_candidates(candidates)
    cfg = json.loads(config)
    rr = None
    if int(cfg.get("rerank_depth", 0)) > 0:
        assert reranker, "--reranker required"
        rr = RerankCache(
            rerank_cache_dir
            / f"{suite.stem}__{candidates.stem.split('__')[-1]}__{slug(reranker)}.jsonl"
        ).scores
    summary, rows = score_config(qs, cands, cfg, rr)
    write_json(out, {"meta": run_meta(suite=str(suite), reranker=reranker), "summary": summary})
    write_jsonl(out.with_suffix(".rows.jsonl"), rows)
    _echo({k: v for k, v in summary.items() if k != "config"})


@ret_app.command("select")
def ret_select(
    sweeps: Annotated[list[Path], typer.Option(help="dev sweep JSON files")],
    out: Annotated[Path, typer.Option()],
    tolerance: float = 0.005,
) -> None:
    """Pick the dev-best config: highest recall@10; among configs within `tolerance` of it, the
    shallowest rerank depth (cheaper online), then higher nDCG@10. Also writes the final
    comparison grid (BM25, dense, fusion without reranking, selected) for test/full."""
    rows = []
    for p in sweeps:
        d = read_json(p)
        emb = d["meta"]["candidates"].split("__")[-1].removesuffix(".jsonl")
        rr = (d["meta"].get("rerank") or {}).get("reranker")
        for r in d["results"]:
            rows.append(
                {
                    "embedder_slug": emb,
                    "reranker": rr,
                    "config": r["config"],
                    "recall@10": r["recall@10"]["mean"],
                    "ndcg@10": r["ndcg@10"]["mean"],
                    "all_recall@10": r["all_recall@10"]["mean"],
                }
            )
    best = max(r["recall@10"] for r in rows)
    near = [r for r in rows if r["recall@10"] >= best - tolerance]
    # shallowest reranking first (0 = no reranking, cheapest), then nDCG@10
    near.sort(key=lambda r: (int(r["config"].get("rerank_depth", 0) or 0), -r["ndcg@10"]))
    pick = near[0]
    cfg = {k: v for k, v in pick["config"].items() if k != "family"}
    no_rr = {k: v for k, v in cfg.items() if k not in ("rerank_depth", "rerank_blend")}
    final = [
        {"family": "bm25", "fusion": "bm25"},
        {"family": "dense", "fusion": "dense", "dense_index": cfg.get("dense_index", "hnsw")},
        {"family": "hybrid", **no_rr},
        {"family": "hybrid+rerank", **cfg},
    ]
    payload = {
        "selected": pick,
        "best_dev_recall@10": best,
        "tolerance": tolerance,
        "final_grid": final,
        "dev_top": sorted(rows, key=lambda r: -r["recall@10"])[:15],
    }
    write_json(out, payload)
    out.with_suffix(".grid.yaml").write_text(yaml.safe_dump(final, sort_keys=True))
    _echo({k: payload[k] for k in ("selected", "best_dev_recall@10")})


@ret_app.command("materialize")
def ret_materialize(
    selected: Annotated[Path, typer.Option()],
    models_template: Annotated[Path, typer.Option()],
    pipelines_template: Annotated[Path, typer.Option()],
    out_dir: Annotated[Path, typer.Option()],
    ef_search: int = 256,
    multihop: Annotated[
        Path | None, typer.Option(help="dev multihop grid; adds the best splice")
    ] = None,
    choices: Annotated[
        Path | None, typer.Option(help="dev choices (gen choose) for test templates")
    ] = None,
) -> None:
    """Fill the dev-selected embedder, reranker and retrieval config into config templates
    (__EMBEDDER__, __RERANKER__, __RETRIEVAL__), and keep the filled copies with the results."""
    from rageval.retrieval.dense import FORMATS
    from rageval.retrieval.experiments import slug

    sel = read_json(selected)["selected"]
    by_slug = {slug(m): m for m in FORMATS}
    embedder = by_slug[sel["embedder_slug"]]
    cfg = {k: v for k, v in sel["config"].items() if k not in ("family", "dense_index")}
    cfg["rerank"] = int(cfg.get("rerank_depth", 0) or 0) > 0
    cfg["ef_search"] = ef_search
    if multihop is not None:
        mh = read_json(multihop)
        b = mh["best"]
        gain = b["recall@10"]["mean"] - mh["baseline_hop1"]["recall@10"]["mean"]
        if gain > 0:
            cfg.update(
                hops=2,
                plan_k=b["plan_k"],
                hop2_keep=b["keep"],
                hop2_take=b["take"],
                hop2_depth=mh["meta"]["hop2_depth"],
            )
    retrieval = json.dumps(cfg, sort_keys=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = (
        models_template.read_text()
        .replace("__EMBEDDER__", embedder)
        .replace("__RERANKER__", sel["reranker"] or "null")
    )
    m = m.replace("dense_index: hnsw", f"dense_index: {sel['config'].get('dense_index', 'hnsw')}")
    (out_dir / "models.yaml").write_text(m)
    ptxt = pipelines_template.read_text().replace("__RETRIEVAL__", retrieval)
    if choices is not None:
        ch = read_json(choices)
        for key, val in (
            ("__K__", ch["context_k"]),
            ("__BEST__", ch["best_prompt"]),
            ("__VPROMPT__", ch["verify_prompt"]),
            ("__POLICY__", ch["verify_policy"]),
            ("__TAU__", ch["verify_tau"]),
        ):
            ptxt = ptxt.replace(key, str(val))
    (out_dir / "pipelines.yaml").write_text(ptxt)
    _echo({"embedder": embedder, "reranker": sel["reranker"], "retrieval": cfg})


@ret_app.command("multihop")
def ret_multihop(
    suite: Annotated[Path, typer.Option()],
    models_cfg: Annotated[Path, typer.Option("--models")],
    selected: Annotated[Path, typer.Option()],
    llm_url: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    llm_model: str = "Qwen/Qwen3-8B",
    plan_k: Annotated[list[int] | None, typer.Option()] = None,
    keep: Annotated[list[int] | None, typer.Option()] = None,
    take: Annotated[list[int] | None, typer.Option()] = None,
    hop2_depth: int = 30,
    candidates: Annotated[Path | None, typer.Option(help="cached first-hop candidates")] = None,
    rerank_cache_dir: Annotated[Path, typer.Option()] = Path("artifacts/rerank"),
    rows_out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Two-hop retrieval: first hop = the selected single-hop ranking (from the M1 caches when
    given), planner LLM names the missing article, second hybrid+rerank hop, splice grid."""
    import asyncio

    from rageval.corpus import load_suite
    from rageval.eval.retrieval_metrics import per_question
    from rageval.eval.stats import bootstrap_ci
    from rageval.llm.client import QWEN3_NO_THINK, ChatClient
    from rageval.pipeline import ModelsConfig, load_retriever
    from rageval.retrieval.experiments import RerankCache, load_candidates, rank_one, slug
    from rageval.retrieval.hybrid import RetrievalConfig
    from rageval.retrieval.multihop import merge_hops, second_hop

    models = ModelsConfig.model_validate(yaml.safe_load(models_cfg.read_text()))
    sel = read_json(selected)["selected"]
    scfg = {k: v for k, v in sel["config"].items() if k != "family"}
    rcfg = RetrievalConfig.model_validate(
        {k: v for k, v in scfg.items() if k != "dense_index"}
        | {"rerank": int(scfg.get("rerank_depth", 0) or 0) > 0, "ef_search": 256}
    )
    qs = load_suite(suite)
    retriever = load_retriever(models, rcfg)
    if candidates is not None:
        cands = load_candidates(candidates)
        rr = None
        if rcfg.rerank:
            rr = RerankCache(
                rerank_cache_dir
                / f"{suite.stem}__{candidates.stem.split('__')[-1]}__{slug(models.reranker or '')}.jsonl"
            ).scores
        hop1 = [rank_one(cands[q.qid], scfg, rr, q.qid, final_k=100) for q in qs]
    else:
        hop1 = []
        for s in range(0, len(qs), 256):
            hop1 += [
                r.doc_ids
                for r in retriever.retrieve_batch(
                    [q.question for q in qs[s : s + 256]], final_k=100
                )
            ]
    params = QWEN3_NO_THINK if "qwen3" in llm_model.lower() else {}
    ans = [i for i, q in enumerate(qs) if q.answerable]

    def score(lists: list[list[str]]) -> dict[str, Any]:
        rows = [per_question(lists[i], qs[i].gold_doc_ids, ks=(5, 10)) for i in ans]
        outd: dict[str, Any] = {
            k: bootstrap_ci([r[k] for r in rows]).to_dict()
            for k in ("recall@10", "all_recall@10", "recall@5", "ndcg@10")
        }
        for t in ("bridge", "comparison"):
            sub = [r["recall@10"] for i, r in zip(ans, rows, strict=True) if qs[i].qtype == t]
            if sub:
                outd[f"recall@10/{t}"] = bootstrap_ci(sub).to_dict()
        return outd

    result: dict[str, Any] = {"baseline_hop1": score(hop1), "grid": []}
    best: tuple[float, dict[str, Any], list[Any]] | None = None
    for pk in plan_k or [5]:
        client = ChatClient(llm_url, llm_model, timeout=300)  # one per event loop
        mh = asyncio.run(
            second_hop(
                retriever, client, [q.question for q in qs], hop1, rcfg, pk, hop2_depth, 128, params
            )
        )
        fu_rate = sum(1 for m in mh if m.followup) / len(mh)
        for kp in keep or [3]:
            for tk in take or [3]:
                merged = [merge_hops(m.hop1, m.hop2, kp, tk) if m.hop2 else m.hop1 for m in mh]
                sc = score(merged)
                entry = {"plan_k": pk, "keep": kp, "take": tk, "followup_rate": fu_rate, **sc}
                result["grid"].append(entry)
                key = sc["recall@10"]["mean"]
                if best is None or key > best[0]:
                    best = (
                        key,
                        entry,
                        [
                            (qs[i].qid, m.followup, m.hop1[:10], m.hop2[:5], merged[i][:10])
                            for i, m in enumerate(mh)
                        ],
                    )
        result.setdefault("planner", []).append(
            {
                "plan_k": pk,
                "followup_rate": fu_rate,
                "planner_ms_mean": sum(m.planner_ms for m in mh) / len(mh),
                "examples": [(m.question, m.followup) for m in mh[:12]],
            }
        )
    assert best is not None
    result["best"] = best[1]
    result["meta"] = run_meta(
        suite=str(suite),
        selected=sel,
        llm=llm_model,
        hop2_depth=hop2_depth,
        hop1_from_cache=candidates is not None,
    )
    write_json(out, result)
    if rows_out:
        write_jsonl(
            rows_out,
            (
                {"qid": a, "followup": b, "hop1": c, "hop2": d, "merged": e}
                for a, b, c, d, e in best[2]
            ),
        )
    _echo(
        {
            "baseline_recall@10": result["baseline_hop1"]["recall@10"],
            "best": {k: (v["mean"] if isinstance(v, dict) else v) for k, v in best[1].items()},
        }
    )


# ---------------------------------------------------------------- generation / judging

gen_app = typer.Typer(no_args_is_help=True, help="Offline generation and judging")
app.add_typer(gen_app, name="gen")


@gen_app.command("run")
def gen_run(
    suite: Annotated[Path, typer.Option()],
    pipelines: Annotated[Path, typer.Option(help="YAML with base + pipelines")],
    out: Annotated[Path, typer.Option()],
    only: Annotated[list[str] | None, typer.Option(help="subset of pipeline names")] = None,
    models_cfg: Annotated[Path | None, typer.Option("--models")] = None,
    generator_url: str | None = None,
    concurrency: int = 128,
    limit: int | None = None,
) -> None:
    """Retrieve + generate (+ verify) for every pipeline, write one JSONL record per answer."""
    import asyncio

    from rageval.corpus import load_suite
    from rageval.eval.gen_runner import apply_verifier, generate, retrieve_all
    from rageval.llm.client import QWEN3_NO_THINK, ChatClient
    from rageval.pipeline import ModelsConfig, load_pipelines, load_retriever

    models = ModelsConfig.model_validate(
        yaml.safe_load(models_cfg.read_text()) if models_cfg else {}
    )
    if generator_url:
        models.generator_url = generator_url
    cfgs = load_pipelines(pipelines)
    if only:
        cfgs = [c for c in cfgs if c.name in only]
    qs = load_suite(suite)[:limit] if limit else load_suite(suite)
    retriever = load_retriever(models)
    params = QWEN3_NO_THINK if "qwen3" in models.generator.lower() else {}
    client = ChatClient(models.generator_url, models.generator)
    retrieved = retrieve_all(retriever, qs, cfgs, client=client, params=params)
    records = asyncio.run(
        generate(qs, retriever.corpus, retrieved, cfgs, client, concurrency, params)
    )
    prob = None
    if any(c.verify != "none" for c in cfgs):
        from rageval.judges.nli import ClassNLI

        prob = ClassNLI(models.verifier).prob
    apply_verifier(records, retriever.corpus, prob, client=client, params=params)
    wall = records[0].pop("_wall_seconds", None) if records else None
    write_jsonl(out, records)
    meta = run_meta(
        suite=str(suite),
        pipelines=[c.name for c in cfgs],
        models=models.model_dump(),
        gen_wall_seconds=wall,
    )
    write_json(out.with_suffix(".meta.json"), meta)
    _echo(
        {
            "out": str(out),
            "records": len(records),
            "wall_s": wall,
            "errors": sum(1 for r in records if r.get("error")),
        }
    )


@gen_app.command("judge-nli")
def gen_judge_nli(
    records: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    corpus: Annotated[Path, typer.Option()] = Path("data/hotpot100k/corpus.jsonl"),
    batch: int = 16,
    model: Annotated[
        str | None, typer.Option(help="override the judge model (smoke tests)")
    ] = None,
    kind: Annotated[
        str, typer.Option(help="true (seq2seq TRUE) | class (NLI classifier)")
    ] = "true",
) -> None:
    """ALCE citation precision/recall with the TRUE T5-11B NLI judge."""
    from rageval.corpus import Corpus
    from rageval.eval.judge_runner import nli_citations
    from rageval.io import read_jsonl
    from rageval.judges.nli import ClassNLI, TrueNLI

    recs = list(read_jsonl(records))
    entail: Any
    if kind == "true":
        tj = TrueNLI(model or TrueNLI.MODEL, batch_size=batch)
        entail, name = tj.entail, tj.model_name
    else:
        cj = ClassNLI(model or ClassNLI.MODEL, batch_size=batch)

        def class_entail(p: Any, h: Any) -> list[bool]:
            return [bool(x > 0.5) for x in cj.prob(p, h)]

        entail, name = class_entail, cj.model_name
    res = nli_citations(recs, Corpus.load(corpus), entail)
    meta = {**res.pop("_meta"), "model": name}
    write_json(out, {"meta": run_meta(records=str(records), **meta), "scores": res})
    _echo(meta)


@gen_app.command("judge-llm")
def gen_judge_llm(
    records: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    llm_url: Annotated[str, typer.Option(help="judge LLM endpoint")],
    corpus: Annotated[Path, typer.Option()] = Path("data/hotpot100k/corpus.jsonl"),
    llm_model: str = "openai/gpt-oss-120b",
    reasoning_effort: str = "low",
    concurrency: int = 256,
) -> None:
    """Unsupported claims and answer correctness with an LLM judge."""
    import asyncio

    from rageval.corpus import Corpus
    from rageval.eval.judge_runner import llm_judgments
    from rageval.io import read_jsonl
    from rageval.judges.llm_judge import LLMJudge
    from rageval.llm.client import ChatClient

    recs = list(read_jsonl(records))
    lj = LLMJudge(
        ChatClient(llm_url, llm_model, timeout=900),
        concurrency=concurrency,
        reasoning_effort=reasoning_effort,
    )
    res = asyncio.run(llm_judgments(recs, Corpus.load(corpus), lj))
    meta = {**res.pop("_meta"), "model": llm_model, "reasoning_effort": reasoning_effort}
    write_json(out, {"meta": run_meta(records=str(records), **meta), "judgments": res})
    _echo(meta)


@gen_app.command("merge")
def gen_merge(
    records: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    nli: Annotated[Path | None, typer.Option()] = None,
    llm: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Join generation records with judge outputs into one row per answer."""
    from rageval.eval.judge_runner import merge
    from rageval.io import read_jsonl

    recs = list(read_jsonl(records))
    n = read_json(nli)["scores"] if nli else None
    j = read_json(llm)["judgments"] if llm else None
    rows = merge(recs, n, j)
    write_jsonl(out, rows)
    _echo({"out": str(out), "rows": len(rows)})


@gen_app.command("choose")
def gen_choose(
    dev_report: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    tolerance: float = 0.02,
) -> None:
    """Fix the test-suite settings from the dev report (pipelines named
    <prompt>_k<K>[_<policy>_t<tau*100>]) with explicit rules:
      context_k   : the zero-shot K with the highest correctness; within `tolerance`, the higher
                    citation precision
      best_prompt : at that K, the strategy with the lowest unsupported-claim rate whose
                    correctness is no more than `tolerance` below zero-shot's
      verifier    : at that K, the verified variant with the highest citation precision whose
                    correctness is within `tolerance` of the same prompt unverified
    """
    import re

    S = read_json(dev_report)["summary"]
    pat = re.compile(r"^(zero|few|cot)_k(\d+)(?:_(repair|prune|revise)_t(\d+))?$")
    strat = {"zero": "zero_shot", "few": "few_shot", "cot": "cot"}

    def m(name: str, key: str) -> float:
        v = S[name].get(key)
        return float(v["mean"]) if isinstance(v, dict) else float("nan")

    parsed = {n: pat.match(n) for n in S}
    ks = sorted(
        {int(x.group(2)) for x in parsed.values() if x and x.group(1) == "zero" and not x.group(3)}
    )
    best_c = max(m(f"zero_k{k}", "correct") for k in ks)
    k = max(
        (k for k in ks if m(f"zero_k{k}", "correct") >= best_c - tolerance),
        key=lambda k: m(f"zero_k{k}", "citation_precision"),
    )
    base_c = m(f"zero_k{k}", "correct")
    prompt = "zero"
    for p in ("few", "cot"):
        name = f"{p}_k{k}"
        if (
            name in S
            and m(name, "unsupported_rate") < m(f"{prompt}_k{k}", "unsupported_rate")
            and m(name, "correct") >= base_c - tolerance
        ):
            prompt = p
    best_v: tuple[float, str] | None = None
    for n, x in parsed.items():
        if not x or not x.group(3) or int(x.group(2)) != k:
            continue
        ref = f"{x.group(1)}_k{k}"
        if ref in S and m(n, "correct") < m(ref, "correct") - tolerance:
            continue
        if best_v is None or m(n, "citation_precision") > best_v[0]:
            best_v = (m(n, "citation_precision"), n)
    policy, tau, vprompt = "none", 0.5, strat[prompt]
    if best_v is not None:
        x = pat.match(best_v[1])
        assert x is not None
        policy, tau, vprompt = x.group(3), int(x.group(4)) / 100, strat[x.group(1)]
    choice = {
        "context_k": k,
        "best_prompt": strat[prompt],
        "verify_policy": policy,
        "verify_tau": tau,
        "verify_prompt": vprompt,
        "verify_pipeline": best_v[1] if best_v else None,
        "tolerance": tolerance,
        "dev_evidence": {
            n: {
                key: m(n, key)
                for key in (
                    "correct",
                    "citation_precision",
                    "unsupported_rate",
                    "abstain_unanswerable",
                )
            }
            for n in S
        },
    }
    write_json(out, choice)
    _echo({k2: v for k2, v in choice.items() if k2 != "dev_evidence"})


@gen_app.command("report")
def gen_report(
    judged: Annotated[list[Path], typer.Option(help="judged JSONL file(s)")],
    out: Annotated[Path, typer.Option()],
    pairs: Annotated[
        list[str] | None, typer.Option(help="a:b pipeline pairs for paired tests")
    ] = None,
    metrics: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    from rageval.eval.report import by_pipeline, paired, summarize
    from rageval.io import read_jsonl

    rows = [r for p in judged for r in read_jsonl(p)]
    groups = by_pipeline(rows)
    summary = {name: summarize(rs) for name, rs in groups.items()}
    comps = []
    if pairs:
        pp = [tuple(p.split(":", 1)) for p in pairs]
        ms = metrics or [
            "unsupported_count",
            "has_unsupported",
            "unsupported_rate",
            "citation_precision",
            "correct",
            "abstain_unanswerable",
        ]
        comps = paired(groups, pp, ms)  # type: ignore[arg-type]
    write_json(
        out,
        {"meta": run_meta(judged=[str(p) for p in judged]), "summary": summary, "paired": comps},
    )
    brief = {
        n: {
            k: (round(v["mean"], 4) if isinstance(v, dict) and "mean" in v else v)
            for k, v in s.items()
            if k
            in (
                "n",
                "answered",
                "correct",
                "abstain_unanswerable",
                "citation_precision",
                "citation_recall",
                "unsupported_claims_total",
                "claims_total",
                "unsupported_rate",
                "context_recall",
            )
        }
        for n, s in summary.items()
    }
    _echo(brief)


bench_app = typer.Typer(no_args_is_help=True, help="Micro-benchmarks")
app.add_typer(bench_app, name="bench")


@bench_app.command("encoders")
def bench_encoders(
    out: Annotated[Path, typer.Option()],
    candidates: Annotated[
        Path, typer.Option(help="cached candidates for realistic query/doc pairs")
    ],
    suite: Annotated[Path, typer.Option()] = Path("data/hotpot100k/dev.jsonl"),
    corpus: Annotated[Path, typer.Option()] = Path("data/hotpot100k/corpus.jsonl"),
    reranker: Annotated[list[str] | None, typer.Option()] = None,
    embedder: Annotated[list[str] | None, typer.Option()] = None,
    verifier: Annotated[list[str] | None, typer.Option()] = None,
    pairs: int = 6000,
    batch: Annotated[list[int] | None, typer.Option()] = None,
) -> None:
    """GPU throughput of the serving encoders on realistic inputs (pairs/s, queries/s)."""
    import time

    import torch

    from rageval.corpus import Corpus, load_suite
    from rageval.judges.nli import ClassNLI
    from rageval.models import HFCrossEncoder, HFEmbedder
    from rageval.retrieval.dense import format_query
    from rageval.retrieval.experiments import load_candidates

    c = Corpus.load(corpus)
    qs = {q.qid: q for q in load_suite(suite)}
    cands = load_candidates(candidates)
    pp: list[tuple[str, str]] = []
    for qid, cc in cands.items():
        for h in cc["hnsw"][:50]:
            pp.append((qs[qid].question, c.index_text(h.doc_id)))
    pp = pp[:pairs]
    res: dict[str, Any] = {
        "pairs": len(pp),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    def timeit(fn: Any, items: list[Any]) -> float:
        fn(items[:256])  # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(items)
        torch.cuda.synchronize()
        return len(items) / (time.perf_counter() - t0)

    for m in reranker or []:
        for b in batch or [64, 128, 256]:
            ce = HFCrossEncoder(m, batch_size=b)
            res[f"rerank/{m}/b{b}"] = round(timeit(ce.score, pp), 1)
            del ce
            torch.cuda.empty_cache()
    for m in embedder or []:
        emb = HFEmbedder(m, batch_size=64)
        qtexts = [format_query(m, q.question) for q in qs.values()] * 4
        res[f"embed_queries/{m}"] = round(timeit(emb.embed, qtexts), 1)
        del emb
    for m in verifier or []:
        nli = ClassNLI(m, batch_size=128)

        def nli_score(x: list[tuple[str, str]], model: ClassNLI = nli) -> Any:
            return model.prob([a for a, _ in x], [b for _, b in x])

        res[f"nli/{m}"] = round(timeit(nli_score, [(d, q) for q, d in pp]), 1)
        del nli
    write_json(out, {"meta": run_meta(), **res})
    _echo(res)


@bench_app.command("parity")
def bench_parity(
    out: Annotated[Path, typer.Option()],
    embed_url: Annotated[str, typer.Option()],
    embedder: Annotated[str, typer.Option()],
    rerank_url: Annotated[str, typer.Option()],
    reranker: Annotated[str, typer.Option()],
    candidates: Annotated[Path, typer.Option()],
    suite: Annotated[Path, typer.Option()] = Path("data/hotpot100k/dev.jsonl"),
    corpus: Annotated[Path, typer.Option()] = Path("data/hotpot100k/corpus.jsonl"),
    n: int = 100,
) -> None:
    """Served (vLLM) embeddings and rerank scores vs the in-process HF models the indexes and
    offline results were built with: cosine for embeddings, rank correlation for reranking."""
    import asyncio

    import numpy as np
    from scipy.stats import spearmanr

    from rageval.corpus import Corpus, load_suite
    from rageval.models import HFCrossEncoder, HFEmbedder
    from rageval.retrieval.dense import format_query
    from rageval.retrieval.experiments import load_candidates
    from rageval.serve.encoders import RemoteEmbedClient, RemoteRerankClient

    c = Corpus.load(corpus)
    qs = load_suite(suite)[:n]
    cands = load_candidates(candidates)
    qtexts = [format_query(embedder, q.question) for q in qs]
    dtexts = [c.index_text(cands[q.qid]["hnsw"][0].doc_id) for q in qs]
    hf_e = HFEmbedder(embedder, batch_size=64)
    ref = np.concatenate([hf_e.embed(qtexts), hf_e.embed(dtexts)])
    del hf_e

    async def served() -> tuple[np.ndarray, list[np.ndarray]]:
        ec = RemoteEmbedClient(embed_url, model=embedder)
        rc = RemoteRerankClient(rerank_url, model=reranker)
        e = np.concatenate([await ec.submit(qtexts), await ec.submit(dtexts)])
        rs = []
        for q in qs[:30]:
            docs = [c.index_text(h.doc_id) for h in cands[q.qid]["hnsw"][:50]]
            rs.append(await rc.submit([(q.question, d) for d in docs]))
        return e, rs

    e, rs = asyncio.run(served())
    cos = (ref * e).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(e, axis=1))
    ce = HFCrossEncoder(reranker, batch_size=64)
    rhos, top10 = [], []
    for q, r in zip(qs[:30], rs, strict=True):
        docs = [c.index_text(h.doc_id) for h in cands[q.qid]["hnsw"][:50]]
        h = ce.score([(q.question, d) for d in docs])
        rhos.append(float(spearmanr(h, r).statistic))
        top10.append(len(set(np.argsort(-h)[:10]) & set(np.argsort(-r)[:10])) / 10)
    res = {
        "embed_cosine_min": float(cos.min()),
        "embed_cosine_mean": float(cos.mean()),
        "rerank_spearman_mean": float(np.mean(rhos)),
        "rerank_spearman_min": float(np.min(rhos)),
        "rerank_top10_overlap_mean": float(np.mean(top10)),
    }
    write_json(out, {"meta": run_meta(embedder=embedder, reranker=reranker), **res})
    _echo(res)


# ---------------------------------------------------------------- serving / load


@app.command("serve")
def serve(
    config: Annotated[Path, typer.Option()],
    host: str = "127.0.0.1",
    port: int = 8080,
    set_: Annotated[
        list[str] | None, typer.Option("--set", help="dotted overrides, e.g. router=static")
    ] = None,
    workers: int = 1,
) -> None:
    """Run the LangGraph agent behind FastAPI. With --workers > 1 every worker process runs its own
    agent; the encoders should then be remote (models.*_url) so the GPU models load once."""
    import uvicorn

    from rageval.serve.app import create_app, load_config

    if workers > 1:
        # each worker process builds its own agent (indexes, router state, telemetry files)
        import os

        os.environ["RAG_SERVE_CONFIG"] = str(config)
        os.environ["RAG_SERVE_SET"] = json.dumps(set_ or [])
        os.environ["RAG_SERVE_WORKERS"] = str(workers)
        load_config(config, set_)  # validate before forking
        uvicorn.run(
            "rageval.serve.app:app_from_env",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            log_level="warning",
            access_log=False,
            timeout_keep_alive=75,
        )
        return
    cfg = load_config(config, set_)
    uvicorn.run(
        create_app(cfg),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        timeout_keep_alive=75,
    )


@app.command("serve-materialize")
def serve_materialize(
    template: Annotated[Path, typer.Option()],
    out: Annotated[Path, typer.Option()],
    embedder: Annotated[str, typer.Option()],
    reranker: Annotated[str, typer.Option()],
    agent: Annotated[str, typer.Option(help="AgentConfig as JSON")],
) -> None:
    """Fill a serving config template; the filled file is kept next to the load-test results."""
    from rageval.agent.graph import AgentConfig

    AgentConfig.model_validate(json.loads(agent))  # fail early on a bad config
    txt = template.read_text().replace("__EMBEDDER__", embedder).replace("__RERANKER__", reranker)
    txt = txt.replace("__AGENT__", json.dumps(json.loads(agent), sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    _echo({"out": str(out)})


enc_app = typer.Typer(no_args_is_help=True, help="Stand-alone encoder servers")
app.add_typer(enc_app, name="encoders")


@enc_app.command("serve")
def encoders_serve(
    kind: Annotated[str, typer.Option(help="embed | rerank | nli")],
    model: Annotated[str, typer.Option()],
    port: Annotated[int, typer.Option()],
    host: str = "127.0.0.1",
    device: str | None = None,
    max_items: int = 512,
    max_wait_ms: float = 4.0,
    batch_size: int = 128,
) -> None:
    """One encoder model per process, micro-batched across concurrent requests."""
    import uvicorn

    from rageval.serve.encoders import create_encoder_app

    app_ = create_encoder_app(kind, model, device, max_items, max_wait_ms, batch_size)
    uvicorn.run(
        app_, host=host, port=port, log_level="warning", access_log=False, timeout_keep_alive=75
    )


lt_app = typer.Typer(no_args_is_help=True, help="Open-loop load tests")
app.add_typer(lt_app, name="loadtest")


@lt_app.command("run")
def loadtest_run(
    url: Annotated[str, typer.Option()],
    out: Annotated[Path, typer.Option()],
    suites: Annotated[list[Path], typer.Option(help="question pools")],
    qps: float = 50.0,
    duration: float = 300.0,
    warmup: float = 30.0,
    seed: int = 0,
    budget_ms: float | None = None,
    label: str = "",
) -> None:
    import asyncio

    from rageval.io import read_jsonl
    from rageval.loadtest.loadgen import run_load

    pool = [(str(r["qid"]), str(r["question"])) for p in suites for r in read_jsonl(p)]
    rows, summary = asyncio.run(run_load(url, pool, qps, duration, warmup, seed, budget_ms))
    write_jsonl(out, rows)
    write_json(
        out.with_suffix(".summary.json"),
        {"meta": run_meta(label=label, url=url, budget_ms=budget_ms, seed=seed), **summary},
    )
    _echo(
        {
            k: summary[k]
            for k in (
                "offered_qps",
                "completed_ok",
                "errors",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "route_share",
                "degraded_share",
            )
        }
    )


@lt_app.command("to-records")
def loadtest_to_records(
    rows: Annotated[Path, typer.Option(help="loadgen JSONL")],
    suite: Annotated[Path, typer.Option(help="labelled suite; other questions are skipped")],
    out: Annotated[Path, typer.Option()],
    tag: str = "load",
    warmup: float = 30.0,
) -> None:
    """Turn answers served under load into generation records so the same judges can score
    quality per route (primary vs fallback)."""
    from rageval.corpus import load_suite
    from rageval.io import read_jsonl

    qs = {q.qid: q for q in load_suite(suite)}
    recs = []
    for r in read_jsonl(rows):
        q = qs.get(r["qid"])
        if q is None or r.get("status") != 200 or r["at"] < warmup:
            continue
        recs.append(
            {
                "pipeline": f"{tag}/{r.get('route')}",
                "qid": f"{q.qid}#{r['i']}",
                "qtype": q.qtype,
                "answerable": q.answerable,
                "question": q.question,
                "gold_answer": q.answer,
                "gold_doc_ids": q.gold_doc_ids,
                "context": r.get("context") or [],
                "context_recall": None,
                "latency_ms": r.get("latency_ms"),
                "final": {
                    "answer": r.get("answer") or "",
                    "abstained": bool(r.get("abstained")),
                    "reasoning": "",
                    "statements": [
                        {"text": s["text"], "citations": s["citations"], "abstain": False}
                        for s in (r.get("statements") or [])
                    ],
                },
            }
        )
    write_jsonl(out, recs)
    _echo({"out": str(out), "records": len(recs)})


@lt_app.command("compare")
def loadtest_compare(
    base: Annotated[list[Path], typer.Option(help="baseline run dirs (one per seed)")],
    treat: Annotated[list[Path], typer.Option(help="treatment run dirs (one per seed)")],
    out: Annotated[Path, typer.Option()],
    sweep: Annotated[list[Path] | None, typer.Option(help="capacity sweep run dirs")] = None,
    block_s: float = 10.0,
) -> None:
    """Pooled latency percentiles per mode with block-bootstrap CIs for the reduction."""
    from rageval.loadtest.analyze import compare_modes, load_run, node_breakdown

    b = [load_run(p) for p in base]
    t = [load_run(p) for p in treat]
    res = compare_modes(b, t, block_s)
    res["node_p95_ms"] = {"base": node_breakdown(b), "treat": node_breakdown(t)}
    res["node_p50_ms"] = {"base": node_breakdown(b, 0.5), "treat": node_breakdown(t, 0.5)}
    if sweep:
        res["sweep"] = [
            {
                k: read_json(p / "load.summary.json")[k]
                for k in (
                    "target_qps",
                    "offered_qps",
                    "p50_ms",
                    "p95_ms",
                    "p99_ms",
                    "error_rate",
                    "completed_ok",
                )
            }
            for p in sweep
        ]
    write_json(
        out, {"meta": run_meta(base=[str(p) for p in base], treat=[str(p) for p in treat]), **res}
    )
    _echo(
        {
            q: {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in res[q].items()
                if not k.endswith("per_seed_ms")
            }
            for q in ("p50", "p95", "p99")
        }
    )


if __name__ == "__main__":
    app()
