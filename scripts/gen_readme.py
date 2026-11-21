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


# ---------------------------------------------------------------- render


BLOCKS = {
    "retrieval": retrieval_block,
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
