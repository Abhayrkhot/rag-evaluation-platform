"""Split quality-under-load rows by serving path (full pipeline vs degraded stages).

Judged rows carry qid "<question>#<request index>" and pipeline "<mode>/<route>"; the per-seed
record files (quality/q50_<mode>_s<seed>.jsonl) say which seed an answer came from, and that
seed's load log gives the degraded stages of that request.

    python scripts/quality_by_path.py results/explorer/m4/main_fp8   ->   <run>/quality/by_path.json
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict]:
    if not path.exists() and Path(str(path) + ".gz").exists():
        path = Path(str(path) + ".gz")
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return [json.loads(x) for x in f if x.strip()]


def main(run: Path) -> None:
    q = run / "quality"
    path_of: dict[tuple[str, str], str] = {}  # (mode, qid#i) -> degraded path
    for d in sorted(run.glob("q50_*_s*")):
        mode = d.name.rsplit("_s", 1)[0]
        by_i = {
            str(r["i"]): "+".join(sorted(r.get("degraded") or [])) or "full"
            for r in read(d / "load.jsonl")
            if r.get("status") == 200
        }
        for rec in read(q / f"{d.name}.jsonl"):
            path_of[(mode, rec["qid"])] = by_i.get(rec["qid"].rsplit("#", 1)[1], "?")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in read(q / "judged.jsonl"):
        mode = r["pipeline"].split("/")[0]
        groups[f"{mode}/{path_of.get((mode, r['qid']), '?')}"].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        answered = [r for r in rs if not r["abstained"]]
        ans = [r for r in rs if r["answerable"]]
        judged = [r for r in answered if "n_unsupported" in r]
        claims = sum(r["n_claims"] for r in judged)
        cps = [r["cit_precision"] for r in answered if "cit_precision" in r]
        out[g] = {
            "n": len(rs),
            "citation_precision": float(np.mean(cps)) if cps else None,
            "correct": float(np.mean([bool(r.get("judge_correct")) for r in ans])) if ans else None,
            "unsupported_rate": sum(r["n_unsupported"] for r in judged) / claims
            if claims
            else None,
        }
    (q / "by_path.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
