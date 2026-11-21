"""JSONL / JSON helpers and run metadata."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import orjson


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row, option=orjson.OPT_SERIALIZE_NUMPY))
            f.write(b"\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write(orjson.dumps(row, option=orjson.OPT_SERIALIZE_NUMPY))
        f.write(b"\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(
        orjson.dumps(
            obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY
        )
    )
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    return orjson.loads(Path(path).read_bytes())


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)).hexdigest()[:16]


def git_sha() -> str:
    env_sha = os.environ.get("RAGEVAL_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "configs"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        gf = Path("GIT_SHA")
        return gf.read_text().strip() if gf.exists() else "unknown"


def run_meta(**extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "git_sha": git_sha(),
        "host": os.environ.get("RAGEVAL_HOST_LABEL", platform.node()),
        "python": platform.python_version(),
        "started_unix": time.time(),
    }
    meta.update(extra)
    return meta
