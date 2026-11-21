"""String-level answer checks (SQuAD normalisation). The LLM judge is the primary correctness
metric; these are cheap, deterministic cross-checks."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def answer_match(response: str, gold: str) -> bool:
    r, g = normalize(response), normalize(gold)
    if not g:
        return False
    if g in ("yes", "no"):
        m = re.search(r"\b(yes|no)\b", r)
        return m is not None and m.group(1) == g
    return re.search(rf"\b{re.escape(g)}\b", r) is not None


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)
