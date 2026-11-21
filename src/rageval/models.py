"""In-process transformer encoders: bi-encoder embedders and cross-encoder rerankers.

Implemented directly on `transformers` (no sentence-transformers) so the exact pooling, padding
side, dtype and batching are explicit and match what the model cards specify.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

# pooling per embedding model family (from the model cards)
POOLING = {
    "qwen3-embedding": "last",  # last token, left padding
    "bge-": "cls",
    "e5-": "mean",
    "snowflake-arctic-embed": "cls",
}


def pooling_for(model: str) -> str:
    m = model.lower()
    for key, pool in POOLING.items():
        if key in m:
            return pool
    return "mean"


def encode_sorted_batches(
    tok: Any,
    first: Sequence[str],
    second: Sequence[str] | None,
    max_length: int,
    batch_size: int,
    truncation: str | bool = True,
) -> Iterator[tuple[list[int], Any]]:
    """Tokenize every input in one (Rust-parallel) call without padding, then yield
    length-sorted batches padded only to their own longest sequence."""
    enc = tok(
        list(first),
        list(second) if second is not None else None,
        truncation=truncation,
        max_length=max_length,
        padding=False,
    )
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    order = sorted(range(len(first)), key=lambda i: len(enc["input_ids"][i]))
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        batch = tok.pad(
            {k: [enc[k][i] for i in idx] for k in keys}, padding=True, return_tensors="pt"
        )
        yield idx, batch


def _device(device: str | None) -> str:
    import torch

    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _dtype(device: str, fp16: bool | None) -> Any:
    import torch

    use = fp16 if fp16 is not None else device.startswith("cuda")
    if not use:
        return torch.float32
    return (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )


class HFEmbedder:
    def __init__(
        self,
        model: str,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
        fp16: bool | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model = model
        self.device = _device(device)
        self.pool = pooling_for(model)
        self.tok = AutoTokenizer.from_pretrained(
            model, padding_side="left" if self.pool == "last" else "right"
        )
        self.net = (
            AutoModel.from_pretrained(model, dtype=_dtype(self.device, fp16)).to(self.device).eval()
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        out: list[np.ndarray | None] = [None] * len(texts)
        with self._lock:
            for idx, enc in encode_sorted_batches(
                self.tok, texts, None, self.max_length, self.batch_size
            ):
                enc = enc.to(self.device)
                with torch.inference_mode():
                    h = self.net(**enc).last_hidden_state
                    mask = enc["attention_mask"]
                    if self.pool == "cls":
                        v = h[:, 0]
                    elif self.pool == "last":
                        v = h[:, -1]  # left padded: last position is the final token
                    else:
                        m = mask.unsqueeze(-1).to(h.dtype)
                        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                    v = torch.nn.functional.normalize(v.float(), dim=-1)
                arr = v.cpu().numpy()
                for j, i in enumerate(idx):
                    out[i] = arr[j]
        return np.stack(out).astype(np.float32)  # type: ignore[arg-type]


class HFCrossEncoder:
    def __init__(
        self,
        model: str,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
        fp16: bool | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model = model
        self.device = _device(device)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.net = (
            AutoModelForSequenceClassification.from_pretrained(
                model, dtype=_dtype(self.device, fp16)
            )
            .to(self.device)
            .eval()
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.Lock()

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        torch = self.torch
        out = np.zeros(len(pairs), dtype=np.float32)
        if not pairs:
            return out
        with self._lock:
            for idx, enc in encode_sorted_batches(
                self.tok,
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                self.max_length,
                self.batch_size,
                "only_second",
            ):
                enc = enc.to(self.device)
                with torch.inference_mode():
                    logits = self.net(**enc).logits.float()
                score = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1] - logits[:, 0]
                out[idx] = score.cpu().numpy()
        return out
