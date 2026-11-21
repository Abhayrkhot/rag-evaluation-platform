"""NLI models used as judges (evaluation) and as the in-pipeline citation verifier.

TrueNLI   : google/t5_xxl_true_nli_mixture, the T5-11B model ALCE uses for its citation metrics.
            Only used for evaluation; never inside the pipeline being evaluated.
ClassNLI  : a sequence-classification NLI model (DeBERTa-v3-large by default) returning
            P(entailment). Used by the pipeline's citation verifier.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np


def _length_sorted_batches(texts: Sequence[Any], key: Any, batch: int) -> list[list[int]]:
    order = sorted(range(len(texts)), key=lambda i: key(texts[i]))
    return [order[i : i + batch] for i in range(0, len(order), batch)]


class TrueNLI:
    MODEL = "google/t5_xxl_true_nli_mixture"

    def __init__(
        self,
        model: str = MODEL,
        device: str = "cuda",
        batch_size: int = 32,
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.model_name = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(model, dtype=getattr(torch, dtype))
            .to(device)
            .eval()
        )
        self.device = device
        self.batch_size = batch_size
        ids = self.tok("1", add_special_tokens=False).input_ids
        self.yes_id = ids[-1]
        self.no_id = self.tok("0", add_special_tokens=False).input_ids[-1]
        self.calls = 0
        self.seconds = 0.0

    def entail(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[bool]:
        return [p > 0.5 for p in self.prob(premises, hypotheses)]

    def prob(self, premises: Sequence[str], hypotheses: Sequence[str]) -> list[float]:
        """P("1") renormalised over {"1", "0"}; > 0.5 matches ALCE's greedy decode == "1"
        whenever the argmax token is one of the two labels (it always is in practice)."""
        torch = self.torch
        texts = [f"premise: {p} hypothesis: {h}" for p, h in zip(premises, hypotheses, strict=True)]
        out = np.zeros(len(texts), dtype=np.float32)
        t0 = time.perf_counter()
        start = self.model.config.decoder_start_token_id
        for idx in _length_sorted_batches(texts, len, self.batch_size):
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(
                self.device
            )
            dec = torch.full((len(idx), 1), start, dtype=torch.long, device=self.device)
            with torch.inference_mode():
                logits = (
                    self.model(**enc, decoder_input_ids=dec, use_cache=False)
                    .logits[:, -1, :]
                    .float()
                )
            two = torch.stack([logits[:, self.yes_id], logits[:, self.no_id]], dim=-1).softmax(-1)[
                :, 0
            ]
            # ALCE treats anything but a decoded "1" as not entailed; mirror argmax over the vocab
            argmax_yes = logits.argmax(-1) == self.yes_id
            p = torch.where(argmax_yes, two.clamp(min=0.5 + 1e-6), two.clamp(max=0.5))
            out[idx] = p.cpu().numpy()
        self.calls += len(texts)
        self.seconds += time.perf_counter() - t0
        return out.tolist()


class ClassNLI:
    MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

    def __init__(
        self,
        model: str = MODEL,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 512,
        fp16: bool | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = fp16 if fp16 is not None else device.startswith("cuda")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model, dtype=torch.float16 if use_fp16 else torch.float32
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        # NLI models name the class "entailment"; binary grounding checkers (MiniCheck) put
        # "supported" last
        self.entail_idx = labels.get("entailment", labels.get("supported", len(labels) - 1))

    def prob(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        from rageval.models import encode_sorted_batches

        torch = self.torch
        out = np.zeros(len(premises), dtype=np.float32)
        if not premises:
            return out
        for idx, enc in encode_sorted_batches(
            self.tok, premises, hypotheses, self.max_length, self.batch_size, "only_first"
        ):
            enc = enc.to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits.float()
            out[idx] = logits.softmax(-1)[:, self.entail_idx].cpu().numpy()
        return out


class EnsembleNLI:
    """Several checkers that must agree: P(supported) = min over models. Used to cut the false
    positives a single NLI model makes when the repair search tries many candidate documents."""

    def __init__(self, models: Sequence[str], **kw: Any) -> None:
        self.members = [ClassNLI(m, **kw) for m in models]
        self.model_name = "+".join(models)

    def prob(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        return np.minimum.reduce([m.prob(premises, hypotheses) for m in self.members])


def load_verifier(spec: str, **kw: Any) -> Any:
    """'modelA' or 'modelA+modelB' (ensemble)."""
    parts = [x for x in spec.split("+") if x]
    return ClassNLI(parts[0], **kw) if len(parts) == 1 else EnsembleNLI(parts, **kw)
