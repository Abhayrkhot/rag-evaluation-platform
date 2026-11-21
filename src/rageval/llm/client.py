"""Async client for OpenAI-compatible chat completion servers (vLLM)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    reasoning: str = ""
    endpoint: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class ChatClient:
    """One model behind one OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        name: str | None = None,
        timeout: float = 120.0,
        max_connections: int = 512,
        default_params: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name or model
        self.timeout = timeout
        self.default_params = dict(default_params or {})
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
        )

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float | None = None,
        **params: Any,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.default_params,
            **params,
        }
        t0 = time.perf_counter()
        try:
            r = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                timeout=timeout if timeout is not None else self.timeout,
            )
        except httpx.TimeoutException as e:
            raise LLMError(f"{self.name}: timeout") from e
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name}: {type(e).__name__}: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            text=(msg.get("content") or "").strip(),
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1e3,
            finish_reason=choice.get("finish_reason") or "",
            reasoning=(msg.get("reasoning_content") or msg.get("reasoning") or ""),
            endpoint=self.name,
            raw={},
        )

    async def health(self) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


async def gather_limited(coros: Sequence[Any], limit: int) -> list[Any]:
    """Run coroutines with bounded concurrency, preserving order. Exceptions are returned."""
    sem = asyncio.Semaphore(limit)

    async def run(c: Any) -> Any:
        async with sem:
            try:
                return await c
            except Exception as e:
                return e

    return await asyncio.gather(*(run(c) for c in coros))


QWEN3_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
