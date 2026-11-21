"""Deadline-aware LLM routing between a primary and a fallback model.

Policies (each can be switched off for the static baseline):
  overflow : when the primary already has `max_inflight` requests, send new ones to the fallback
             (load shedding to a cheaper model instead of queueing behind a saturated engine)
  deadline : when the primary's predicted latency (EWMA scaled by its current load) exceeds the
             request's remaining budget, go straight to the fallback
  hedge    : if the primary has not answered after `hedge_ms`, also send the request to the
             fallback and return whichever finishes first (Dean & Barroso, "The Tail at Scale")
  breaker  : after `breaker_failures` consecutive primary errors/timeouts, route everything to the
             fallback for `breaker_cooldown_s`, then let one probe request through (half-open)
  retry    : a primary error or timeout is retried once on the fallback while budget remains
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rageval.llm.client import ChatClient, ChatResult, LLMError


@dataclass
class RouterPolicy:
    overflow: bool = True
    max_inflight: int = 64
    deadline: bool = True
    deadline_safety: float = 1.0
    hedge: bool = True
    hedge_ms: float = 2500.0
    breaker: bool = True
    breaker_failures: int = 5
    breaker_cooldown_s: float = 10.0
    retry_on_fallback: bool = True
    primary_timeout_s: float = 60.0
    fallback_timeout_s: float = 60.0

    @classmethod
    def static(cls) -> RouterPolicy:
        """Baseline: always the primary, no shedding, hedging, breaker or retries."""
        return cls(
            overflow=False, deadline=False, hedge=False, breaker=False, retry_on_fallback=False
        )


class CircuitBreaker:
    def __init__(self, failures: int, cooldown_s: float) -> None:
        self.threshold = failures
        self.cooldown = cooldown_s
        self.consecutive = 0
        self.opened_at: float | None = None
        self.half_open_probe = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if time.monotonic() - self.opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        st = self.state
        if st == "closed":
            return True
        if st == "half_open" and not self.half_open_probe:
            self.half_open_probe = True
            return True
        return False

    def success(self) -> None:
        self.consecutive = 0
        self.opened_at = None
        self.half_open_probe = False

    def failure(self) -> None:
        self.consecutive += 1
        self.half_open_probe = False
        if self.consecutive >= self.threshold or self.opened_at is not None:
            self.opened_at = time.monotonic()


@dataclass
class EndpointStats:
    inflight: int = 0
    ewma_ms: float = 0.0
    n: int = 0
    alpha: float = 0.05

    def observe(self, ms: float) -> None:
        self.n += 1
        self.ewma_ms = ms if self.n == 1 else (1 - self.alpha) * self.ewma_ms + self.alpha * ms


@dataclass
class RouteResult:
    result: ChatResult
    route: str  # primary | fallback
    reason: str  # why this route answered: ok | overflow | deadline | hedge_won | breaker_open | primary_error | primary_timeout
    attempts: list[dict[str, Any]] = field(default_factory=list)


class FallbackRouter:
    def __init__(
        self, primary: ChatClient, fallback: ChatClient | None, policy: RouterPolicy | None = None
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.policy = policy or RouterPolicy()
        self.stats = {"primary": EndpointStats(), "fallback": EndpointStats()}
        self.breaker = CircuitBreaker(self.policy.breaker_failures, self.policy.breaker_cooldown_s)
        self.counters: dict[str, int] = {}

    def _count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    def predicted_primary_ms(self) -> float:
        s = self.stats["primary"]
        if s.n < 20:
            return 0.0
        load = s.inflight / max(self.policy.max_inflight, 1)
        return s.ewma_ms * (1.0 + load)

    def decide(self, remaining_ms: float | None) -> tuple[str, str]:
        p = self.policy
        if self.fallback is None:
            return "primary", "ok"
        if p.breaker and not self.breaker.allow():
            return "fallback", "breaker_open"
        if p.overflow and self.stats["primary"].inflight >= p.max_inflight:
            return "fallback", "overflow"
        if (
            p.deadline
            and remaining_ms is not None
            and self.predicted_primary_ms() * p.deadline_safety > remaining_ms
        ):
            return "fallback", "deadline"
        return "primary", "ok"

    def _start(
        self, which: str, messages: Sequence[dict[str, str]], timeout_s: float, **kw: Any
    ) -> asyncio.Future[ChatResult]:
        """Reserve an in-flight slot synchronously (so concurrent routing decisions see it), then
        schedule the call."""
        client = self.primary if which == "primary" else self.fallback
        assert client is not None
        st = self.stats[which]
        st.inflight += 1

        async def run() -> ChatResult:
            t0 = time.perf_counter()
            try:
                res = await client.chat(messages, timeout=timeout_s, **kw)
                st.observe((time.perf_counter() - t0) * 1e3)
                return res
            except asyncio.CancelledError:
                # a hedged-away request still tells us the primary was at least this slow
                st.observe((time.perf_counter() - t0) * 1e3)
                raise
            finally:
                st.inflight -= 1

        return asyncio.ensure_future(run())

    async def chat(
        self, messages: Sequence[dict[str, str]], remaining_ms: float | None = None, **kw: Any
    ) -> RouteResult:
        p = self.policy
        route, reason = self.decide(remaining_ms)
        self._count(f"route:{route}:{reason}")
        if route == "fallback":
            res = await self._start("fallback", messages, p.fallback_timeout_s, **kw)
            return RouteResult(res, "fallback", reason)

        budget_s = p.primary_timeout_s
        if remaining_ms is not None and p.deadline:
            budget_s = max(0.5, min(budget_s, remaining_ms / 1e3))
        primary = self._start("primary", messages, budget_s, **kw)
        attempts: list[dict[str, Any]] = []
        try:
            if p.hedge and self.fallback is not None:
                done, _ = await asyncio.wait({primary}, timeout=p.hedge_ms / 1e3)
                if not done:
                    self._count("hedge_fired")
                    hedge = self._start("fallback", messages, p.fallback_timeout_s, **kw)
                    pending: set[asyncio.Future[ChatResult]] = {primary, hedge}
                    while pending:
                        finished, pending = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )
                        for t in finished:
                            if t.exception() is None:
                                for o in pending:
                                    o.cancel()
                                if t is primary:
                                    self.breaker.success()
                                    return RouteResult(t.result(), "primary", "ok", attempts)
                                attempts.append({"route": "primary", "outcome": "hedged"})
                                return RouteResult(t.result(), "fallback", "hedge_won", attempts)
                    raise primary.exception() or LLMError("hedge: both routes failed")
            res = await primary
            self.breaker.success()
            return RouteResult(res, "primary", "ok", attempts)
        except (TimeoutError, LLMError) as e:
            if not primary.done():
                primary.cancel()
            self.breaker.failure()
            why = "primary_timeout" if "timeout" in str(e).lower() else "primary_error"
            attempts.append({"route": "primary", "outcome": why, "error": str(e)[:200]})
            self._count(why)
            if p.retry_on_fallback and self.fallback is not None:
                res = await self._start("fallback", messages, p.fallback_timeout_s, **kw)
                return RouteResult(res, "fallback", why, attempts)
            raise
