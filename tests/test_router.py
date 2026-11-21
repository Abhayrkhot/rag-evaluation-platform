import asyncio

import pytest

from rageval.llm.client import ChatResult, LLMError
from rageval.llm.router import CircuitBreaker, FallbackRouter, RouterPolicy


class FakeClient:
    def __init__(self, name, delay=0.0, fail=False):
        self.name = name
        self.delay = delay
        self.fail = fail
        self.calls = 0
        self.cancelled = 0

    async def chat(self, messages, timeout=None, **kw):
        self.calls += 1
        try:
            if timeout is not None and self.delay > timeout:
                await asyncio.sleep(timeout)
                raise LLMError(f"{self.name}: timeout")
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.fail:
            raise LLMError(f"{self.name}: HTTP 500")
        return ChatResult(text=f"from {self.name}", model=self.name, latency_ms=self.delay * 1e3)


MSG = [{"role": "user", "content": "q"}]


async def test_static_policy_always_primary():
    p, f = FakeClient("p"), FakeClient("f")
    r = FallbackRouter(p, f, RouterPolicy.static())
    res = await r.chat(MSG)
    assert res.route == "primary" and f.calls == 0


async def test_overflow_sheds_to_fallback():
    p, f = FakeClient("p", delay=0.2), FakeClient("f")
    r = FallbackRouter(p, f, RouterPolicy(max_inflight=2, hedge=False, deadline=False))
    results = await asyncio.gather(*(r.chat(MSG) for _ in range(5)))
    routes = [x.route for x in results]
    assert routes.count("primary") == 2 and routes.count("fallback") == 3
    assert {x.reason for x in results if x.route == "fallback"} == {"overflow"}


async def test_hedge_wins_and_cancels_primary():
    p, f = FakeClient("p", delay=0.5), FakeClient("f", delay=0.01)
    r = FallbackRouter(p, f, RouterPolicy(hedge_ms=50, deadline=False, overflow=False))
    res = await r.chat(MSG)
    assert res.route == "fallback" and res.reason == "hedge_won"
    await asyncio.sleep(0)
    assert p.cancelled == 1


async def test_primary_fast_no_hedge():
    p, f = FakeClient("p", delay=0.01), FakeClient("f")
    r = FallbackRouter(p, f, RouterPolicy(hedge_ms=200))
    res = await r.chat(MSG)
    assert res.route == "primary" and f.calls == 0


async def test_error_retries_on_fallback_and_opens_breaker():
    p, f = FakeClient("p", fail=True), FakeClient("f")
    pol = RouterPolicy(hedge=False, breaker_failures=2, breaker_cooldown_s=60)
    r = FallbackRouter(p, f, pol)
    a = await r.chat(MSG)
    b = await r.chat(MSG)
    c = await r.chat(MSG)
    assert (a.route, a.reason) == ("fallback", "primary_error")
    assert b.reason == "primary_error"
    assert (c.route, c.reason) == ("fallback", "breaker_open")
    assert p.calls == 2


async def test_static_policy_raises_on_error():
    r = FallbackRouter(FakeClient("p", fail=True), FakeClient("f"), RouterPolicy.static())
    with pytest.raises(LLMError):
        await r.chat(MSG)


async def test_deadline_routes_to_fallback_once_primary_is_known_slow():
    p, f = FakeClient("p", delay=0.05), FakeClient("f")
    r = FallbackRouter(p, f, RouterPolicy(hedge=False, overflow=False))
    for _ in range(25):
        await r.chat(MSG, remaining_ms=10_000)
    res = await r.chat(MSG, remaining_ms=10)
    assert (res.route, res.reason) == ("fallback", "deadline")


def test_breaker_half_open_allows_one_probe():
    b = CircuitBreaker(failures=1, cooldown_s=0.0)
    b.failure()
    assert b.state == "half_open"
    assert b.allow() and not b.allow()
    b.success()
    assert b.state == "closed" and b.allow()
