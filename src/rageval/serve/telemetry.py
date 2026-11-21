"""Tracing, trace logs and metrics for the serving path.

- OpenTelemetry spans (one root span per request, one child per graph node and model call),
  exported as JSONL span logs and, when RAG_OTLP_ENDPOINT is set, over OTLP/HTTP (Jaeger, Tempo).
- A per-request trace log line (JSONL) with node timings, route, fallback reason and token counts,
  written by a background thread so request handling never blocks on disk.
- Prometheus histograms and counters exposed at /metrics.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Any

import orjson
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from prometheus_client import Counter, Gauge, Histogram

LAT_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60)
NODE_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16)

REQUEST_SECONDS = Histogram(
    "rag_request_seconds",
    "End-to-end /v1/answer latency",
    ["route", "degraded"],
    buckets=LAT_BUCKETS,
)
NODE_SECONDS = Histogram(
    "rag_node_seconds", "Latency of each agent graph node", ["node"], buckets=NODE_BUCKETS
)
ROUTE_TOTAL = Counter("rag_route_total", "Generation route decisions", ["route", "reason"])
DEGRADED_TOTAL = Counter("rag_degraded_total", "Stages skipped to meet the deadline", ["stage"])
ERRORS_TOTAL = Counter("rag_errors_total", "Request errors", ["stage"])
LLM_INFLIGHT = Gauge("rag_llm_inflight", "In-flight generation requests", ["endpoint"])
BREAKER_OPEN = Gauge("rag_breaker_open", "1 when the primary circuit breaker is open")
TOKENS_TOTAL = Counter("rag_tokens_total", "Tokens by endpoint and kind", ["endpoint", "kind"])
BATCH_ITEMS = Histogram(
    "rag_batch_items",
    "Items per micro-batch",
    ["batcher"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
)


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "ab")  # noqa: SIM115 - closed in shutdown()
        self._lock = threading.Lock()

    def export(self, spans: Any) -> SpanExportResult:
        lines = []
        for s in spans:
            assert isinstance(s, ReadableSpan)
            ctx = s.get_span_context()
            lines.append(
                orjson.dumps(
                    {
                        "trace_id": f"{ctx.trace_id:032x}" if ctx else None,
                        "span_id": f"{ctx.span_id:016x}" if ctx else None,
                        "parent_id": f"{s.parent.span_id:016x}" if s.parent else None,
                        "name": s.name,
                        "start_ns": s.start_time,
                        "duration_ms": ((s.end_time or 0) - (s.start_time or 0)) / 1e6,
                        "status": s.status.status_code.name,
                        "attributes": dict(s.attributes or {}),
                    }
                )
            )
        with self._lock:
            self._f.write(b"\n".join(lines) + b"\n")
            self._f.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        with self._lock:
            self._f.close()


def setup_tracing(
    service: str, span_log: str | None, otlp_endpoint: str | None = None
) -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    if span_log:
        provider.add_span_processor(
            BatchSpanProcessor(JsonlSpanExporter(span_log), max_queue_size=65536)
        )
    endpoint = otlp_endpoint or os.environ.get("RAG_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("rageval")


class TraceLog:
    """Asynchronous JSONL writer for per-request trace records."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=100_000)
        self.dropped = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._t = threading.Thread(target=self._run, name="tracelog", daemon=True)
            self._t.start()

    def write(self, rec: dict[str, Any]) -> None:
        if not self.path:
            return
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        assert self.path is not None
        with open(self.path, "ab") as f:
            while True:
                rec = self._q.get()
                if rec is None:
                    return
                f.write(orjson.dumps(rec, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n")
                if self._q.empty():
                    f.flush()

    def close(self) -> None:
        if self.path:
            self._q.put(None)
            self._t.join(timeout=5)
