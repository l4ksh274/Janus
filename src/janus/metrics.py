"""Custom Prometheus metrics for the Janus semantic-caching proxy.

Metrics exposed (all on a dedicated registry at ``/metrics``):

  requests_total              Counter  {method, endpoint, cache_result}
  cache_hit_total             Counter  {level, scope}
  request_latency_seconds     Histogram {method, endpoint, cache_result}
  estimated_cost_saved_gbp    Counter  {model, scope}

Grafana queries derived from these four:
  • Total cache hits:       sum(cache_hit_total)
  • Cache hits by level:    sum by (level)(cache_hit_total)
  • Cache hit ratio:        sum(rate(requests_total{cache_result=~"HIT.*"}[5m]))
                            / sum(rate(requests_total[5m]))
  • Request latency:        histogram_quantile(0.95, sum(rate(
                              request_latency_seconds_bucket[5m])) by (le, cache_result))
  • Requests per second:    sum(rate(requests_total[5m]))
  • Estimated cost saved:   sum(estimated_cost_saved_gbp)
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from fastapi import Response as FastAPIResponse

# Dedicated registry so the /metrics endpoint only shows our custom metrics.
REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# 1. requests_total  – enables RPS and cache-hit-ratio in Grafana
# ---------------------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "requests_total",
    "Total requests handled by the proxy.",
    labelnames=["method", "endpoint", "cache_result"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# 2. cache_hit_total  – total cache hits, sliceable by level and scope
# ---------------------------------------------------------------------------
CACHE_HIT_TOTAL = Counter(
    "cache_hit_total",
    "Total number of cache hits served by the proxy.",
    labelnames=["level", "scope"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# 3. request_latency_seconds  – latency histogram
# ---------------------------------------------------------------------------
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "End-to-end latency of proxied requests.",
    labelnames=["method", "endpoint", "cache_result"],
    buckets=(
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf"),
    ),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# 4. estimated_cost_saved_gbp  – running cost savings counter
# ---------------------------------------------------------------------------
MODEL_COST_GBP: dict[str, float] = {
    "meta/llama-3.1-8b-instruct": 0.0001,
    "meta/llama-3.1-70b-instruct": 0.0008,
    "meta/llama-3.1-405b-instruct": 0.004,
    "nvidia/llama-3.1-nemotron-70b-instruct": 0.001,
    "mistralai/mistral-large-4-1t-a52b": 0.0012,
    "deepseek-ai/deepseek-r1": 0.002,
}
DEFAULT_COST_GBP = 0.0005

COST_SAVED_GBP = Counter(
    "estimated_cost_saved_gbp",
    "Estimated upstream API cost avoided by serving from cache (GBP).",
    labelnames=["model", "scope"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Helper functions called from the request handlers
# ---------------------------------------------------------------------------
def record_cache_hit(level: str, model: str, scope: str) -> None:
    """Record a cache hit and the associated cost saving."""
    CACHE_HIT_TOTAL.labels(level=level, scope=scope).inc()
    cost = MODEL_COST_GBP.get(model, DEFAULT_COST_GBP)
    COST_SAVED_GBP.labels(model=model, scope=scope).inc(cost)


def record_request_latency(
    method: str, endpoint: str, cache_result: str, elapsed: float,
) -> None:
    """Record one request in both the requests counter and latency histogram."""
    REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, cache_result=cache_result).inc()
    REQUEST_LATENCY.labels(
        method=method, endpoint=endpoint, cache_result=cache_result,
    ).observe(elapsed)


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------
def instrument_app(app) -> None:
    """Add ``GET /metrics`` to *app* for Prometheus scraping."""

    @app.get("/metrics")
    async def metrics_endpoint():
        body = generate_latest(REGISTRY)
        return FastAPIResponse(
            content=body,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

def reset_metrics() -> None:
    """Reset all Prometheus metrics to their initial state."""
    REQUESTS_TOTAL.clear()
    CACHE_HIT_TOTAL.clear()
    REQUEST_LATENCY.clear()
    COST_SAVED_GBP.clear()