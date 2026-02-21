"""
Cost tracking and performance metrics.

CostTracker logs every LLM call with model, tokens, and estimated cost.
PerformanceTimer provides a context manager for timing operations.
MetricsCollector aggregates in-memory stats for the /metrics endpoint.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

# Per-model pricing (USD per 1K tokens) — OpenRouter pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4": {"input": 0.003, "output": 0.015},
    "anthropic/claude-haiku-4": {"input": 0.0008, "output": 0.004},
    "deepseek/deepseek-chat-v3": {"input": 0.00014, "output": 0.00028},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD for a model call."""
    pricing = MODEL_PRICING.get(model, {"input": 0.001, "output": 0.002})
    return (input_tokens / 1000 * pricing["input"]) + (
        output_tokens / 1000 * pricing["output"]
    )


@dataclass
class MetricsCollector:
    """Aggregates in-memory metrics across the application lifetime."""

    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_retrieval_queries: int = 0
    total_uploads: int = 0
    total_errors: int = 0
    total_sessions: int = 0
    active_sessions: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)

    # Per-model tracking
    model_call_counts: dict[str, int] = field(default_factory=dict)
    model_token_counts: dict[str, int] = field(default_factory=dict)
    model_error_counts: dict[str, int] = field(default_factory=dict)

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        operation: str,
        session_id: str | None = None,
    ) -> float:
        """Record an LLM call and return estimated cost."""
        cost = estimate_cost(model, input_tokens, output_tokens)

        self.total_llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.latencies.append(latency_ms)

        # Per-model tracking
        self.model_call_counts[model] = self.model_call_counts.get(model, 0) + 1
        self.model_token_counts[model] = (
            self.model_token_counts.get(model, 0) + input_tokens + output_tokens
        )

        logger.info(
            "llm_call",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 1),
            operation=operation,
            session_id=session_id,
        )

        return cost

    def record_retrieval(self, query: str, result_count: int, latency_ms: float) -> None:
        """Record a RAG retrieval query."""
        self.total_retrieval_queries += 1
        logger.info(
            "retrieval_query",
            query_length=len(query),
            result_count=result_count,
            latency_ms=round(latency_ms, 1),
        )

    def record_error(self, error_type: str, detail: str = "", model: str = "") -> None:
        """Record an error by type, optionally per-model."""
        self.total_errors += 1
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
        if model:
            self.model_error_counts[model] = self.model_error_counts.get(model, 0) + 1
        logger.error("error_recorded", error_type=error_type, detail=detail, model=model or None)

    def record_session_start(self) -> None:
        """Record a new session starting."""
        self.total_sessions += 1
        self.active_sessions += 1

    def record_session_end(self) -> None:
        """Record a session ending."""
        self.active_sessions = max(0, self.active_sessions - 1)

    def get_summary(self) -> dict:
        """Return metrics summary for the /metrics endpoint."""
        avg_latency = (
            sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        )
        return {
            "llm": {
                "total_calls": self.total_llm_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cost_usd": round(self.total_cost_usd, 4),
                "avg_latency_ms": round(avg_latency, 1),
            },
            "retrieval": {
                "total_queries": self.total_retrieval_queries,
            },
            "uploads": {
                "total": self.total_uploads,
            },
            "errors": {
                "total": self.total_errors,
                "by_type": self.errors_by_type,
            },
        }


@contextmanager
def performance_timer(operation: str) -> Generator[dict, None, None]:
    """Context manager that times an operation and logs the result."""
    result: dict = {"operation": operation}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = elapsed_ms
        logger.debug("performance_timer", operation=operation, latency_ms=round(elapsed_ms, 1))


# Singleton metrics collector
metrics = MetricsCollector()
