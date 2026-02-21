"""Tests for the LLM client components."""

import pytest
from backend.llm.models import MODELS, FALLBACK_CHAIN, estimate_cost
from backend.monitoring.metrics import MetricsCollector, estimate_cost as metrics_estimate


class TestModelConfig:
    """Tests for model configuration."""

    def test_models_registry_not_empty(self):
        """MODELS registry has entries."""
        assert len(MODELS) > 0

    def test_fallback_chain_not_empty(self):
        """FALLBACK_CHAIN has entries."""
        assert len(FALLBACK_CHAIN) > 0

    def test_all_fallback_models_in_registry(self):
        """Every model in the fallback chain is in the registry."""
        for model_id in FALLBACK_CHAIN:
            assert model_id in MODELS, f"Fallback model {model_id} not in MODELS"

    def test_estimate_cost_known_model(self):
        """Cost estimation works for known models."""
        cost = estimate_cost("anthropic/claude-sonnet-4", 1000, 500)
        assert cost > 0

    def test_estimate_cost_unknown_model(self):
        """Cost estimation falls back for unknown models."""
        cost = estimate_cost("unknown/model", 1000, 500)
        assert cost > 0  # Uses default pricing


class TestCircuitBreaker:
    """Tests for the circuit breaker pattern."""

    def test_circuit_starts_closed(self):
        """Circuit breaker starts in closed state."""
        from backend.llm.client import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.can_execute() is True

    def test_circuit_opens_after_threshold(self):
        """Circuit opens after enough failures."""
        from backend.llm.client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert cb.can_execute() is False

    def test_circuit_resets_on_success(self):
        """Success resets failure count."""
        from backend.llm.client import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb.state == "closed"
        assert cb.failure_count == 0


class TestMetricsCollector:
    """Tests for the metrics collector."""

    def test_record_llm_call(self):
        """Recording a call updates all counters."""
        mc = MetricsCollector()
        cost = mc.record_llm_call(
            model="anthropic/claude-sonnet-4",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
            operation="chat",
        )

        assert mc.total_llm_calls == 1
        assert mc.total_input_tokens == 100
        assert mc.total_output_tokens == 50
        assert mc.total_cost_usd > 0
        assert cost > 0
        assert mc.model_call_counts["anthropic/claude-sonnet-4"] == 1

    def test_record_error(self):
        """Recording errors updates counters."""
        mc = MetricsCollector()
        mc.record_error("TimeoutError", "connection timed out")

        assert mc.total_errors == 1
        assert mc.errors_by_type["TimeoutError"] == 1

    def test_record_session(self):
        """Session tracking works."""
        mc = MetricsCollector()
        mc.record_session_start()
        mc.record_session_start()

        assert mc.total_sessions == 2
        assert mc.active_sessions == 2

        mc.record_session_end()
        assert mc.active_sessions == 1

    def test_get_summary(self):
        """Summary returns all expected keys."""
        mc = MetricsCollector()
        summary = mc.get_summary()

        assert "llm" in summary
        assert "retrieval" in summary
        assert "uploads" in summary
        assert "errors" in summary
