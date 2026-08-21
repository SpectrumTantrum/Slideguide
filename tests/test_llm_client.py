"""Tests for the LLM client components."""

import pytest

from backend.config import settings
from backend.llm.models import get_fallback_chain
from backend.monitoring.metrics import MetricsCollector
from backend.monitoring.metrics import estimate_cost as metrics_estimate


class TestModelSelection:
    """Tests for model selection with the single OpenAI-compatible endpoint."""

    def test_fallback_chain_is_primary_model(self):
        """The fallback chain is just the configured primary model."""
        expected = [settings.active_primary_model] if settings.active_primary_model else []
        assert get_fallback_chain() == expected

    def test_routing_falls_back_to_primary(self):
        """active_routing_model falls back to the primary model when unset."""
        assert settings.active_routing_model in (
            settings.routing_model or settings.primary_model,
            settings.primary_model,
        )

    def test_estimate_cost_unknown_model_is_free(self):
        """Unknown models (e.g. local endpoints) are tracked at $0.00."""
        assert metrics_estimate("some-local-model", 1000, 500) == 0.0


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

    def test_llm_client_breakers_are_split_by_sdk(self):
        from backend.llm.client import LLMClient

        client = LLMClient()
        client._breaker("cursor").record_failure()
        client._breaker("cursor").record_failure()
        client._breaker("cursor").record_failure()
        client._breaker("cursor").record_failure()
        client._breaker("cursor").record_failure()
        assert client._breaker("cursor").state == "open"
        assert client._breaker("openai").state == "closed"
        client.reset_breaker("cursor")
        assert client._breaker("cursor").state == "closed"

    @pytest.mark.asyncio
    async def test_stream_chat_respects_open_breaker(self, monkeypatch):
        from backend.llm.client import AllModelsExhaustedError, LLMClient
        from backend.llm.runtime import bind_chat_sdk, reset_chat_sdk

        monkeypatch.setattr(settings, "cursor_api_key", "crsr_test")
        client = LLMClient()
        for _ in range(5):
            client._breaker("cursor").record_failure()

        called = False

        async def boom(*_args, **_kwargs):
            nonlocal called
            called = True
            if False:
                yield {}

        monkeypatch.setattr(client, "_stream", boom)
        token = bind_chat_sdk("cursor")
        try:
            with pytest.raises(AllModelsExhaustedError, match="Circuit breaker open"):
                async for _ in client.stream_chat(
                    messages=[{"role": "user", "content": "hi"}]
                ):
                    pass
        finally:
            reset_chat_sdk(token)
        assert called is False


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
