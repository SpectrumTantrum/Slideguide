"""
Retrieval evaluation and metrics logging.

Logs retrieval quality metrics via structlog for observability.
Does not perform offline evaluation (that requires ground truth labels).
"""

from __future__ import annotations

from backend.models.schemas import RetrievalResult
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)


class RetrievalEvaluator:
    """Log retrieval metrics for monitoring and debugging."""

    def log_retrieval(
        self,
        query: str,
        results: list[RetrievalResult],
        latency_ms: float,
    ) -> None:
        """Log detailed retrieval metrics for a single query."""
        if not results:
            logger.info(
                "retrieval_empty",
                query=query[:100],
                latency_ms=round(latency_ms, 1),
            )
            return

        scores = [r.score for r in results]
        sources = [r.source for r in results]
        slides = [r.metadata.slide_number for r in results]

        source_counts = {}
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        logger.info(
            "retrieval_metrics",
            query=query[:100],
            result_count=len(results),
            top_score=round(max(scores), 4),
            avg_score=round(sum(scores) / len(scores), 4),
            min_score=round(min(scores), 4),
            source_distribution=source_counts,
            unique_slides=len(set(slides)),
            latency_ms=round(latency_ms, 1),
        )
