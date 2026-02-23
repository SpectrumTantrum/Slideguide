"""
Long-term student progress tracking via Supabase PostgreSQL.

Persists learning progress across sessions: topics covered,
quiz scores, confidence levels, and study recommendations.
All data stored via Supabase in the student_progress table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.db.repositories.progress import ProgressRepository
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

# Confidence calculation weights
QUIZ_WEIGHT = 0.6
COVERAGE_WEIGHT = 0.4

# Minimum quiz attempts before confidence is meaningful
MIN_QUIZ_ATTEMPTS = 3


class StudentProgressTracker:
    """
    Track and query student learning progress.

    Backed by Supabase student_progress table for persistence.
    Provides methods for updating scores, computing confidence,
    and suggesting what to study next.
    """

    def __init__(self, supabase: Any) -> None:
        """Initialize with a Supabase client."""
        self.repo = ProgressRepository(supabase)

    def get_or_create(
        self,
        session_id: str,
        upload_id: str,
    ) -> dict[str, Any]:
        """Get existing progress or create a new record."""
        progress = self.repo.get_by_session_id(session_id)

        if progress:
            return self._to_dict(progress)

        progress = self.repo.create(
            session_id=session_id,
            upload_id=upload_id,
            topics_covered=[],
            quiz_scores={},
            total_questions=0,
            correct_answers=0,
            confidence_level=0.0,
        )

        logger.info(
            "progress_created",
            session_id=session_id,
            upload_id=upload_id,
        )

        return self._to_dict(progress)

    def update_topic_covered(
        self,
        session_id: str,
        topic: str,
    ) -> None:
        """Mark a topic as covered in this session."""
        progress = self.repo.get_by_session_id(session_id)
        if not progress:
            return

        # JSONB returns native list — no json.loads needed
        topics = progress["topics_covered"] or []
        if topic not in topics:
            topics.append(topic)
            self.repo.update_by_session_id(
                session_id,
                topics_covered=topics,
                last_active=datetime.now().isoformat(),
            )

    def record_quiz_result(
        self,
        session_id: str,
        topic: str,
        is_correct: bool,
        partial_credit: float = 0.0,
    ) -> dict[str, Any]:
        """
        Record a quiz answer and update scores.

        Returns the updated quiz score summary.
        """
        progress = self.repo.get_by_session_id(session_id)
        if not progress:
            return {"correct": 0, "total": 0}

        # JSONB returns native dict — no json.loads needed
        scores = progress["quiz_scores"] or {}
        total = progress["total_questions"] + 1
        correct = progress["correct_answers"] + (1 if is_correct else 0)

        # Track per-topic scores
        if topic not in scores:
            scores[topic] = {"correct": 0, "total": 0, "partial_sum": 0.0}

        scores[topic]["total"] += 1
        if is_correct:
            scores[topic]["correct"] += 1
        scores[topic]["partial_sum"] += partial_credit

        # Recompute confidence
        confidence = self._compute_confidence(correct, total, scores)

        self.repo.update_by_session_id(
            session_id,
            quiz_scores=scores,
            total_questions=total,
            correct_answers=correct,
            confidence_level=confidence,
            last_active=datetime.now().isoformat(),
        )

        logger.info(
            "quiz_result_recorded",
            session_id=session_id,
            topic=topic,
            is_correct=is_correct,
            total=total,
            correct=correct,
            confidence=round(confidence, 3),
        )

        return {
            "correct": correct,
            "total": total,
            "by_topic": scores,
            "confidence": confidence,
        }

    def get_progress(self, session_id: str) -> dict[str, Any] | None:
        """Get full progress data for a session."""
        progress = self.repo.get_by_session_id(session_id)
        if not progress:
            return None

        return self._to_dict(progress)

    def suggest_next_topic(
        self,
        session_id: str,
        available_topics: list[str],
    ) -> str | None:
        """
        Suggest the best topic to study next.

        Priority: uncovered topics > low-confidence topics > least-recent topics.
        """
        progress = self.repo.get_by_session_id(session_id)
        if not progress:
            return available_topics[0] if available_topics else None

        covered = progress["topics_covered"] or []
        scores = progress["quiz_scores"] or {}

        # Priority 1: topics not yet covered
        uncovered = [t for t in available_topics if t not in covered]
        if uncovered:
            return uncovered[0]

        # Priority 2: topics with lowest quiz accuracy
        topic_accuracy: list[tuple[str, float]] = []
        for topic in available_topics:
            if topic in scores and scores[topic]["total"] > 0:
                accuracy = scores[topic]["correct"] / scores[topic]["total"]
                topic_accuracy.append((topic, accuracy))

        if topic_accuracy:
            topic_accuracy.sort(key=lambda x: x[1])
            weakest = topic_accuracy[0]
            if weakest[1] < 0.8:
                return weakest[0]

        # Priority 3: first available topic for review
        return available_topics[0] if available_topics else None

    def _compute_confidence(
        self,
        correct: int,
        total: int,
        by_topic: dict[str, Any],
    ) -> float:
        """
        Compute overall confidence level (0.0 to 1.0).

        Blends quiz accuracy with topic coverage breadth.
        Requires minimum attempts before confidence is meaningful.
        """
        if total < MIN_QUIZ_ATTEMPTS:
            return (correct / max(total, 1)) * 0.3

        quiz_accuracy = correct / total

        topics_with_scores = [
            t for t in by_topic.values() if t.get("total", 0) > 0
        ]
        if topics_with_scores:
            strong_topics = sum(
                1 for t in topics_with_scores
                if (t["correct"] / t["total"]) >= 0.7
            )
            coverage_score = strong_topics / len(topics_with_scores)
        else:
            coverage_score = 0.0

        confidence = (QUIZ_WEIGHT * quiz_accuracy) + (COVERAGE_WEIGHT * coverage_score)
        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _to_dict(progress: dict) -> dict[str, Any]:
        """Convert a student_progress row dict to a standard dict."""
        return {
            "session_id": progress["session_id"],
            "upload_id": progress["upload_id"],
            "topics_covered": progress["topics_covered"] or [],
            "quiz_scores": progress["quiz_scores"] or {},
            "total_questions": progress["total_questions"],
            "correct_answers": progress["correct_answers"],
            "confidence_level": progress["confidence_level"],
            "last_active": progress["last_active"],
        }
