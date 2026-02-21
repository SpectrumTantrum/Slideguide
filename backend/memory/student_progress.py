"""
Long-term student progress tracking via PostgreSQL.

Persists learning progress across sessions: topics covered,
quiz scores, confidence levels, and study recommendations.
All data stored via Prisma ORM in the StudentProgress model.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

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

    Backed by the Prisma StudentProgress model for persistence.
    Provides methods for updating scores, computing confidence,
    and suggesting what to study next.
    """

    def __init__(self, db: Any) -> None:
        """Initialize with a connected Prisma client."""
        self.db = db

    async def get_or_create(
        self,
        session_id: str,
        upload_id: str,
    ) -> dict[str, Any]:
        """Get existing progress or create a new record."""
        progress = await self.db.studentprogress.find_unique(
            where={"sessionId": session_id},
        )

        if progress:
            return self._to_dict(progress)

        progress = await self.db.studentprogress.create(
            data={
                "sessionId": session_id,
                "uploadId": upload_id,
                "topicsCovered": json.dumps([]),
                "quizScores": json.dumps({}),
                "totalQuestions": 0,
                "correctAnswers": 0,
                "confidenceLevel": 0.0,
            },
        )

        logger.info(
            "progress_created",
            session_id=session_id,
            upload_id=upload_id,
        )

        return self._to_dict(progress)

    async def update_topic_covered(
        self,
        session_id: str,
        topic: str,
    ) -> None:
        """Mark a topic as covered in this session."""
        progress = await self.db.studentprogress.find_unique(
            where={"sessionId": session_id},
        )
        if not progress:
            return

        topics = json.loads(progress.topicsCovered) if progress.topicsCovered else []
        if topic not in topics:
            topics.append(topic)
            await self.db.studentprogress.update(
                where={"sessionId": session_id},
                data={
                    "topicsCovered": json.dumps(topics),
                    "lastActive": datetime.now(),
                },
            )

    async def record_quiz_result(
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
        progress = await self.db.studentprogress.find_unique(
            where={"sessionId": session_id},
        )
        if not progress:
            return {"correct": 0, "total": 0}

        scores = json.loads(progress.quizScores) if progress.quizScores else {}
        total = progress.totalQuestions + 1
        correct = progress.correctAnswers + (1 if is_correct else 0)

        # Track per-topic scores
        if topic not in scores:
            scores[topic] = {"correct": 0, "total": 0, "partial_sum": 0.0}

        scores[topic]["total"] += 1
        if is_correct:
            scores[topic]["correct"] += 1
        scores[topic]["partial_sum"] += partial_credit

        # Recompute confidence
        confidence = self._compute_confidence(correct, total, scores)

        await self.db.studentprogress.update(
            where={"sessionId": session_id},
            data={
                "quizScores": json.dumps(scores),
                "totalQuestions": total,
                "correctAnswers": correct,
                "confidenceLevel": confidence,
                "lastActive": datetime.now(),
            },
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

    async def get_progress(self, session_id: str) -> dict[str, Any] | None:
        """Get full progress data for a session."""
        progress = await self.db.studentprogress.find_unique(
            where={"sessionId": session_id},
        )
        if not progress:
            return None

        return self._to_dict(progress)

    async def suggest_next_topic(
        self,
        session_id: str,
        available_topics: list[str],
    ) -> str | None:
        """
        Suggest the best topic to study next.

        Priority: uncovered topics > low-confidence topics > least-recent topics.
        """
        progress = await self.db.studentprogress.find_unique(
            where={"sessionId": session_id},
        )
        if not progress:
            return available_topics[0] if available_topics else None

        covered = json.loads(progress.topicsCovered) if progress.topicsCovered else []
        scores = json.loads(progress.quizScores) if progress.quizScores else {}

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
            # Suggest the weakest topic
            weakest = topic_accuracy[0]
            if weakest[1] < 0.8:  # Only suggest if accuracy below 80%
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
            # Not enough data — return a scaled preliminary score
            return (correct / max(total, 1)) * 0.3

        # Quiz accuracy component (0-1)
        quiz_accuracy = correct / total

        # Topic coverage component: what fraction of quizzed topics
        # have accuracy >= 70%?
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
    def _to_dict(progress: Any) -> dict[str, Any]:
        """Convert a Prisma StudentProgress record to a plain dict."""
        topics = json.loads(progress.topicsCovered) if progress.topicsCovered else []
        scores = json.loads(progress.quizScores) if progress.quizScores else {}

        return {
            "session_id": progress.sessionId,
            "upload_id": progress.uploadId,
            "topics_covered": topics,
            "quiz_scores": scores,
            "total_questions": progress.totalQuestions,
            "correct_answers": progress.correctAnswers,
            "confidence_level": progress.confidenceLevel,
            "last_active": progress.lastActive.isoformat() if progress.lastActive else None,
        }
