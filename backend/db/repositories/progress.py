"""Student progress repository — CRUD operations for the student_progress table."""
from __future__ import annotations

from typing import Any

from supabase import Client


class ProgressRepository:
    def __init__(self, client: Client):
        self.client = client

    def create(
        self,
        *,
        session_id: str,
        upload_id: str,
        topics_covered: list | None = None,
        quiz_scores: dict | None = None,
        total_questions: int = 0,
        correct_answers: int = 0,
        confidence_level: float = 0.0,
    ) -> dict:
        result = (
            self.client.table("student_progress")
            .insert(
                {
                    "session_id": session_id,
                    "upload_id": upload_id,
                    "topics_covered": topics_covered or [],
                    "quiz_scores": quiz_scores or {},
                    "total_questions": total_questions,
                    "correct_answers": correct_answers,
                    "confidence_level": confidence_level,
                }
            )
            .execute()
        )
        return result.data[0]

    def get_by_session_id(self, session_id: str) -> dict | None:
        result = (
            self.client.table("student_progress")
            .select("*")
            .eq("session_id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data

    def update_by_session_id(self, session_id: str, **data: Any) -> dict:
        result = (
            self.client.table("student_progress")
            .update(data)
            .eq("session_id", session_id)
            .execute()
        )
        return result.data[0]
