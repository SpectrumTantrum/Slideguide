"""Session repository — CRUD operations for the sessions table."""
from __future__ import annotations

from typing import Any

from supabase import Client


class SessionRepository:
    def __init__(self, client: Client):
        self.client = client

    def create(self, *, upload_id: str, phase: str = "GREETING", **kwargs: Any) -> dict:
        result = (
            self.client.table("sessions")
            .insert({"upload_id": upload_id, "phase": phase, **kwargs})
            .execute()
        )
        return result.data[0]

    def get_by_id(self, session_id: str) -> dict | None:
        result = (
            self.client.table("sessions")
            .select("*")
            .eq("id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data

    def update(self, session_id: str, **data: Any) -> dict:
        result = (
            self.client.table("sessions")
            .update(data)
            .eq("id", session_id)
            .execute()
        )
        return result.data[0]
