"""Message repository — CRUD operations for the messages table."""
from __future__ import annotations

from typing import Any

from supabase import Client


class MessageRepository:
    def __init__(self, client: Client):
        self.client = client

    def create(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Any = None,
    ) -> dict:
        data: dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": content,
        }
        if tool_calls is not None:
            data["tool_calls"] = tool_calls
        result = self.client.table("messages").insert(data).execute()
        return result.data[0]

    def get_by_session(
        self,
        session_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        order_asc: bool = True,
    ) -> list[dict]:
        query = (
            self.client.table("messages")
            .select("*")
            .eq("session_id", session_id)
            .order("created_at", desc=not order_asc)
        )
        if offset:
            query = query.range(offset, offset + limit - 1)
        else:
            query = query.limit(limit)
        result = query.execute()
        return result.data

    def count(self, session_id: str) -> int:
        result = (
            self.client.table("messages")
            .select("*", count="exact")
            .eq("session_id", session_id)
            .execute()
        )
        return result.count or 0
