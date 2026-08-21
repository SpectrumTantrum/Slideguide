"""Upload repository — CRUD operations for the uploads table."""
from __future__ import annotations

from typing import Any

from supabase import Client


class UploadRepository:
    def __init__(self, client: Client):
        self.client = client

    def create(
        self,
        *,
        filename: str,
        file_type: str,
        file_size: int,
        status: str = "PROCESSING",
        **kwargs: Any,
    ) -> dict:
        result = (
            self.client.table("uploads")
            .insert(
                {
                    "filename": filename,
                    "file_type": file_type,
                    "file_size": file_size,
                    "status": status,
                    **kwargs,
                }
            )
            .execute()
        )
        return result.data[0]

    def get_by_id(self, upload_id: str) -> dict | None:
        result = (
            self.client.table("uploads")
            .select("*")
            .eq("id", upload_id)
            .maybe_single()
            .execute()
        )
        # supabase-py returns None (not a response) when no row matches.
        return result.data if result is not None else None

    def update(self, upload_id: str, **data: Any) -> dict:
        result = (
            self.client.table("uploads")
            .update(data)
            .eq("id", upload_id)
            .execute()
        )
        return result.data[0]
