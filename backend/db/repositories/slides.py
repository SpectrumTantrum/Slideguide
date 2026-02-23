"""Slide repository — CRUD operations for the slides table."""
from __future__ import annotations

from typing import Any

from supabase import Client


class SlideRepository:
    def __init__(self, client: Client):
        self.client = client

    def create(
        self,
        *,
        upload_id: str,
        slide_number: int,
        text_content: str,
        title: str | None = None,
        has_images: bool = False,
        image_paths: Any = None,
        metadata: Any = None,
    ) -> dict:
        data: dict[str, Any] = {
            "upload_id": upload_id,
            "slide_number": slide_number,
            "text_content": text_content,
            "title": title,
            "has_images": has_images,
        }
        if image_paths is not None:
            data["image_paths"] = image_paths
        if metadata is not None:
            data["metadata"] = metadata

        result = self.client.table("slides").insert(data).execute()
        return result.data[0]

    def get_by_upload(self, upload_id: str) -> list[dict]:
        result = (
            self.client.table("slides")
            .select("*")
            .eq("upload_id", upload_id)
            .order("slide_number")
            .execute()
        )
        return result.data

    def get_by_upload_and_number(
        self, upload_id: str, slide_number: int
    ) -> dict | None:
        result = (
            self.client.table("slides")
            .select("*")
            .eq("upload_id", upload_id)
            .eq("slide_number", slide_number)
            .maybe_single()
            .execute()
        )
        return result.data
