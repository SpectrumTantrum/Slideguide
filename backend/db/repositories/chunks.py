"""Slide chunks repository — pgvector storage and search via Supabase RPC."""

from __future__ import annotations

from typing import Any

from supabase import Client


class ChunkRepository:
    """CRUD and search operations for the slide_chunks table."""

    def __init__(self, client: Client) -> None:
        self.client = client

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Batch upsert chunks with embeddings."""
        if not chunks:
            return 0
        result = self.client.table("slide_chunks").upsert(chunks).execute()
        return len(result.data)

    def semantic_search(
        self,
        query_embedding: list[float],
        upload_id: str,
        n_results: int = 10,
        slide_filter: int | None = None,
    ) -> list[dict[str, Any]]:
        """Call match_slide_chunks RPC for cosine similarity search."""
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "filter_upload_id": upload_id,
            "match_count": n_results,
        }
        if slide_filter is not None:
            params["filter_slide_number"] = slide_filter
        result = self.client.rpc("match_slide_chunks", params).execute()
        return result.data

    def text_search(
        self,
        query_text: str,
        upload_id: str,
        n_results: int = 10,
        slide_filter: int | None = None,
    ) -> list[dict[str, Any]]:
        """Call search_slide_chunks_text RPC for full-text search."""
        params: dict[str, Any] = {
            "query_text": query_text,
            "filter_upload_id": upload_id,
            "match_count": n_results,
        }
        if slide_filter is not None:
            params["filter_slide_number"] = slide_filter
        result = self.client.rpc("search_slide_chunks_text", params).execute()
        return result.data

    def delete_by_upload(self, upload_id: str) -> None:
        """Delete all chunks for an upload."""
        self.client.table("slide_chunks").delete().eq("upload_id", upload_id).execute()

    def count_by_upload(self, upload_id: str) -> int:
        """Count chunks for an upload."""
        result = (
            self.client.table("slide_chunks")
            .select("id", count="exact")
            .eq("upload_id", upload_id)
            .execute()
        )
        return result.count or 0
