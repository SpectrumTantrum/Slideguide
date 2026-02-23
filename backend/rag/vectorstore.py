"""
pgvector-based vector store (replaces ChromaDB).

Stores slide chunk embeddings in Supabase PostgreSQL with pgvector.
Search via SQL RPC functions for cosine similarity.
"""

from __future__ import annotations

from typing import Any

from backend.db.client import get_supabase
from backend.db.repositories.chunks import ChunkRepository
from backend.models.schemas import ChunkMetadata, RetrievalResult
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Interface to pgvector for storing and querying slide embeddings."""

    def __init__(self) -> None:
        self._repo: ChunkRepository | None = None

    @property
    def repo(self) -> ChunkRepository:
        if self._repo is None:
            self._repo = ChunkRepository(get_supabase())
        return self._repo

    def add_chunks(
        self,
        upload_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Add document chunks with embeddings to pgvector."""
        if not chunks:
            return 0

        rows = []
        for c in chunks:
            meta = c.get("metadata", {})
            rows.append({
                "id": c["id"],
                "upload_id": meta.get("upload_id", upload_id),
                "slide_number": meta.get("slide_number", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "title": meta.get("title", ""),
                "content_type": meta.get("content_type", "text"),
                "content": c["document"],
                "embedding": c["embedding"],
                "metadata": meta,
            })

        count = self.repo.upsert_chunks(rows)
        logger.info("chunks_added", upload_id=upload_id, count=count)
        return count

    def query(
        self,
        upload_id: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Query pgvector for similar chunks."""
        slide_filter = where.get("slide_number") if where else None

        results = self.repo.semantic_search(
            query_embedding=query_embedding,
            upload_id=upload_id,
            n_results=n_results,
            slide_filter=slide_filter,
        )

        return [
            RetrievalResult(
                content=r["content"],
                metadata=ChunkMetadata(
                    upload_id=r["upload_id"],
                    slide_number=r["slide_number"],
                    chunk_index=r["chunk_index"],
                    title=r["title"],
                    content_type=r["content_type"],
                ),
                score=r["similarity"],
                source="semantic",
            )
            for r in results
        ]

    def delete_collection(self, upload_id: str) -> None:
        """Delete all chunks for an upload."""
        self.repo.delete_by_upload(upload_id)
        logger.info("chunks_deleted", upload_id=upload_id)

    def collection_count(self, upload_id: str) -> int:
        """Return the number of chunks for an upload."""
        return self.repo.count_by_upload(upload_id)
