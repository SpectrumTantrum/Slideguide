"""
ChromaDB vector store wrapper.

Manages collections per upload, handles document storage with metadata,
and provides semantic similarity search.
"""

from __future__ import annotations

from typing import Any

import chromadb

from backend.config import settings
from backend.models.schemas import ChunkMetadata, RetrievalResult
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Interface to ChromaDB for storing and querying slide embeddings."""

    def __init__(self) -> None:
        self._client: chromadb.HttpClient | None = None

    @property
    def client(self) -> chromadb.HttpClient:
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=settings.chromadb_host,
                port=settings.chromadb_port,
            )
            logger.info(
                "chromadb_connected",
                host=settings.chromadb_host,
                port=settings.chromadb_port,
            )
        return self._client

    def _collection_name(self, upload_id: str) -> str:
        """Generate a valid ChromaDB collection name from upload_id."""
        # ChromaDB requires 3-63 chars, alphanumeric + underscores/hyphens
        safe_id = upload_id.replace("-", "_")[:50]
        return f"upload_{safe_id}"

    def get_or_create_collection(self, upload_id: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection for an upload."""
        name = self._collection_name(upload_id)
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("collection_accessed", name=name, upload_id=upload_id)
        return collection

    def add_chunks(
        self,
        upload_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """
        Add document chunks to the vector store.

        Each chunk dict must have: id, document, embedding, metadata.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(upload_id)

        ids = [c["id"] for c in chunks]
        documents = [c["document"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        # ChromaDB supports batch upsert
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "chunks_added",
            upload_id=upload_id,
            count=len(chunks),
        )
        return len(chunks)

    def query(
        self,
        upload_id: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Query the vector store for similar chunks.

        Returns RetrievalResult objects with content, metadata, and similarity scores.
        """
        collection = self.get_or_create_collection(upload_id)

        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, collection.count() or n_results),
        }
        if where:
            query_params["where"] = where

        results = collection.query(**query_params)

        retrieval_results: list[RetrievalResult] = []

        if not results["documents"] or not results["documents"][0]:
            return retrieval_results

        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            similarity = 1.0 - (distance / 2.0)

            retrieval_results.append(
                RetrievalResult(
                    content=doc,
                    metadata=ChunkMetadata(
                        upload_id=meta.get("upload_id", upload_id),
                        slide_number=meta.get("slide_number", 0),
                        chunk_index=meta.get("chunk_index", 0),
                        title=meta.get("title", ""),
                        content_type=meta.get("content_type", "text"),
                    ),
                    score=similarity,
                    source="semantic",
                )
            )

        return retrieval_results

    def delete_collection(self, upload_id: str) -> None:
        """Delete the collection for an upload (cleanup)."""
        name = self._collection_name(upload_id)
        try:
            self.client.delete_collection(name)
            logger.info("collection_deleted", name=name, upload_id=upload_id)
        except Exception as e:
            logger.warning("collection_delete_failed", name=name, error=str(e))

    def collection_count(self, upload_id: str) -> int:
        """Return the number of chunks in a collection."""
        collection = self.get_or_create_collection(upload_id)
        return collection.count()
