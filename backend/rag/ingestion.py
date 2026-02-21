"""
RAG ingestion pipeline: chunking, embedding, and indexing.

Takes a ParsedDocument and produces:
1. Embedded chunks in ChromaDB (for semantic search)
2. A BM25 index (for keyword search)

Chunking is slide-aware: chunks never cross slide boundaries.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import settings
from backend.models.schemas import ChunkMetadata, ParsedDocument, SlideContent
from backend.monitoring.logger import get_logger
from backend.monitoring.metrics import metrics, performance_timer
from backend.rag.vectorstore import VectorStore

logger = get_logger(__name__)

# Chunking config
CHUNK_SIZE = 500  # Target tokens (~4 chars per token ≈ 2000 chars)
CHUNK_OVERLAP = 50  # Overlap tokens (~200 chars)
CHARS_PER_TOKEN = 4


class IngestionPipeline:
    """Ingest parsed documents into the vector store and BM25 index."""

    def __init__(self, vectorstore: VectorStore) -> None:
        self.vectorstore = vectorstore
        self._embedding_client: Any = None

    @property
    def embedding_client(self) -> Any:
        """Lazy-init the embedding client using the active provider."""
        if self._embedding_client is None:
            import openai

            from backend.llm.providers import get_embedding_provider_config

            config = get_embedding_provider_config()
            self._embedding_client = openai.AsyncOpenAI(**config.client_kwargs())
        return self._embedding_client

    async def ingest(self, parsed_doc: ParsedDocument) -> int:
        """
        Ingest a parsed document: chunk → embed → store.

        Returns the total number of chunks created.
        """
        upload_id = parsed_doc.upload_id
        logger.info("ingestion_start", upload_id=upload_id, total_slides=parsed_doc.total_slides)

        with performance_timer("ingestion") as timer_result:
            # Step 1: Chunk all slides
            all_chunks = self._chunk_document(parsed_doc)
            logger.info("chunking_complete", upload_id=upload_id, total_chunks=len(all_chunks))

            if not all_chunks:
                logger.warning("no_chunks_generated", upload_id=upload_id)
                return 0

            # Step 2: Generate embeddings
            texts = [c["document"] for c in all_chunks]
            embeddings = await self._embed_texts(texts)

            # Step 3: Attach embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk["embedding"] = embedding

            # Step 4: Store in ChromaDB
            self.vectorstore.add_chunks(upload_id, all_chunks)

            # Step 5: Build BM25 index
            self._build_bm25_index(upload_id, all_chunks)

        logger.info(
            "ingestion_complete",
            upload_id=upload_id,
            total_chunks=len(all_chunks),
            latency_ms=round(timer_result.get("latency_ms", 0), 1),
        )
        return len(all_chunks)

    def _chunk_document(self, parsed_doc: ParsedDocument) -> list[dict[str, Any]]:
        """Chunk all slides in a document. Never merges across slide boundaries."""
        all_chunks: list[dict[str, Any]] = []

        for slide in parsed_doc.slides:
            slide_chunks = self._chunk_slide(slide, parsed_doc.upload_id)
            all_chunks.extend(slide_chunks)

        return all_chunks

    def _chunk_slide(self, slide: SlideContent, upload_id: str) -> list[dict[str, Any]]:
        """
        Chunk a single slide's content into overlapping text chunks.

        Each slide produces at least one chunk. Long slides are split with overlap.
        Speaker notes and tables are included as separate chunks.
        """
        chunks: list[dict[str, Any]] = []
        chunk_idx = 0

        # Main text content
        if slide.text_content.strip():
            text_chunks = self._split_text(slide.text_content)
            for text in text_chunks:
                chunk_id = self._make_chunk_id(upload_id, slide.slide_number, chunk_idx)
                chunks.append({
                    "id": chunk_id,
                    "document": text,
                    "metadata": {
                        "upload_id": upload_id,
                        "slide_number": slide.slide_number,
                        "chunk_index": chunk_idx,
                        "title": slide.title or "",
                        "content_type": "text",
                    },
                })
                chunk_idx += 1

        # Speaker notes as separate chunk
        if slide.speaker_notes.strip():
            chunk_id = self._make_chunk_id(upload_id, slide.slide_number, chunk_idx)
            chunks.append({
                "id": chunk_id,
                "document": f"[Speaker Notes] {slide.speaker_notes}",
                "metadata": {
                    "upload_id": upload_id,
                    "slide_number": slide.slide_number,
                    "chunk_index": chunk_idx,
                    "title": slide.title or "",
                    "content_type": "speaker_notes",
                },
            })
            chunk_idx += 1

        # Tables as separate chunks
        for table in slide.tables:
            if table.strip():
                chunk_id = self._make_chunk_id(upload_id, slide.slide_number, chunk_idx)
                chunks.append({
                    "id": chunk_id,
                    "document": f"[Table] {table}",
                    "metadata": {
                        "upload_id": upload_id,
                        "slide_number": slide.slide_number,
                        "chunk_index": chunk_idx,
                        "title": slide.title or "",
                        "content_type": "table",
                    },
                })
                chunk_idx += 1

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks based on token estimates."""
        max_chars = CHUNK_SIZE * CHARS_PER_TOKEN
        overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOKEN

        if len(text) <= max_chars:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + max_chars

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the boundary
                for sep in [". ", ".\n", "\n\n", "\n", " "]:
                    break_pos = text.rfind(sep, start + max_chars // 2, end)
                    if break_pos != -1:
                        end = break_pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap_chars

        return chunks

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using the active embedding provider.

        Batches requests to respect API limits (max 2048 per batch).
        """
        all_embeddings: list[list[float]] = []
        batch_size = 2048
        model = settings.active_embedding_model

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            with performance_timer("embedding_batch"):
                response = await self.embedding_client.embeddings.create(
                    model=model,
                    input=batch,
                )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Track cost
            if hasattr(response, "usage"):
                metrics.record_llm_call(
                    model=model,
                    input_tokens=response.usage.total_tokens,
                    output_tokens=0,
                    latency_ms=0,
                    operation="embedding",
                    provider=settings.embedding_provider,
                )

        logger.debug("embeddings_generated", total_texts=len(texts), model=model)
        return all_embeddings

    def _build_bm25_index(
        self, upload_id: str, chunks: list[dict[str, Any]]
    ) -> None:
        """
        Build and persist a BM25 index for keyword search.

        Stored as a pickle file keyed by upload_id.
        """
        from rank_bm25 import BM25Okapi

        documents = [c["document"] for c in chunks]
        # Tokenize by splitting on whitespace (simple but effective)
        tokenized = [doc.lower().split() for doc in documents]

        bm25 = BM25Okapi(tokenized)

        # Save index and document mapping
        index_data = {
            "bm25": bm25,
            "documents": documents,
            "metadatas": [c["metadata"] for c in chunks],
            "chunk_ids": [c["id"] for c in chunks],
        }

        index_path = self._bm25_index_path(upload_id)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(index_data, f)

        logger.info("bm25_index_built", upload_id=upload_id, documents=len(documents))

    @staticmethod
    def _bm25_index_path(upload_id: str) -> Path:
        """Get the file path for a BM25 index."""
        return Path(tempfile.gettempdir()) / "slideguide" / "bm25" / f"{upload_id}.pkl"

    @staticmethod
    def _make_chunk_id(upload_id: str, slide_number: int, chunk_index: int) -> str:
        """Generate a deterministic chunk ID."""
        raw = f"{upload_id}:{slide_number}:{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()
