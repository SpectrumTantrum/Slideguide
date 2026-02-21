"""Tests for the RAG pipeline components."""

import pytest
from backend.models.schemas import ChunkMetadata, RetrievalResult


class TestRetrievalResult:
    """Tests for RAG data models."""

    def test_retrieval_result_creation(self):
        """RetrievalResult can be created with required fields."""
        result = RetrievalResult(
            content="Photosynthesis is the process...",
            metadata=ChunkMetadata(
                upload_id="test-upload",
                slide_number=3,
                chunk_index=0,
                title="Photosynthesis",
                content_type="text",
            ),
            score=0.95,
            source="semantic",
        )
        assert result.content.startswith("Photosynthesis")
        assert result.metadata.slide_number == 3
        assert result.score == 0.95

    def test_chunk_metadata_defaults(self):
        """ChunkMetadata has correct defaults."""
        meta = ChunkMetadata(upload_id="up1", slide_number=1)
        assert meta.chunk_index == 0
        assert meta.title == ""
        assert meta.content_type == "text"


class TestHybridRetriever:
    """Tests for the hybrid retriever logic."""

    def test_text_similarity_identical(self):
        """Identical texts have similarity 1.0."""
        from backend.rag.retriever import HybridRetriever

        sim = HybridRetriever._text_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_text_similarity_disjoint(self):
        """Completely different texts have similarity 0.0."""
        from backend.rag.retriever import HybridRetriever

        sim = HybridRetriever._text_similarity("hello world", "foo bar")
        assert sim == 0.0

    def test_text_similarity_partial(self):
        """Partially overlapping texts have intermediate similarity."""
        from backend.rag.retriever import HybridRetriever

        sim = HybridRetriever._text_similarity("hello world foo", "hello bar foo")
        assert 0.0 < sim < 1.0

    def test_text_similarity_empty(self):
        """Empty texts return 0.0."""
        from backend.rag.retriever import HybridRetriever

        assert HybridRetriever._text_similarity("", "hello") == 0.0
        assert HybridRetriever._text_similarity("hello", "") == 0.0

    def test_result_key(self):
        """Result key combines slide and chunk index."""
        from backend.rag.retriever import HybridRetriever

        result = RetrievalResult(
            content="test",
            metadata=ChunkMetadata(
                upload_id="u1", slide_number=5, chunk_index=2
            ),
        )
        assert HybridRetriever._result_key(result) == "5:2"

    def test_reciprocal_rank_fusion(self):
        """RRF combines two result lists and deduplicates."""
        from backend.rag.retriever import HybridRetriever
        from backend.rag.vectorstore import VectorStore

        retriever = HybridRetriever(VectorStore())

        r1 = RetrievalResult(
            content="A",
            metadata=ChunkMetadata(upload_id="u", slide_number=1, chunk_index=0),
            score=0.9,
            source="semantic",
        )
        r2 = RetrievalResult(
            content="B",
            metadata=ChunkMetadata(upload_id="u", slide_number=2, chunk_index=0),
            score=0.8,
            source="semantic",
        )
        r3 = RetrievalResult(
            content="A",
            metadata=ChunkMetadata(upload_id="u", slide_number=1, chunk_index=0),
            score=5.0,
            source="keyword",
        )

        fused = retriever._reciprocal_rank_fusion([r1, r2], [r3])

        # r1/r3 should be deduplicated (same key)
        assert len(fused) == 2
        # The deduped result should have higher RRF score (appears in both lists)
        assert fused[0].metadata.slide_number == 1
        assert fused[0].source == "hybrid"
