"""
Hybrid retrieval pipeline: semantic search + PostgreSQL full-text search.

Combines results using Reciprocal Rank Fusion (RRF) and applies
Maximum Marginal Relevance (MMR) for diversity.
"""

from __future__ import annotations

from typing import Any

from backend.config import settings
from backend.models.schemas import ChunkMetadata, RetrievalResult
from backend.monitoring.logger import get_logger
from backend.monitoring.metrics import metrics, performance_timer
from backend.rag.vectorstore import VectorStore

logger = get_logger(__name__)

# RRF constant (standard value from the literature)
RRF_K = 60

# MMR diversity parameter: 0 = max diversity, 1.0 = max relevance
MMR_LAMBDA = 0.7


class HybridRetriever:
    """
    Hybrid retrieval combining semantic and keyword search.

    Pipeline: query -> [semantic search, FTS search] -> RRF fusion -> MMR -> results
    """

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

    async def retrieve(
        self,
        query: str,
        upload_id: str,
        n_results: int = 5,
        slide_filter: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Run the full hybrid retrieval pipeline.

        1. Embed query
        2. Semantic search (pgvector)
        3. Full-text search (PostgreSQL tsvector)
        4. Reciprocal Rank Fusion
        5. Maximum Marginal Relevance for diversity
        6. Return top-n results with scores and citations
        """
        with performance_timer("hybrid_retrieval") as timer_result:
            # Step 1: Embed the query
            query_embedding = await self._embed_query(query)

            # Step 2: Semantic search
            candidate_count = n_results * 3  # Over-fetch for fusion
            where_filter = None
            if slide_filter is not None:
                where_filter = {"slide_number": slide_filter}

            semantic_results = self.vectorstore.query(
                upload_id=upload_id,
                query_embedding=query_embedding,
                n_results=candidate_count,
                where=where_filter,
            )

            # Step 3: Full-text search (replaces BM25)
            fts_results = self._text_search(query, upload_id, candidate_count, slide_filter)

            # Step 4: Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(semantic_results, fts_results)

            # Step 5: MMR for diversity
            if query_embedding and len(fused_results) > n_results:
                final_results = self._mmr_rerank(
                    fused_results, query_embedding, n_results
                )
            else:
                final_results = fused_results[:n_results]

        latency_ms = timer_result.get("latency_ms", 0)
        metrics.record_retrieval(query, len(final_results), latency_ms)

        logger.info(
            "retrieval_complete",
            upload_id=upload_id,
            query_length=len(query),
            semantic_count=len(semantic_results),
            fts_count=len(fts_results),
            fused_count=len(fused_results),
            final_count=len(final_results),
            latency_ms=round(latency_ms, 1),
        )

        return final_results

    async def _embed_query(self, query: str) -> list[float]:
        """Embed a single query string using the active embedding provider."""
        response = await self.embedding_client.embeddings.create(
            model=settings.active_embedding_model,
            input=[query],
        )
        return response.data[0].embedding

    def _text_search(
        self,
        query: str,
        upload_id: str,
        n_results: int,
        slide_filter: int | None = None,
    ) -> list[RetrievalResult]:
        """Full-text search using PostgreSQL tsvector (replaces BM25)."""
        from backend.db.client import get_supabase
        from backend.db.repositories.chunks import ChunkRepository

        repo = ChunkRepository(get_supabase())
        results = repo.text_search(query, upload_id, n_results, slide_filter)

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
                score=r["rank"],
                source="keyword",
            )
            for r in results
        ]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Merge results from semantic and keyword search using RRF.

        RRF score = sum(1 / (k + rank_i)) for each result list.
        Deduplicates by content hash.
        """
        score_map: dict[str, dict] = {}

        for rank, result in enumerate(semantic_results):
            key = self._result_key(result)
            if key not in score_map:
                score_map[key] = {"result": result, "rrf_score": 0.0}
            score_map[key]["rrf_score"] += 1.0 / (RRF_K + rank + 1)

        for rank, result in enumerate(keyword_results):
            key = self._result_key(result)
            if key not in score_map:
                score_map[key] = {"result": result, "rrf_score": 0.0}
            score_map[key]["rrf_score"] += 1.0 / (RRF_K + rank + 1)

        sorted_items = sorted(
            score_map.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        results: list[RetrievalResult] = []
        for item in sorted_items:
            result = item["result"]
            result.score = item["rrf_score"]
            result.source = "hybrid"
            results.append(result)

        return results

    def _mmr_rerank(
        self,
        results: list[RetrievalResult],
        query_embedding: list[float],
        n_results: int,
    ) -> list[RetrievalResult]:
        """
        Apply Maximum Marginal Relevance to reduce redundancy.

        MMR = lambda * sim(doc, query) - (1-lambda) * max(sim(doc, selected))
        """
        if len(results) <= n_results:
            return results

        selected: list[int] = []
        remaining = list(range(len(results)))

        # Normalize RRF scores to 0-1
        max_score = max(r.score for r in results) if results else 1.0
        if max_score == 0:
            max_score = 1.0
        relevance = [r.score / max_score for r in results]

        for _ in range(n_results):
            if not remaining:
                break

            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                rel = relevance[idx]

                max_sim_to_selected = 0.0
                if selected:
                    for sel_idx in selected:
                        sim = self._text_similarity(
                            results[idx].content, results[sel_idx].content
                        )
                        max_sim_to_selected = max(max_sim_to_selected, sim)

                mmr_score = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * max_sim_to_selected

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [results[i] for i in selected]

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """Simple Jaccard similarity between two texts (for MMR diversity)."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    @staticmethod
    def _result_key(result: RetrievalResult) -> str:
        """Generate a deduplication key for a retrieval result."""
        return f"{result.metadata.slide_number}:{result.metadata.chunk_index}"
