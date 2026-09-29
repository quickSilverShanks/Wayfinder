"""
Search Service orchestrating Hybrid Retrieval, Cross-Encoder Reranking,
and Relevance Threshold Filtering for Wayfinder.
"""

import time
from typing import Optional, List
from wayfinder_ingestion.config import get_settings
from wayfinder_ingestion.logging_config import setup_logger
from wayfinder_search.hybrid_retriever import HybridRetriever
from wayfinder_search.reranker import BGEReranker, get_reranker
from wayfinder_search.models import SearchRequest, SearchResponse, SearchResultItem

logger = setup_logger(__name__)


class SearchService:
    """
    Core search business logic decoupling FastAPI HTTP transport from retrieval engines.
    Can be invoked by FastAPI, MCP servers, or background tasks.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[BGEReranker] = None
    ):
        self.settings = get_settings()
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()
        self.reranker = reranker or get_reranker()

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Executes end-to-end intelligent search:
        Query -> Hybrid Retrieval -> BGE Reranker -> Threshold Filtering -> Top-K
        """
        start_time = time.perf_counter()
        query = request.query.strip()

        # 1. Determine and validate requested number of results
        if request.number_of_results is not None:
            requested_top_k = request.number_of_results
        else:
            requested_top_k = self.settings.DEFAULT_TOP_K

        # Guard against values outside min/max bounds
        effective_top_k = max(
            self.settings.MIN_TOP_K,
            min(requested_top_k, self.settings.MAX_TOP_K)
        )

        threshold = self.settings.RELEVANCE_THRESHOLD
        logger.info(
            f"Executing search: query='{query[:50]}...', requested_k={effective_top_k}, "
            f"category={request.category}, sub_category={request.sub_category}"
        )

        # 2. Hybrid Candidate Retrieval (Dense Vector + BM25 Lexical)
        candidates = self.hybrid_retriever.retrieve_hybrid(
            query=query,
            category=request.category,
            sub_category=request.sub_category
        )
        candidates_count = len(candidates)

        # Handle zero candidates retrieved from stores
        if candidates_count == 0:
            duration = round(time.perf_counter() - start_time, 4)
            logger.info("Zero candidate chunks found across dense and lexical indexes.")
            return SearchResponse(
                query=query,
                requested_results=effective_top_k,
                candidates_retrieved=0,
                search_duration_seconds=duration,
                threshold_met=False,
                relevance_threshold=threshold,
                total_results=0,
                results=[],
                below_threshold_results=[],
                message="No candidate documents found matching the search criteria or category filters."
            )

        # 3. Cross-Encoder Reranking
        reranked_candidates = self.reranker.rerank(
            query=query,
            candidates=candidates
        )

        # 4. Relevance Threshold Filtering
        passing_candidates = [
            c for c in reranked_candidates if c["relevance_score"] >= threshold
        ]
        failing_candidates = [
            c for c in reranked_candidates if c["relevance_score"] < threshold
        ]

        # 5. Format Top-K Results
        def to_search_item(c: dict) -> SearchResultItem:
            meta = c.get("metadata", {})
            return SearchResultItem(
                document_title=str(meta.get("doc_title") or meta.get("file_name") or "Untitled Document"),
                category=str(meta.get("category") or "General"),
                sub_category=str(meta.get("sub_category") or "General"),
                source_document=str(meta.get("file_path") or meta.get("file_name") or "Unknown"),
                relevance_score=c["relevance_score"],
                chunk_text=c.get("text", ""),
                page_number=int(meta.get("page_number", 1)),
                document_id=str(meta.get("file_hash") or meta.get("file_name") or "Unknown"),
                chunk_id=str(c.get("chunk_id", "Unknown"))
            )

        threshold_met = len(passing_candidates) > 0
        final_passing = [to_search_item(c) for c in passing_candidates[:effective_top_k]]
        below_threshold_items = [to_search_item(c) for c in failing_candidates[:10]]

        duration = round(time.perf_counter() - start_time, 4)

        if threshold_met:
            message = (
                f"Successfully retrieved {len(final_passing)} document chunk(s) "
                f"exceeding relevance threshold {threshold}."
            )
        else:
            message = (
                f"No retrieved document chunk met the minimum relevance threshold of {threshold}. "
                f"Retrieved {candidates_count} candidate(s) but none demonstrated sufficient semantic confidence."
            )

        logger.info(
            f"Search complete in {duration:.4f}s: {len(final_passing)} results passing threshold ({threshold_met})"
        )

        return SearchResponse(
            query=query,
            requested_results=effective_top_k,
            candidates_retrieved=candidates_count,
            search_duration_seconds=duration,
            threshold_met=threshold_met,
            relevance_threshold=threshold,
            total_results=len(final_passing),
            results=final_passing,
            below_threshold_results=below_threshold_items if not threshold_met else None,
            message=message
        )
