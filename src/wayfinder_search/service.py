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
        current_settings = get_settings()

        # 1. Determine and validate requested number of results
        if request.number_of_results is not None:
            requested_top_k = request.number_of_results
        else:
            requested_top_k = current_settings.DEFAULT_TOP_K

        # Guard against values outside min/max bounds
        effective_top_k = max(
            current_settings.MIN_TOP_K,
            min(requested_top_k, current_settings.MAX_TOP_K)
        )

        threshold_green = getattr(current_settings, "THRESHOLD_GREEN", 0.40)
        threshold_amber = getattr(current_settings, "THRESHOLD_AMBER", 0.10)
        if request.relevance_threshold is not None:
            threshold = request.relevance_threshold
        else:
            threshold = getattr(current_settings, "RELEVANCE_THRESHOLD", threshold_amber)

        logger.info(
            f"Executing search: query='{query[:50]}...', requested_k={effective_top_k}, "
            f"category={request.category}, sub_category={request.sub_category}, threshold={threshold}"
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
                threshold_green=threshold_green,
                threshold_amber=threshold_amber,
                total_results=0,
                results=[],
                below_threshold_results=None,
                message="No candidate documents found matching the search criteria or category filters."
            )

        # 3. Cross-Encoder Reranking
        reranked_candidates = self.reranker.rerank(
            query=query,
            candidates=candidates
        )

        # Helper to categorize score into Red-Amber-Green
        def get_confidence_category(score: float) -> str:
            if score >= threshold_green:
                return "green"
            elif score >= threshold_amber:
                return "amber"
            else:
                return "red"

        # 4. Relevance Threshold Filtering
        passing_candidates = [
            c for c in reranked_candidates if c["relevance_score"] >= threshold
        ]
        failing_candidates = [
            c for c in reranked_candidates if c["relevance_score"] < threshold
        ]

        # 5. Format Top-K Results with Red-Amber-Green categories
        def to_search_item(c: dict) -> SearchResultItem:
            meta = c.get("metadata", {})
            score = c["relevance_score"]
            return SearchResultItem(
                document_title=str(meta.get("doc_title") or meta.get("file_name") or "Untitled Document"),
                category=str(meta.get("category") or "General"),
                sub_category=str(meta.get("sub_category") or "General"),
                source_document=str(meta.get("file_path") or meta.get("file_name") or "Unknown"),
                relevance_score=score,
                confidence_category=get_confidence_category(score),
                chunk_text=c.get("text", ""),
                page_number=int(meta.get("page_number", 1)),
                document_id=str(meta.get("file_hash") or meta.get("file_name") or "Unknown"),
                chunk_id=str(c.get("chunk_id", "Unknown"))
            )

        threshold_met = len(passing_candidates) > 0
        final_passing = [to_search_item(c) for c in passing_candidates[:effective_top_k]]
        
        # Combined budget: remaining slots filled by below-threshold (irrelevant) items
        remaining_slots = max(0, effective_top_k - len(final_passing))
        below_threshold_items = [to_search_item(c) for c in failing_candidates[:remaining_slots]]

        duration = round(time.perf_counter() - start_time, 4)

        if threshold_met:
            if len(below_threshold_items) > 0:
                message = (
                    f"Retrieved {len(final_passing)} relevant document chunk(s) (>= {threshold}) "
                    f"and {len(below_threshold_items)} below-threshold chunk(s) to meet the requested count of {effective_top_k}."
                )
            else:
                message = (
                    f"Successfully retrieved {len(final_passing)} document chunk(s) "
                    f"exceeding relevance threshold {threshold}."
                )
        else:
            message = (
                f"No retrieved document chunk met the minimum relevance threshold of {threshold}. "
                f"Showing top {len(below_threshold_items)} candidate(s) below threshold."
            )

        logger.info(
            f"Search complete in {duration:.4f}s: {len(final_passing)} passing, "
            f"{len(below_threshold_items)} below-threshold (combined: {len(final_passing) + len(below_threshold_items)}/{effective_top_k})"
        )

        return SearchResponse(
            query=query,
            requested_results=effective_top_k,
            candidates_retrieved=candidates_count,
            search_duration_seconds=duration,
            threshold_met=threshold_met,
            relevance_threshold=threshold,
            threshold_green=threshold_green,
            threshold_amber=threshold_amber,
            total_results=len(final_passing),
            results=final_passing,
            below_threshold_results=below_threshold_items if below_threshold_items else None,
            message=message
        )
