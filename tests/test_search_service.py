from unittest.mock import MagicMock
import pytest
from wayfinder_search.service import SearchService
from wayfinder_search.models import SearchRequest
from wayfinder_ingestion.config import get_settings


def test_search_service_default_and_explicit_number_of_results():
    settings = get_settings()
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()

    service = SearchService(hybrid_retriever=mock_retriever, reranker=mock_reranker)

    # Mock 10 candidates with high relevance scores >= threshold
    mock_candidates = [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"Content for chunk {i}",
            "metadata": {"doc_title": "Doc A", "category": "HR", "sub_category": "Policies", "page_number": 1},
            "relevance_score": 0.85 - (i * 0.02)
        }
        for i in range(10)
    ]
    mock_retriever.retrieve_hybrid.return_value = mock_candidates
    mock_reranker.rerank.return_value = mock_candidates

    # Case 1: Omitted number_of_results -> should use DEFAULT_TOP_K
    req_default = SearchRequest(query="leave entitlement")
    res_default = service.search(req_default)
    assert res_default.requested_results == settings.DEFAULT_TOP_K
    assert len(res_default.results) == settings.DEFAULT_TOP_K
    assert res_default.threshold_met is True

    # Case 2: Explicit number_of_results = 3
    req_3 = SearchRequest(query="leave entitlement", number_of_results=3)
    res_3 = service.search(req_3)
    assert res_3.requested_results == 3
    assert len(res_3.results) == 3

    # Case 3: Explicit number_of_results = 8
    req_8 = SearchRequest(query="leave entitlement", number_of_results=8)
    res_8 = service.search(req_8)
    assert res_8.requested_results == 8
    assert len(res_8.results) == 8


def test_search_service_relevance_threshold_filtering():
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()

    service = SearchService(hybrid_retriever=mock_retriever, reranker=mock_reranker)

    # Reranker returns 2 above threshold (0.20), 1 below threshold (0.05)
    mock_reranked = [
        {
            "chunk_id": "c1",
            "text": "Highly relevant content",
            "metadata": {"doc_title": "Policy", "category": "HR", "sub_category": "General", "page_number": 1},
            "relevance_score": 0.75
        },
        {
            "chunk_id": "c2",
            "text": "Partially relevant content",
            "metadata": {"doc_title": "Policy", "category": "HR", "sub_category": "General", "page_number": 2},
            "relevance_score": 0.40
        },
        {
            "chunk_id": "c3",
            "text": "Irrelevant noise",
            "metadata": {"doc_title": "Noise", "category": "HR", "sub_category": "General", "page_number": 3},
            "relevance_score": 0.05
        }
    ]
    mock_retriever.retrieve_hybrid.return_value = mock_reranked
    mock_reranker.rerank.return_value = mock_reranked

    req = SearchRequest(query="policy questions", number_of_results=5)
    res = service.search(req)

    assert res.threshold_met is True
    assert len(res.results) == 2
    assert res.results[0].chunk_id == "c1"
    assert res.results[0].confidence_category == "green"
    assert res.results[1].chunk_id == "c2"
    assert res.results[1].confidence_category == "green"
    # Combined budget: 5 requested - 2 passing = 3 slots for below threshold (only 1 available)
    assert len(res.below_threshold_results) == 1
    assert res.below_threshold_results[0].chunk_id == "c3"
    assert res.below_threshold_results[0].confidence_category == "red"


def test_search_service_no_results_scenario():
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()

    service = SearchService(hybrid_retriever=mock_retriever, reranker=mock_reranker)

    # All candidates have low scores below threshold (e.g. 0.02 < 0.10)
    mock_reranked = [
        {
            "chunk_id": "c_low",
            "text": "Irrelevant content",
            "metadata": {"doc_title": "Other", "category": "IT", "sub_category": "General", "page_number": 1},
            "relevance_score": 0.02
        }
    ]
    mock_retriever.retrieve_hybrid.return_value = mock_reranked
    mock_reranker.rerank.return_value = mock_reranked

    req = SearchRequest(query="random quantum mechanics", number_of_results=5)
    res = service.search(req)

    assert res.threshold_met is False
    assert len(res.results) == 0
    assert res.total_results == 0
    assert "No retrieved document chunk met the minimum relevance threshold" in res.message
    assert res.below_threshold_results is not None
    assert len(res.below_threshold_results) == 1
    assert res.below_threshold_results[0].relevance_score == 0.02
    assert res.below_threshold_results[0].confidence_category == "red"


def test_search_service_combined_max_retrieval_and_amber_category():
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()

    service = SearchService(hybrid_retriever=mock_retriever, reranker=mock_reranker)

    # 2 passing (1 green >=0.40, 1 amber 0.10-0.40) and 5 failing (<0.10)
    candidates = [
        {"chunk_id": "c_green", "text": "high match", "metadata": {}, "relevance_score": 0.55},
        {"chunk_id": "c_amber", "text": "mid match", "metadata": {}, "relevance_score": 0.25},
        {"chunk_id": "c_fail_1", "text": "low match 1", "metadata": {}, "relevance_score": 0.08},
        {"chunk_id": "c_fail_2", "text": "low match 2", "metadata": {}, "relevance_score": 0.05},
        {"chunk_id": "c_fail_3", "text": "low match 3", "metadata": {}, "relevance_score": 0.02},
        {"chunk_id": "c_fail_4", "text": "low match 4", "metadata": {}, "relevance_score": 0.01},
    ]
    mock_retriever.retrieve_hybrid.return_value = candidates
    mock_reranker.rerank.return_value = candidates

    # User requested 5 total: 2 relevant, so irrelevant should show next 3 (2 + 3 = 5)
    req = SearchRequest(query="test", number_of_results=5)
    res = service.search(req)

    assert len(res.results) == 2
    assert res.results[0].confidence_category == "green"
    assert res.results[1].confidence_category == "amber"

    assert res.below_threshold_results is not None
    assert len(res.below_threshold_results) == 3
    assert res.below_threshold_results[0].chunk_id == "c_fail_1"
    assert res.below_threshold_results[0].confidence_category == "red"
    assert res.below_threshold_results[2].chunk_id == "c_fail_3"
    assert res.below_threshold_results[2].confidence_category == "red"

    # Total combined items returned is exactly 5
    assert len(res.results) + len(res.below_threshold_results) == 5
