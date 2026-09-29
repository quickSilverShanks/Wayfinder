from unittest.mock import MagicMock, patch
import pytest
from wayfinder_search.reranker import BGEReranker
from wayfinder_ingestion.config import get_settings


def test_reranker_picks_model_from_settings():
    settings = get_settings()
    reranker = BGEReranker()
    # Ensure model_name matches settings.RERANKER_MODEL and is not hard-coded
    assert reranker.model_name == settings.RERANKER_MODEL
    assert "BAAI" in reranker.model_name or "bge" in reranker.model_name


def test_reranker_ordering_and_top_k():
    reranker = BGEReranker()
    # Mock compute_scores to return deterministic scores without GPU/CPU heavy model load
    mock_scores = [0.15, 0.92, 0.45]
    with patch.object(reranker, "compute_scores", return_value=mock_scores):
        candidates = [
            {"chunk_id": "c1", "text": "chunk 1 text", "metadata": {}},
            {"chunk_id": "c2", "text": "chunk 2 text", "metadata": {}},
            {"chunk_id": "c3", "text": "chunk 3 text", "metadata": {}},
        ]
        results = reranker.rerank(query="test query", candidates=candidates)

        assert len(results) == 3
        # Should be ordered descending: c2 (0.92), c3 (0.45), c1 (0.15)
        assert results[0]["chunk_id"] == "c2"
        assert results[0]["relevance_score"] == 0.92
        assert results[1]["chunk_id"] == "c3"
        assert results[1]["relevance_score"] == 0.45
        assert results[2]["chunk_id"] == "c1"
        assert results[2]["relevance_score"] == 0.15

        # Test top_k parameter
        top_1 = reranker.rerank(query="test query", candidates=candidates, top_k=1)
        assert len(top_1) == 1
        assert top_1[0]["chunk_id"] == "c2"


def test_reranker_empty_candidates():
    reranker = BGEReranker()
    results = reranker.rerank(query="test", candidates=[])
    assert results == []
