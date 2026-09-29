from unittest.mock import MagicMock
import pytest
from wayfinder_search.hybrid_retriever import HybridRetriever, build_chroma_where_filter
from wayfinder_search.bm25 import BM25Index


def test_build_chroma_where_filter():
    # None when no filters provided
    assert build_chroma_where_filter(None, None) is None
    assert build_chroma_where_filter("", "  ") is None

    # Single category
    single_cat = build_chroma_where_filter(category="HR")
    assert single_cat == {"category": "HR"}

    # Single sub_category
    single_sub = build_chroma_where_filter(sub_category="Policies")
    assert single_sub == {"sub_category": "Policies"}

    # Both category and sub_category
    both = build_chroma_where_filter(category="HR", sub_category="Policies")
    assert both == {"$and": [{"category": "HR"}, {"sub_category": "Policies"}]}


def test_hybrid_retrieval_merging_and_deduplication():
    mock_vector_store = MagicMock()
    mock_embedding_service = MagicMock()
    mock_bm25 = MagicMock()

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        bm25_index=mock_bm25
    )

    # Mock dense search returning 2 chunks
    mock_dense_results = [
        {"chunk_id": "c1", "text": "text 1", "metadata": {"category": "HR"}, "dense_rank": 1},
        {"chunk_id": "c2", "text": "text 2", "metadata": {"category": "HR"}, "dense_rank": 2},
    ]
    # Mock lexical search returning 2 chunks (c2 overlapping with dense, and c3 new)
    mock_lexical_results = [
        {"chunk_id": "c2", "text": "text 2", "metadata": {"category": "HR"}, "lexical_rank": 1},
        {"chunk_id": "c3", "text": "text 3", "metadata": {"category": "IT"}, "lexical_rank": 2},
    ]

    retriever.retrieve_dense = MagicMock(return_value=mock_dense_results)
    retriever.retrieve_lexical = MagicMock(return_value=mock_lexical_results)

    merged = retriever.retrieve_hybrid(query="policy guidelines", dense_k=5, bm25_k=5, rerank_k=10)

    # Should contain 3 unique chunks: c1, c2, c3
    chunk_ids = [c["chunk_id"] for c in merged]
    assert len(chunk_ids) == 3
    assert set(chunk_ids) == {"c1", "c2", "c3"}

    # Chunk c2 appeared in both dense and lexical, so its RRF score must be highest!
    assert merged[0]["chunk_id"] == "c2"
    assert merged[0]["dense_rank"] == 2
    assert merged[0]["lexical_rank"] == 1
