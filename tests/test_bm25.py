import pytest
from wayfinder_search.bm25 import BM25Index, default_tokenize


def test_tokenize():
    text = "Wayfinder Enterprise Annual Leave Policy, 2026!"
    tokens = default_tokenize(text)
    assert "wayfinder" in tokens
    assert "leave" in tokens
    assert "policy" in tokens
    assert "2026" in tokens


def test_bm25_indexing_and_search():
    index = BM25Index()
    docs = [
        {
            "chunk_id": "c1",
            "text": "Wayfinder annual leave and paid vacation entitlements for employees.",
            "metadata": {"category": "HR", "sub_category": "Policies"}
        },
        {
            "chunk_id": "c2",
            "text": "IT security guidelines and password authentication standards.",
            "metadata": {"category": "IT", "sub_category": "Security"}
        },
        {
            "chunk_id": "c3",
            "text": "Customer transaction dispute redressal and refund procedures.",
            "metadata": {"category": "Payments", "sub_category": "Disputes"}
        }
    ]

    index.index_documents(docs)
    assert index.corpus_size == 3

    # Search for vacation / leave
    results = index.search("annual vacation leave", top_k=2)
    assert len(results) > 0
    assert results[0][0]["chunk_id"] == "c1"

    # Search for customer dispute
    results_dispute = index.search("customer dispute transaction", top_k=2)
    assert len(results_dispute) > 0
    assert results_dispute[0][0]["chunk_id"] == "c3"


def test_bm25_category_filtering():
    index = BM25Index()
    docs = [
        {
            "chunk_id": "c1",
            "text": "Enterprise security policy and procedures",
            "metadata": {"category": "HR", "sub_category": "General"}
        },
        {
            "chunk_id": "c2",
            "text": "Enterprise security policy and firewall standards",
            "metadata": {"category": "IT", "sub_category": "Security"}
        }
    ]
    index.index_documents(docs)

    # Search without filter -> matches both
    res_all = index.search("security policy", top_k=5)
    assert len(res_all) == 2

    # Filter by category IT
    res_it = index.search("security policy", top_k=5, category="IT")
    assert len(res_it) == 1
    assert res_it[0][0]["chunk_id"] == "c2"

    # Filter by non-existent category
    res_finance = index.search("security policy", top_k=5, category="Finance")
    assert len(res_finance) == 0


def test_bm25_empty_query():
    index = BM25Index()
    index.index_documents([{"chunk_id": "c1", "text": "hello", "metadata": {}}])
    assert index.search("") == []
    assert index.search("   ") == []
