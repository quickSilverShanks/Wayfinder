#!/usr/bin/env python
"""
Wayfinder Part 2 — Search & Retrieval Backend Verification Script.
Validates the complete search flow:
Hybrid Retrieval (ChromaDB Dense + BM25 Lexical) -> BGE Reranker -> Relevance Threshold -> Top-K Results.

Usage:
    python verify_search_backend.py
"""

import sys
import json
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi.testclient import TestClient
from wayfinder_search.api import app
from wayfinder_ingestion.config import get_settings


def print_banner(text: str):
    print(f"\n{'=' * 25} {text} {'=' * 25}")


def main():
    settings = get_settings()
    print_banner("Wayfinder Search & Retrieval Backend Verification")
    print(f"ChromaDB Persist Dir  : {settings.CHROMA_PERSIST_DIR}")
    print(f"Ollama Base URL       : {settings.OLLAMA_BASE_URL}")
    print(f"Embedding Model       : {settings.EMBEDDING_MODEL}")
    print(f"Reranker Model        : {settings.RERANKER_MODEL}")
    print(f"Relevance Threshold   : {settings.RELEVANCE_THRESHOLD}")
    print(f"Default Top-K         : {settings.DEFAULT_TOP_K}")

    client = TestClient(app)

    # 1. Health Check
    print_banner("1. Testing Health Endpoint: GET /api/health")
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Expected 200, got {res_health.status_code}"
    health_data = res_health.json()
    print("Health Status:", health_data.get("status"))
    print("Total Indexed Chunks in ChromaDB:", health_data.get("indexed_chunks_in_chroma"))
    print("Active Embedding Model:", health_data.get("embedding_model"))
    print("Active Reranker Model:", health_data.get("reranker_model"))

    # 2. Query with default number_of_results
    query_text = "How should I handle a customer disputing a transaction?"
    print_banner(f"2. Testing Search with Default Top-K ({settings.DEFAULT_TOP_K})")
    print(f"Query: '{query_text}'")

    t0 = time.time()
    res_default = client.post("/api/search", json={"query": query_text})
    t_default = time.time() - t0

    assert res_default.status_code == 200, f"Search failed: {res_default.text}"
    data_default = res_default.json()
    print(f"HTTP Status           : {res_default.status_code}")
    print(f"Requested Results     : {data_default['requested_results']} (Default)")
    print(f"Candidates Retrieved  : {data_default['candidates_retrieved']}")
    print(f"Threshold Met         : {data_default['threshold_met']}")
    print(f"Total Results Returned: {data_default['total_results']}")
    print(f"Search Duration       : {data_default['search_duration_seconds']}s (Total roundtrip: {t_default:.2f}s)")
    print(f"Status Message        : {data_default['message']}")

    if data_default["results"]:
        top_res = data_default["results"][0]
        print(f"\nTop Result Chunk:")
        print(f"  Title     : {top_res['document_title']}")
        print(f"  Category  : {top_res['category']} > {top_res['sub_category']}")
        print(f"  Source    : {top_res['source_document']} (Page {top_res['page_number']})")
        print(f"  Score     : {top_res['relevance_score']}")
        print(f"  Preview   : {top_res['chunk_text'][:120].strip()}...")

    # 3. Dynamic Number of Results (Dynamic API input parameter verification)
    for requested_k in [1, 3]:
        print_banner(f"3. Testing Dynamic number_of_results = {requested_k}")
        res_k = client.post("/api/search", json={
            "query": query_text,
            "number_of_results": requested_k
        })
        assert res_k.status_code == 200
        data_k = res_k.json()
        print(f"Requested K : {data_k['requested_results']}")
        print(f"Returned    : {data_k['total_results']} result(s)")
        assert data_k['requested_results'] == requested_k
        assert len(data_k['results']) <= requested_k

    # 4. Category Filtering
    print_banner("4. Testing Category Filtering: category='HR'")
    res_hr = client.post("/api/search", json={
        "query": "leave entitlements and vacation rules",
        "category": "HR",
        "number_of_results": 2
    })
    assert res_hr.status_code == 200
    data_hr = res_hr.json()
    print(f"HR Filter Results: {data_hr['total_results']}")
    for r in data_hr["results"]:
        print(f"  [{r['relevance_score']}] Category: {r['category']} | Title: {r['document_title']}")
        assert r['category'] == "HR", f"Expected category 'HR', got '{r['category']}'"

    # 5. Irrelevant Query / Below Threshold Scenario
    print_banner("5. Testing Irrelevant Query / Relevance Threshold Rejection")
    irrelevant_query = "quantum physics entanglement superstring dimensions"
    res_irr = client.post("/api/search", json={
        "query": irrelevant_query,
        "number_of_results": 5
    })
    assert res_irr.status_code == 200
    data_irr = res_irr.json()
    print(f"Query          : '{irrelevant_query}'")
    print(f"Threshold Met  : {data_irr['threshold_met']}")
    print(f"Total Results  : {data_irr['total_results']}")
    print(f"Message        : {data_irr['message']}")
    assert data_irr['threshold_met'] is False
    assert len(data_irr['results']) == 0

    # 6. Input Validation Handling
    print_banner("6. Testing API Input Validation (422 Unprocessable Content)")
    bad_requests = [
        ({}, "Missing query"),
        ({"query": ""}, "Empty string query"),
        ({"query": "   "}, "Whitespace query"),
        ({"query": "valid query", "number_of_results": 0}, "Below MIN_TOP_K"),
        ({"query": "valid query", "number_of_results": 100}, "Above MAX_TOP_K")
    ]
    for bad_payload, desc in bad_requests:
        res_bad = client.post("/api/search", json=bad_payload)
        assert res_bad.status_code == 422, f"Expected 422 for '{desc}', got {res_bad.status_code}"
        print(f"  [OK] Successfully rejected invalid request ({desc}) -> HTTP 422")

    print_banner("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
