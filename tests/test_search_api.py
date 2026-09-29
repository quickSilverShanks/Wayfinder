from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from wayfinder_search.api import app, get_search_service
from wayfinder_search.models import SearchResponse, SearchResultItem


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["search_endpoint"] == "/api/search"
    assert data["documentation"] == "/docs"


def test_health_check_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "embedding_model" in data
    assert "reranker_model" in data
    assert "indexed_chunks_in_chroma" in data


def test_search_api_validation_errors(client):
    # Case 1: Missing query
    res_missing = client.post("/api/search", json={})
    assert res_missing.status_code == 422
    assert "Request validation failed" in res_missing.json()["detail"]

    # Case 2: Empty query
    res_empty = client.post("/api/search", json={"query": ""})
    assert res_empty.status_code == 422

    # Case 3: Whitespace only query
    res_whitespace = client.post("/api/search", json={"query": "   "})
    assert res_whitespace.status_code == 422

    # Case 4: Invalid number_of_results (e.g. 0 or negative)
    res_zero = client.post("/api/search", json={"query": "valid query", "number_of_results": 0})
    assert res_zero.status_code == 422

    res_too_large = client.post("/api/search", json={"query": "valid query", "number_of_results": 100})
    assert res_too_large.status_code == 422


def test_search_api_successful_post(client):
    mock_item = SearchResultItem(
        document_title="Customer Rights Policy",
        category="Payments",
        sub_category="Disputes",
        source_document="samples/customer-rights-policy.pdf",
        relevance_score=0.9123,
        chunk_text="Disputed transaction grievance redressal procedures.",
        page_number=4,
        document_id="doc_hash_123",
        chunk_id="chunk_id_456"
    )
    mock_response = SearchResponse(
        query="disputed transaction",
        requested_results=3,
        candidates_retrieved=15,
        search_duration_seconds=0.1234,
        threshold_met=True,
        relevance_threshold=0.20,
        total_results=1,
        results=[mock_item],
        message="Found 1 relevant result."
    )

    mock_service = MagicMock()
    mock_service.search.return_value = mock_response

    with patch("wayfinder_search.api.get_search_service", return_value=mock_service):
        payload = {
            "query": "How should I handle a customer disputing a transaction?",
            "number_of_results": 3,
            "category": "Payments",
            "sub_category": "Disputes"
        }
        response = client.post("/api/search", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "disputed transaction"
        assert data["requested_results"] == 3
        assert data["threshold_met"] is True
        assert data["total_results"] == 1
        assert len(data["results"]) == 1

        result_0 = data["results"][0]
        assert result_0["document_title"] == "Customer Rights Policy"
        assert result_0["category"] == "Payments"
        assert result_0["sub_category"] == "Disputes"
        assert result_0["source_document"] == "samples/customer-rights-policy.pdf"
        assert result_0["relevance_score"] == 0.9123
        assert result_0["chunk_text"] == "Disputed transaction grievance redressal procedures."
        assert result_0["page_number"] == 4
        assert result_0["document_id"] == "doc_hash_123"
        assert result_0["chunk_id"] == "chunk_id_456"


def test_search_api_no_result_scenario(client):
    mock_response = SearchResponse(
        query="astrophysics black hole",
        requested_results=5,
        candidates_retrieved=10,
        search_duration_seconds=0.05,
        threshold_met=False,
        relevance_threshold=0.20,
        total_results=0,
        results=[],
        message="No retrieved document chunk met the minimum relevance threshold."
    )

    mock_service = MagicMock()
    mock_service.search.return_value = mock_response

    with patch("wayfinder_search.api.get_search_service", return_value=mock_service):
        payload = {"query": "astrophysics black hole"}
        response = client.post("/api/search", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["threshold_met"] is False
        assert data["total_results"] == 0
        assert data["results"] == []
        assert "No retrieved document chunk met" in data["message"]
