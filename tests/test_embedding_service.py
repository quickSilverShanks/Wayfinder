import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.embedding_service import OllamaEmbeddingService
from wayfinder_ingestion.config import Settings


def test_embedding_service_picks_model_from_settings(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3-custom")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-custom:11434")

    service = OllamaEmbeddingService()

    assert service.model_name == "bge-m3-custom"
    assert service.base_url == "http://ollama-custom:11434"


def test_embedding_service_payload_uses_configured_model(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "mxbai-embed-large")

    service = OllamaEmbeddingService()
    assert service.model_name == "mxbai-embed-large"

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_resp

        emb = service.generate_embedding("Sample text for vectorization")

        assert emb == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "mxbai-embed-large"
        assert call_kwargs["json"]["prompt"] == "Sample text for vectorization"


def test_embedding_service_always_picks_from_settings_even_if_passed(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-large-en-v1.5")

    # Even if another model name is supplied to the constructor, settings takes precedence
    service = OllamaEmbeddingService(model_name="unintended-model-name")
    assert service.model_name == "bge-large-en-v1.5"


def test_embedding_service_empty_text_raises():
    service = OllamaEmbeddingService()
    with pytest.raises(ValueError):
        service.generate_embedding("")

    with pytest.raises(ValueError):
        service.generate_embedding("   ")

