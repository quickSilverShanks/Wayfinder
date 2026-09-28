from typing import List, Optional
import requests
from wayfinder_ingestion.config import get_settings
from wayfinder_ingestion.logging_config import setup_logger

logger = setup_logger(__name__)


class OllamaEmbeddingService:
    """
    Service for generating vector embeddings using an Ollama-hosted embedding model.
    Picks the model_name and base_url directly from settings (.env) by default.
    Communicates directly with the Ollama REST API.
    """

    def __init__(self, base_url: Optional[str] = None, model_name: Optional[str] = None):
        """
        Service for generating vector embeddings using an Ollama-hosted embedding model.
        The embedding model is ALWAYS picked from settings in the .env file.
        Communicates directly with the Ollama REST API.
        """
        settings = get_settings()
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        # The embedding model is ALWAYS picked from settings in .env file
        self.model_name = (settings.EMBEDDING_MODEL or "nomic-embed-text").strip()
        if model_name and model_name.strip() != self.model_name:
            logger.info(
                f"Notice: explicit model_name='{model_name}' ignored; "
                f"embedding model is strictly loaded from settings (.env): '{self.model_name}'"
            )
        logger.debug(f"OllamaEmbeddingService active with model '{self.model_name}' from settings (.env) at '{self.base_url}'")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates embedding vector for a single text block.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace text.")

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "embedding" in data:
                return data["embedding"]
            else:
                raise KeyError(f"Ollama API response missing 'embedding' key: {data.keys()}")

        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Ensure Ollama container/service is running and accessible."
            )
            raise RuntimeError(f"Ollama connection error: {e}") from e
        except Exception as e:
            logger.error(f"Error calling Ollama embeddings endpoint for model '{self.model_name}': {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a list of text blocks.
        """
        embeddings: List[List[float]] = []
        for idx, text in enumerate(texts):
            try:
                emb = self.generate_embedding(text)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Batch embedding failed at index {idx}: {e}")
                raise
        return embeddings
