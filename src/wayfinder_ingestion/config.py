import os
from pathlib import Path
from typing import Optional
from pydantic import Field
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError: # Fallback if pydantic_settings isn't installed standalone
    from pydantic import BaseSettings, SettingsConfigDict  # type: ignore


ROOT_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """
    Configuration settings for Wayfinder PDF Document Ingestion Pipeline.
    Loads settings from environment variables or local .env file.
    """
    SOURCE_DOCUMENT_DIR: str = Field(
        default="./data/source_documents",
        description="Path to directory containing source PDF documents"
    )
    CHROMA_PERSIST_DIR: str = Field(
        default="./data/chroma_db",
        description="Path to directory where ChromaDB data is persisted"
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama service"
    )
    EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model name (always picked from settings in .env)"
    )
    CHAT_MODEL: str = Field(
        default="llama3.2",
        description="Ollama chat model name"
    )
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Target maximum size of each document chunk in characters"
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap size between consecutive document chunks in characters"
    )
    PII_MASKING_ENABLED: bool = Field(
        default=True,
        description="Whether to perform PII masking before embedding and indexing"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    INGESTION_LOG_CSV: str = Field(
        default="./data/ingested_documents.csv",
        description="Path to CSV ledger tracking ingested documents and vectorization status"
    )
    WATCH_INTERVAL_SECONDS: int = Field(
        default=5,
        description="Polling interval in seconds for directory watch loop"
    )
    PREFECT_API_URL: Optional[str] = Field(
        default=None,
        description="Prefect API URL. Leave empty for local ephemeral execution or set to http://localhost:4200/api when running Prefect server."
    )

    # Search & Retrieval Backend Settings (Part 2)
    DEFAULT_TOP_K: int = Field(
        default=5,
        description="Default number of search results returned when not specified in API request"
    )
    MIN_TOP_K: int = Field(
        default=1,
        description="Minimum allowed number of search results in API request"
    )
    MAX_TOP_K: int = Field(
        default=50,
        description="Maximum allowed number of search results in API request"
    )
    DENSE_TOP_K: int = Field(
        default=15,
        description="Number of candidate chunks retrieved via ChromaDB dense vector search"
    )
    BM25_TOP_K: int = Field(
        default=15,
        description="Number of candidate chunks retrieved via BM25 lexical search"
    )
    RERANK_TOP_K: int = Field(
        default=25,
        description="Maximum number of merged candidates sent to the BGE reranker"
    )
    MAX_BELOW_THRESHOLD_RESULTS: int = Field(
        default=3,
        description="Maximum number of below-threshold results to return in search response"
    )
    DENSE_WEIGHT: float = Field(
        default=1.0,
        description="Weight factor for dense vector retrieval in RRF candidate fusion"
    )
    BM25_WEIGHT: float = Field(
        default=0.5,
        description="Weight factor for BM25 lexical retrieval in RRF candidate fusion"
    )
    RELEVANCE_THRESHOLD: float = Field(
        default=0.20,
        description="Relevance score threshold (0.0 to 1.0) below which results are excluded from primary results"
    )
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Reranker model identifier (configurable via .env, never hard-coded in logic)"
    )
    RERANKER_DEVICE: str = Field(
        default="auto",
        description="Device for reranker execution ('auto', 'cuda', 'cpu')"
    )

    model_config = SettingsConfigDict(
        env_file=str(ROOT_ENV_FILE) if ROOT_ENV_FILE.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def source_dir_path(self) -> Path:
        return Path(self.SOURCE_DOCUMENT_DIR).resolve()

    @property
    def chroma_dir_path(self) -> Path:
        return Path(self.CHROMA_PERSIST_DIR).resolve()

    @property
    def csv_ledger_path(self) -> Path:
        return Path(self.INGESTION_LOG_CSV).resolve()

    @property
    def effective_prefect_api_url(self) -> str:
        """Returns configured PREFECT_API_URL or default local server endpoint."""
        if self.PREFECT_API_URL and self.PREFECT_API_URL.strip():
            return self.PREFECT_API_URL.strip().rstrip("/")
        return "http://localhost:4200/api"


def get_settings() -> Settings:
    """Get initialized settings singleton instance."""
    return Settings()
