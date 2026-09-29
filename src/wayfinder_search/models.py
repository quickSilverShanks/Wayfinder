"""
Pydantic schemas and data models for Wayfinder Search API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """
    Search request payload for POST /api/search.
    """
    query: str = Field(
        ...,
        description="Natural-language search query submitted by frontline staff",
        min_length=1,
        examples=["How should I handle a customer disputing a transaction?"]
    )
    number_of_results: Optional[int] = Field(
        default=None,
        description="Number of top results to return. If omitted, uses DEFAULT_TOP_K from .env.",
        examples=[5]
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional primary document category filter (e.g. HR, IT, Payments, samples)",
        examples=["Payments"]
    )
    sub_category: Optional[str] = Field(
        default=None,
        description="Optional sub-category filter (e.g. Disputes, Policies, Security)",
        examples=["Disputes"]
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Query string cannot be empty or purely whitespace.")
        return trimmed

    @field_validator("number_of_results")
    @classmethod
    def validate_number_of_results(cls, value: Optional[int]) -> Optional[int]:
        if value is not None:
            from wayfinder_ingestion.config import get_settings
            settings = get_settings()
            if value < settings.MIN_TOP_K:
                raise ValueError(f"number_of_results must be at least {settings.MIN_TOP_K}.")
            if value > settings.MAX_TOP_K:
                raise ValueError(f"number_of_results cannot exceed {settings.MAX_TOP_K}.")
        return value

    @field_validator("category", "sub_category")
    @classmethod
    def clean_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed if trimmed else None


class SearchResultItem(BaseModel):
    """
    Individual document chunk search result with structural metadata and relevance score.
    """
    document_title: str = Field(..., description="Document title extracted from PDF/Markdown")
    category: str = Field(..., description="Primary category derived from folder structure")
    sub_category: str = Field(..., description="Sub-category derived from folder structure")
    source_document: str = Field(..., description="Relative file path to source PDF")
    relevance_score: float = Field(..., description="Reranker relevance score (0.0 to 1.0)")
    chunk_text: str = Field(..., description="Full text content of the retrieved chunk")
    page_number: int = Field(..., description="Page number where this chunk originated")
    document_id: str = Field(..., description="Document identifier / content SHA-256 hash")
    chunk_id: str = Field(..., description="Unique deterministic chunk identifier")


class SearchResponse(BaseModel):
    """
    Complete response payload returned by POST /api/search.
    """
    query: str = Field(..., description="Original search query executed")
    requested_results: int = Field(..., description="Validated requested number of top results")
    candidates_retrieved: int = Field(..., description="Number of candidate chunks retrieved before reranking")
    search_duration_seconds: float = Field(..., description="Total search and rerank latency in seconds")
    threshold_met: bool = Field(..., description="Whether any retrieved chunks met the relevance threshold")
    relevance_threshold: float = Field(..., description="Active relevance threshold applied")
    total_results: int = Field(..., description="Count of qualifying results returned in results list")
    results: List[SearchResultItem] = Field(
        default_factory=list,
        description="Top-K results that met the relevance threshold, sorted descending by score"
    )
    below_threshold_results: Optional[List[SearchResultItem]] = Field(
        default=None,
        description="Candidates below relevance threshold (kept for debugging/diagnostics)"
    )
    message: str = Field(..., description="Human-readable status summary message")
