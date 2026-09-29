"""
FastAPI application for Wayfinder Intelligent Document Retrieval.
Exposes POST /api/search and health/diagnostic endpoints.
"""

from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from wayfinder_ingestion.config import get_settings
from wayfinder_ingestion.logging_config import setup_logger
from wayfinder_search.models import SearchRequest, SearchResponse
from wayfinder_search.service import SearchService

logger = setup_logger(__name__)

settings = get_settings()

app = FastAPI(
    title="Wayfinder Search & Retrieval Backend API",
    description="Intelligent hybrid retrieval with BM25, dense vector search, and BGE cross-encoder reranking.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for future frontend web application integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service singleton
_search_service: SearchService = None


def get_search_service() -> SearchService:
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Standardized validation error handler providing clear feedback for invalid parameters.
    """
    errors = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append({
            "field": field_path,
            "message": err.get("msg"),
            "type": err.get("type")
        })
    logger.warning(f"Validation error on {request.url.path}: {errors}")
    return JSONResponse(
        status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY),
        content={
            "detail": "Request validation failed",
            "errors": errors
        }
    )


@app.get("/", tags=["Info"])
async def root_info() -> Dict[str, Any]:
    """
    Basic API root endpoint with service metadata and documentation links.
    """
    return {
        "service": "Wayfinder Intelligent Document Retrieval API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/api/health",
        "search_endpoint": "/api/search"
    }


@app.get("/api/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint returning system status and component readiness.
    """
    try:
        service = get_search_service()
        total_chunks = service.hybrid_retriever.vector_store.get_count()
        return {
            "status": "healthy",
            "service": "Wayfinder Search Backend",
            "indexed_chunks_in_chroma": total_chunks,
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "relevance_threshold": settings.RELEVANCE_THRESHOLD,
            "default_top_k": settings.DEFAULT_TOP_K
        }
    except Exception as e:
        logger.error(f"Health check failure: {e}")
        return {
            "status": "degraded",
            "error": str(e)
        }


@app.post("/api/search", response_model=SearchResponse, status_code=status.HTTP_200_OK, tags=["Search"])
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Search document repository using hybrid retrieval, cross-encoder reranking,
    and relevance threshold filtering.
    """
    try:
        service = get_search_service()
        response = service.search(request)
        return response
    except Exception as e:
        logger.error(f"Error handling search request for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during search execution: {str(e)}"
        )
