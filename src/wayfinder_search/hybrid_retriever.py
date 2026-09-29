"""
Hybrid Retriever for Wayfinder Intelligent Document Retrieval.
Combines ChromaDB dense vector search with in-memory BM25 lexical keyword search.
"""

from typing import Dict, List, Any, Optional
from wayfinder_ingestion.config import get_settings
from wayfinder_ingestion.embedding_service import OllamaEmbeddingService
from wayfinder_ingestion.vector_store import VectorStoreManager
from wayfinder_ingestion.logging_config import setup_logger
from wayfinder_search.bm25 import BM25Index

logger = setup_logger(__name__)


def build_chroma_where_filter(
    category: Optional[str] = None,
    sub_category: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Constructs a ChromaDB-compliant where filter.
    Handles single conditions directly and multiple conditions via '$and'.
    """
    conditions: List[Dict[str, Any]] = []
    if category and category.strip():
        conditions.append({"category": category.strip()})
    if sub_category and sub_category.strip():
        conditions.append({"sub_category": sub_category.strip()})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class HybridRetriever:
    """
    Modular Hybrid Retriever combining:
    1. Dense semantic vector retrieval via ChromaDB and Ollama embedding model.
    2. BM25Okapi lexical retrieval over indexed document chunks.
    3. Reciprocal Rank Fusion (RRF) to merge candidate results before reranking.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        embedding_service: Optional[OllamaEmbeddingService] = None,
        bm25_index: Optional[BM25Index] = None
    ):
        settings = get_settings()
        self.settings = settings
        self.vector_store = vector_store or VectorStoreManager(persist_dir=settings.CHROMA_PERSIST_DIR)
        self.embedding_service = embedding_service or OllamaEmbeddingService()
        self.bm25_index = bm25_index or BM25Index()
        self._bm25_initialized = False

    def sync_bm25_index(self, force_refresh: bool = False) -> None:
        """
        Synchronizes the in-memory BM25 index with current chunks in ChromaDB.
        Avoids redundant syncs if already initialized and not forced.
        """
        if self._bm25_initialized and not force_refresh:
            return

        try:
            total_chunks = self.vector_store.get_count()
            if total_chunks == 0:
                logger.warning("ChromaDB is empty; BM25 index initialized with 0 documents.")
                self.bm25_index.index_documents([])
                self._bm25_initialized = True
                return

            # Retrieve all documents, IDs, and metadatas from ChromaDB
            data = self.vector_store.collection.get(
                include=["documents", "metadatas"]
            )
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])

            chunk_records: List[Dict[str, Any]] = []
            for chunk_id, doc_text, meta in zip(ids, documents, metadatas):
                chunk_records.append({
                    "chunk_id": chunk_id,
                    "text": doc_text or "",
                    "metadata": meta or {}
                })

            self.bm25_index.index_documents(chunk_records)
            self._bm25_initialized = True
            logger.info(f"Synchronized BM25 index with {len(chunk_records)} document chunks from ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to synchronize BM25 index with ChromaDB: {e}")
            raise

    def retrieve_dense(
        self,
        query: str,
        top_k: int = 15,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes dense semantic retrieval by embedding query via Ollama and querying ChromaDB.
        """
        try:
            query_embedding = self.embedding_service.generate_embedding(query)
            query_response = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=top_k,
                where_filter=where_filter
            )

            ids = query_response.get("ids", [[]])[0]
            documents = query_response.get("documents", [[]])[0]
            metadatas = query_response.get("metadatas", [[]])[0]
            distances = query_response.get("distances", [[]])[0] if "distances" in query_response else []

            results: List[Dict[str, Any]] = []
            for rank, (chunk_id, doc_text, meta) in enumerate(zip(ids, documents, metadatas), start=1):
                dist = distances[rank - 1] if rank - 1 < len(distances) else None
                results.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": meta or {},
                    "dense_rank": rank,
                    "dense_distance": dist
                })

            logger.debug(f"Dense vector search returned {len(results)} chunks for query: '{query[:40]}...'")
            return results
        except Exception as e:
            logger.error(f"Error during dense vector search: {e}")
            raise

    def retrieve_lexical(
        self,
        query: str,
        top_k: int = 15,
        category: Optional[str] = None,
        sub_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes lexical retrieval using the synchronized BM25 index.
        """
        self.sync_bm25_index()
        raw_results = self.bm25_index.search(
            query=query,
            top_k=top_k,
            category=category,
            sub_category=sub_category
        )

        results: List[Dict[str, Any]] = []
        for rank, (doc_record, score) in enumerate(raw_results, start=1):
            results.append({
                "chunk_id": doc_record["chunk_id"],
                "text": doc_record["text"],
                "metadata": doc_record["metadata"],
                "lexical_rank": rank,
                "lexical_score": score
            })

        logger.debug(f"BM25 lexical search returned {len(results)} chunks for query: '{query[:40]}...'")
        return results

    def retrieve_hybrid(
        self,
        query: str,
        dense_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
        rerank_k: Optional[int] = None,
        category: Optional[str] = None,
        sub_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates hybrid retrieval:
        1. Queries Dense retriever (ChromaDB + Ollama)
        2. Queries Lexical retriever (BM25)
        3. Merges and deduplicates candidates using Reciprocal Rank Fusion (RRF)
        4. Returns top `rerank_k` candidates ready for cross-encoder reranking.
        """
        settings = self.settings
        dense_top_k = dense_k or settings.DENSE_TOP_K
        bm25_top_k = bm25_k or settings.BM25_TOP_K
        rerank_candidate_limit = rerank_k or settings.RERANK_TOP_K

        where_filter = build_chroma_where_filter(category=category, sub_category=sub_category)

        # 1. Dense search
        dense_candidates = self.retrieve_dense(
            query=query,
            top_k=dense_top_k,
            where_filter=where_filter
        )

        # 2. Lexical search
        lexical_candidates = self.retrieve_lexical(
            query=query,
            top_k=bm25_top_k,
            category=category,
            sub_category=sub_category
        )

        # 3. Reciprocal Rank Fusion (RRF) merge
        # RRF score = sum(1.0 / (k_rrf + rank))
        k_rrf = 60
        merged_candidates: Dict[str, Dict[str, Any]] = {}

        # Process dense candidates
        for item in dense_candidates:
            cid = item["chunk_id"]
            rrf_score = 1.0 / (k_rrf + item["dense_rank"])
            merged_candidates[cid] = {
                "chunk_id": cid,
                "text": item["text"],
                "metadata": item["metadata"],
                "rrf_score": rrf_score,
                "dense_rank": item["dense_rank"],
                "lexical_rank": None
            }

        # Process lexical candidates
        for item in lexical_candidates:
            cid = item["chunk_id"]
            rrf_contrib = 1.0 / (k_rrf + item["lexical_rank"])
            if cid in merged_candidates:
                merged_candidates[cid]["rrf_score"] += rrf_contrib
                merged_candidates[cid]["lexical_rank"] = item["lexical_rank"]
            else:
                merged_candidates[cid] = {
                    "chunk_id": cid,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf_score": rrf_contrib,
                    "dense_rank": None,
                    "lexical_rank": item["lexical_rank"]
                }

        # Sort candidates by combined RRF score descending
        sorted_candidates = sorted(
            merged_candidates.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        top_candidates = sorted_candidates[:rerank_candidate_limit]
        logger.info(
            f"Hybrid retrieval merged {len(dense_candidates)} dense + {len(lexical_candidates)} lexical "
            f"-> {len(merged_candidates)} unique candidates -> top {len(top_candidates)} selected for reranker."
        )

        return top_candidates
