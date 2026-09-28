from pathlib import Path
from typing import Dict, List, Optional, Any
from wayfinder_ingestion.chunker import Chunk
from wayfinder_ingestion.embedding_service import OllamaEmbeddingService
from wayfinder_ingestion.logging_config import setup_logger

# Use LangChain provided Chroma vector store and Document abstraction
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma  # type: ignore
    except ImportError:
        from langchain.vectorstores import Chroma  # type: ignore

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document  # type: ignore
    except ImportError:
        from dataclasses import dataclass, field

        @dataclass
        class Document:  # type: ignore
            page_content: str
            metadata: dict = field(default_factory=dict)

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    try:
        from langchain.embeddings.base import Embeddings  # type: ignore
    except ImportError:
        class Embeddings:  # type: ignore
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                raise NotImplementedError

            def embed_query(self, text: str) -> List[float]:
                raise NotImplementedError

logger = setup_logger(__name__)


class LangChainOllamaEmbeddings(Embeddings):
    """
    LangChain Embeddings adapter wrapping OllamaEmbeddingService.
    Ensures embeddings are strictly generated using settings (.env EMBEDDING_MODEL).
    """

    def __init__(self, service: Optional[OllamaEmbeddingService] = None):
        self.service = service or OllamaEmbeddingService()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.service.generate_embeddings_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.service.generate_embedding(text)


class VectorStoreManager:
    """
    Manages persistent storage, upserts, deletions, and retrieval queries
    using LangChain's Chroma vector store for document chunks.
    Direct chromadb operations are minimized and only utilized where bulk
    pre-computed vector operations provide maximum execution efficiency.
    """

    COLLECTION_NAME = "wayfinder_documents"

    def __init__(
        self,
        persist_dir: str,
        embedding_function: Optional[Embeddings] = None
    ):
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_function = embedding_function or LangChainOllamaEmbeddings()

        logger.info(
            f"Initializing LangChain Chroma vector store at: {self.persist_dir} "
            f"(collection: '{self.COLLECTION_NAME}')"
        )

        # Initialize LangChain Chroma vector store
        self.vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            persist_directory=str(self.persist_dir),
            collection_metadata={"description": "Wayfinder enterprise PDF document chunks"}
        )

        # Reference to underlying collection and client for efficient direct operations & inspection
        self.collection = self.vectorstore._collection
        self.client = getattr(self.vectorstore, "_client", None)

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Upserts document chunks into LangChain Chroma vector store.
        If precomputed embeddings are provided, efficiently batches them into
        the underlying Chroma collection; otherwise uses LangChain's embedding pipeline.
        Returns list of upserted chunk IDs.
        """
        if not chunks:
            return []

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        if embeddings is not None:
            if len(chunks) != len(embeddings):
                raise ValueError(
                    f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
                )
            # Efficient direct bulk upsert using precomputed embeddings
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        else:
            # Use LangChain native Document addition with embedding generation
            lc_docs = [
                Document(page_content=chunk.text, metadata=chunk.metadata)
                for chunk in chunks
            ]
            self.vectorstore.add_documents(documents=lc_docs, ids=ids)

        logger.info(
            f"Successfully upserted {len(chunks)} chunks into LangChain Chroma collection '{self.COLLECTION_NAME}'."
        )
        return ids

    def delete_by_file_path(self, relative_file_path: str) -> None:
        """
        Deletes all stored chunks associated with a specific file path using LangChain's Chroma interface.
        Used when a PDF is updated or deleted from source directory.
        """
        try:
            # Query existing chunk IDs for this file_path via LangChain get()
            existing = self.vectorstore.get(
                where={"file_path": relative_file_path}
            )
            if existing and existing.get("ids"):
                ids_to_delete = existing["ids"]
                self.vectorstore.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} existing chunks for file: {relative_file_path}")
            else:
                logger.debug(f"No existing chunks found in Chroma for file: {relative_file_path}")
        except Exception as e:
            logger.error(f"Failed to delete chunks for file '{relative_file_path}' in Chroma: {e}")
            raise

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Queries Chroma for similar chunks given a query embedding vector.
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results
        }
        if where_filter:
            kwargs["where"] = where_filter

        return self.collection.query(**kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        LangChain native similarity search by text query.
        """
        return self.vectorstore.similarity_search(query=query, k=k, filter=filter)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        LangChain native similarity search by vector embedding.
        """
        return self.vectorstore.similarity_search_by_vector(embedding=embedding, k=k, filter=filter)

    def as_retriever(self, **kwargs) -> Any:
        """
        Returns a LangChain VectorStoreRetriever for RAG agent integration.
        """
        return self.vectorstore.as_retriever(**kwargs)

    def get_count(self) -> int:
        """Returns total number of chunks stored in the collection."""
        return self.collection.count()
