import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.vector_store import (
    VectorStoreManager,
    LangChainOllamaEmbeddings,
    Chroma,
    Document
)
from wayfinder_ingestion.chunker import Chunk


def test_chroma_vector_store_operations(tmp_path):
    chroma_dir = tmp_path / "chroma_db"
    vs = VectorStoreManager(persist_dir=str(chroma_dir))

    # Verify that VectorStoreManager utilizes LangChain's Chroma
    assert vs.vectorstore is not None
    assert isinstance(vs.vectorstore, Chroma)

    chunk1 = Chunk(
        chunk_id="hash1_p1_c1",
        text="Frontline annual leave entitlement policy.",
        metadata={"file_path": "HR/leave.pdf", "category": "HR", "page_number": 1}
    )
    chunk2 = Chunk(
        chunk_id="hash1_p1_c2",
        text="Sick leave and medical emergency protocol.",
        metadata={"file_path": "HR/leave.pdf", "category": "HR", "page_number": 1}
    )

    # Fake 4-dimensional embeddings for test
    emb1 = [0.1, 0.2, 0.3, 0.4]
    emb2 = [0.2, 0.3, 0.4, 0.5]

    # Upsert using pre-computed embeddings
    inserted_ids = vs.upsert_chunks([chunk1, chunk2], [emb1, emb2])
    assert len(inserted_ids) == 2
    assert vs.get_count() == 2

    # Query with raw embedding vector
    query_res = vs.query(query_embedding=[0.15, 0.25, 0.35, 0.45], n_results=1)
    assert len(query_res["ids"][0]) == 1

    # LangChain native similarity search by vector
    lc_docs = vs.similarity_search_by_vector([0.1, 0.2, 0.3, 0.4], k=1)
    assert len(lc_docs) == 1
    assert "leave" in lc_docs[0].page_content.lower()

    # LangChain retriever interface
    retriever = vs.as_retriever(search_kwargs={"k": 1})
    assert retriever is not None

    # Delete by file path via LangChain interface
    vs.delete_by_file_path("HR/leave.pdf")
    assert vs.get_count() == 0


def test_empty_chunks_upsert(tmp_path):
    chroma_dir = tmp_path / "chroma_db_empty"
    vs = VectorStoreManager(persist_dir=str(chroma_dir))
    assert vs.upsert_chunks([]) == []


def test_langchain_ollama_embeddings_adapter():
    mock_service = MagicMock()
    mock_service.generate_embedding.return_value = [0.1, 0.2]
    mock_service.generate_embeddings_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]

    adapter = LangChainOllamaEmbeddings(service=mock_service)
    assert adapter.embed_query("test query") == [0.1, 0.2]
    assert adapter.embed_documents(["doc1", "doc2"]) == [[0.1, 0.2], [0.3, 0.4]]
