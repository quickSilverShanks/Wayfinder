from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Ensure .env is loaded before prefect client initializes
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file)
else:
    load_dotenv()

# Automatically configure Prefect client to connect to local server if set in .env
if not os.environ.get("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

from prefect import flow, task

from wayfinder_ingestion.config import Settings, get_settings
from wayfinder_ingestion.logging_config import setup_logger
from wayfinder_ingestion.change_detector import ChangeDetector, ChangeSet, DocumentMetadataInfo
from wayfinder_ingestion.pdf_parser import PDFParser
from wayfinder_ingestion.pii_masker import PIIMasker
from wayfinder_ingestion.chunker import DocumentChunker
from wayfinder_ingestion.embedding_service import OllamaEmbeddingService
from wayfinder_ingestion.vector_store import VectorStoreManager
from wayfinder_ingestion.ledger import IngestionLedger

logger = setup_logger(__name__)


@task(name="Detect PDF Document Changes")
def detect_changes_task(source_dir: str, chroma_dir: str, csv_path: Optional[str] = None) -> ChangeSet:
    """
    Prefect Task: Scans source directory and computes ChangeSet relative to saved manifest and CSV ledger.
    """
    detector = ChangeDetector(source_dir=source_dir, manifest_dir=chroma_dir, csv_path=csv_path)
    return detector.detect_changes()


@task(name="Process and Index Single PDF Document")
def process_document_task(
    doc_info: DocumentMetadataInfo,
    settings: Settings,
    is_modified: bool = False
) -> Dict[str, Any]:
    """
    Prefect Task: Processes a single PDF document through:
    1. Docling: Converts PDF to structured Markdown (MD)
    2. PII Masking: Redacts sensitive PII patterns in Markdown text
    3. LangChain: Recursive character text splitting with overlap on Markdown
    4. Ollama: Vectorizes document chunks using Ollama-hosted embedding model
    5. ChromaDB: Stores vectors and preserved structural metadata
    6. Ledger: Records ingestion in CSV audit trail and SHA manifest
    """
    logger.info(f"Starting processing for document: {doc_info.relative_path} (Category: '{doc_info.category}', SubCategory: '{doc_info.sub_category}')")

    result_status = {
        "relative_path": doc_info.relative_path,
        "success": False,
        "chunk_count": 0,
        "error": None
    }

    try:
        # 1. Initialize components
        parser = PDFParser()
        masker = PIIMasker(enabled=settings.PII_MASKING_ENABLED)
        chunker = DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        embedding_service = OllamaEmbeddingService(
            base_url=settings.OLLAMA_BASE_URL,
            model_name=settings.EMBEDDING_MODEL
        )
        vector_store = VectorStoreManager(persist_dir=settings.CHROMA_PERSIST_DIR)
        detector = ChangeDetector(source_dir=settings.SOURCE_DOCUMENT_DIR, manifest_dir=settings.CHROMA_PERSIST_DIR)
        ledger = IngestionLedger(csv_path=settings.INGESTION_LOG_CSV)

        # If document was modified, delete existing vectors first
        if is_modified:
            logger.info(f"Document is modified. Removing old vectors for: {doc_info.relative_path}")
            vector_store.delete_by_file_path(doc_info.relative_path)

        # 2. Parse PDF
        parsed_doc = parser.parse(doc_info.absolute_path)

        # 3. PII Masking on pages
        for page in parsed_doc.pages:
            masked_text, redaction_stats = masker.mask(page.text)
            page.text = masked_text
            if redaction_stats:
                logger.info(f"Masked PII in {doc_info.relative_path} (Page {page.page_number}): {redaction_stats}")

        # 4. Chunk document
        chunks = chunker.chunk_document(parsed_doc, doc_info)
        if not chunks:
            logger.warning(f"No chunks generated for document: {doc_info.relative_path}")
            result_status["success"] = True
            return result_status

        # 5. Generate embeddings via Ollama
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.generate_embeddings_batch(chunk_texts)

        # 6. Upsert into ChromaDB
        vector_store.upsert_chunks(chunks, embeddings)

        # 7. Update JSON manifest
        chunk_ids = [c.chunk_id for c in chunks]
        detector.record_ingested(doc_info, chunk_ids)

        # 8. Record in CSV ledger
        vectorized_time = datetime.now(timezone.utc).isoformat()
        ledger.record_success(
            doc_info=doc_info,
            chunk_count=len(chunks),
            vectorized_at=vectorized_time
        )

        result_status["success"] = True
        result_status["chunk_count"] = len(chunks)
        logger.info(f"Successfully processed, indexed, and logged {len(chunks)} chunks for: {doc_info.relative_path}")

    except Exception as e:
        logger.error(f"Failed to process document {doc_info.relative_path}: {e}", exc_info=True)
        result_status["error"] = str(e)

    return result_status


@task(name="Delete Removed PDF Document Vectors")
def delete_document_task(rel_path: str, settings: Settings) -> bool:
    """
    Prefect Task: Removes vectors, manifest record, and updates CSV ledger for a deleted PDF document.
    """
    try:
        logger.info(f"Removing deleted document from index: {rel_path}")
        vector_store = VectorStoreManager(persist_dir=settings.CHROMA_PERSIST_DIR)
        vector_store.delete_by_file_path(rel_path)

        detector = ChangeDetector(source_dir=settings.SOURCE_DOCUMENT_DIR, manifest_dir=settings.CHROMA_PERSIST_DIR)
        detector.record_deleted(rel_path)

        ledger = IngestionLedger(csv_path=settings.INGESTION_LOG_CSV)
        ledger.record_deletion(rel_path)
        return True
    except Exception as e:
        logger.error(f"Failed to delete document vectors for {rel_path}: {e}")
        return False


@flow(name="Wayfinder Document Ingestion Flow")
def ingest_documents_flow(
    source_dir_override: Optional[str] = None,
    chroma_dir_override: Optional[str] = None,
    csv_path_override: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main Prefect Flow for orchestrating Wayfinder PDF document ingestion.
    Fully idempotent change-based ingestion pipeline.
    """
    settings = get_settings()

    if source_dir_override:
        settings.SOURCE_DOCUMENT_DIR = source_dir_override
    if chroma_dir_override:
        settings.CHROMA_PERSIST_DIR = chroma_dir_override
    if csv_path_override:
        settings.INGESTION_LOG_CSV = csv_path_override

    logger.info("=== Starting Wayfinder Document Ingestion Pipeline ===")
    logger.info(f"Source Directory: {settings.SOURCE_DOCUMENT_DIR}")
    logger.info(f"ChromaDB Directory: {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"CSV Ledger Path: {settings.INGESTION_LOG_CSV}")
    logger.info(f"Ollama Base URL: {settings.OLLAMA_BASE_URL} (Model: {settings.EMBEDDING_MODEL})")
    logger.info(f"PII Masking Enabled: {settings.PII_MASKING_ENABLED}")

    # Step 1: Detect changes
    change_set = detect_changes_task(
        source_dir=settings.SOURCE_DOCUMENT_DIR,
        chroma_dir=settings.CHROMA_PERSIST_DIR,
        csv_path=settings.INGESTION_LOG_CSV
    )

    if not change_set.has_changes:
        logger.info("No document changes detected (New=0, Modified=0, Deleted=0). Ingestion complete.")
        return {
            "processed": 0,
            "failed": 0,
            "deleted": 0,
            "unchanged": len(change_set.unchanged_files),
            "embedding_model": settings.EMBEDDING_MODEL,
            "csv_ledger": settings.INGESTION_LOG_CSV
        }

    successful_count = 0
    failed_count = 0

    # Step 2: Handle deleted files
    deleted_count = 0
    for rel_path in change_set.deleted_files:
        deleted_success = delete_document_task(rel_path, settings)
        if deleted_success:
            deleted_count += 1

    # Step 3: Process new files
    for doc_info in change_set.new_files:
        res = process_document_task(doc_info, settings, is_modified=False)
        if res["success"]:
            successful_count += 1
        else:
            failed_count += 1

    # Step 4: Process modified files
    for doc_info in change_set.modified_files:
        res = process_document_task(doc_info, settings, is_modified=True)
        if res["success"]:
            successful_count += 1
        else:
            failed_count += 1

    summary = {
        "processed": successful_count,
        "failed": failed_count,
        "deleted": deleted_count,
        "unchanged": len(change_set.unchanged_files),
        "embedding_model": settings.EMBEDDING_MODEL,
        "csv_ledger": settings.INGESTION_LOG_CSV
    }

    logger.info(f"=== Wayfinder Ingestion Completed Summary: {summary} ===")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wayfinder PDF Document Ingestion Pipeline")
    parser.add_argument("--watch", action="store_true", help="Run folder watcher ('ONLY' while Prefect server is running)")
    parser.add_argument("--serve", action="store_true", help="Serve flow as a deployment on Prefect Server (registers permanently on UI)")
    parser.add_argument("--source-dir", type=str, default=None, help="Source documents directory")
    parser.add_argument("--chroma-dir", type=str, default=None, help="ChromaDB persist directory")
    parser.add_argument("--csv-path", type=str, default=None, help="CSV ledger path")
    parser.add_argument("--interval", type=int, default=None, help="Watch interval in seconds")
    parser.add_argument("--prefect-url", type=str, default=None, help="Prefect server API URL")
    cli_args = parser.parse_args()

    if cli_args.watch:
        from wayfinder_ingestion.watcher import DocumentFolderWatcher
        folder_watcher = DocumentFolderWatcher(
            source_dir=cli_args.source_dir,
            chroma_dir=cli_args.chroma_dir,
            csv_path=cli_args.csv_path,
            interval_seconds=cli_args.interval,
            prefect_api_url=cli_args.prefect_url
        )
        folder_watcher.watch()
    elif cli_args.serve:
        logger.info("Serving Wayfinder Document Ingestion Flow as a deployment on Prefect Server...")
        ingest_documents_flow.serve(
            name="wayfinder-pdf-ingestor",
            description="Enterprise PDF document ingestion pipeline for frontline knowledge base",
            tags=["ingestion", "pdf", "rag"]
        )
    else:
        ingest_documents_flow(
            source_dir_override=cli_args.source_dir,
            chroma_dir_override=cli_args.chroma_dir,
            csv_path_override=cli_args.csv_path
        )
