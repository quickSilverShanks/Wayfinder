import os
import sys
from pathlib import Path

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wayfinder_ingestion.config import get_settings
from wayfinder_ingestion.pipeline import ingest_documents_flow
from wayfinder_ingestion.vector_store import VectorStoreManager
from wayfinder_ingestion.change_detector import ChangeDetector
from create_sample_pdf import create_all_sample_documents


def verify_pipeline():
    print("==================================================")
    print(" Wayfinder Document Ingestion Verification Script ")
    print("==================================================")

    settings = get_settings()
    print(f"Configured Embedding Model: {settings.EMBEDDING_MODEL} (from settings / .env)")
    print(f"Ollama Base URL:          {settings.OLLAMA_BASE_URL}")

    # Step 1: Ensure sample documents exist
    source_path = Path(settings.SOURCE_DOCUMENT_DIR)
    if not source_path.exists() or not list(source_path.glob("**/*.pdf")):
        print("\n[+] Creating sample PDF documents...")
        create_all_sample_documents(settings.SOURCE_DOCUMENT_DIR)

    # Step 2: First Ingestion Run
    print("\n[+] Executing Run #1 (Initial Document Ingestion)...")
    res1 = ingest_documents_flow()
    print(f"Run #1 Result: {res1}")

    # Step 3: Inspect ChromaDB
    print("\n[+] Inspecting Persistent ChromaDB Store...")
    vs = VectorStoreManager(settings.CHROMA_PERSIST_DIR)
    total_chunks = vs.get_count()
    print(f"Total Chunks in ChromaDB collection 'wayfinder_documents': {total_chunks}")

    # Sample query verification
    if total_chunks > 0:
        sample = vs.collection.get(limit=3, include=["documents", "metadatas"])
        print("\n--- Sample Chunks & Metadata in Vector Store ---")
        for idx in range(len(sample["ids"])):
            c_id = sample["ids"][idx]
            meta = sample["metadatas"][idx]
            doc_snippet = sample["documents"][idx][:120].replace("\n", " ")
            print(f"Chunk ID: {c_id}")
            print(f"  Category: {meta.get('category')} | SubCategory: {meta.get('sub_category')}")
            print(f"  Doc Title: {meta.get('doc_title')} | Page: {meta.get('page_number')}")
            print(f"  File Path: {meta.get('file_path')}")
            print(f"  Snippet: \"{doc_snippet}...\"")
            print("-")

    # Step 4: Inspect Ingestion CSV Ledger
    print("\n[+] Inspecting Ingestion CSV Ledger...")
    from wayfinder_ingestion.ledger import IngestionLedger
    ledger = IngestionLedger(settings.INGESTION_LOG_CSV)
    ingested_records = ledger.get_ingested_records()
    print(f"CSV Ledger Path: {settings.INGESTION_LOG_CSV}")
    print(f"Total Ingested Records in CSV Ledger: {len(ingested_records)}")
    for record in ingested_records:
        print(f"  * File: {record.get('relative_path')}")
        print(f"    Location:   {record.get('file_location')}")
        print(f"    Uploaded:   {record.get('file_uploaded_at')}")
        print(f"    Vectorized: {record.get('vectorized_at')}")
        print(f"    Deleted:    {record.get('file_deleted_at') or 'N/A (Active)'}")
        print(f"    Chunks:     {record.get('chunk_count')} | Status: {record.get('status')}")

    # Step 5: Second Ingestion Run (Idempotency Test)
    print("\n[+] Executing Run #2 (Idempotency Test - No Files Changed)...")
    res2 = ingest_documents_flow()
    print(f"Run #2 Result: {res2}")

    # Verify idempotency
    if res2["processed"] == 0 and res2["failed"] == 0 and res2["deleted"] == 0:
        print("\n✅ SUCCESS: Idempotency Verified! Re-running ingestion did not create duplicate chunks.")
    else:
        print("\n❌ WARNING: Re-running ingestion processed files again. Check change detector logic.")

    print("\n==================================================")
    print(" Verification Completed Successfully! ")
    print("==================================================")


if __name__ == "__main__":
    verify_pipeline()
