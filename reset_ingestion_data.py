#!/usr/bin/env python
"""
Wayfinder Clean Slate & Ingestion Reset Utility

Completely clears all ingested data to provide a clean slate for experiments:
1. Deletes all vectors and collections in ChromaDB (data/chroma_db)
2. Resets the CSV audit ledger (data/ingested_documents.csv) with fresh headers
3. Deletes the SHA-256 ingestion manifest (ingestion_manifest.json)
4. Cleans up any temporary staging files (*.tmp)
5. (Optional) Recreates sample PDFs or clears source documents

Usage examples:
    python reset_ingestion_data.py                      # Clean slate (keeps source PDFs)
    python reset_ingestion_data.py --recreate-samples   # Clean slate + generates fresh sample PDFs
    python reset_ingestion_data.py --delete-source-docs # Clean slate + deletes source PDFs too
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wayfinder_ingestion.config import get_settings
from wayfinder_ingestion.ledger import IngestionLedger


def reset_chroma_db(chroma_dir: Path):
    """Deletes all persistent ChromaDB files, databases, and manifests."""
    if chroma_dir.exists():
        print(f"[1/4] Clearing ChromaDB directory: {chroma_dir}")
        for item in chroma_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                print(f"      Removed: {item.name}")
            except Exception as e:
                print(f"      [!] Could not remove {item.name}: {e}")
    else:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        print(f"[1/4] Created empty ChromaDB directory: {chroma_dir}")

    # Ensure empty directory exists
    chroma_dir.mkdir(parents=True, exist_ok=True)
    print("  [✓] ChromaDB reset complete (0 vectors, 0 collections).")


def reset_csv_ledger(csv_path: Path):
    """Resets the CSV ledger with empty headers."""
    print(f"\n[2/4] Resetting CSV audit ledger: {csv_path}")
    try:
        if csv_path.exists():
            csv_path.unlink()
            print(f"      Removed old ledger: {csv_path.name}")
        # Initialize fresh ledger with correct headers
        ledger = IngestionLedger(csv_path=str(csv_path))
        print("  [✓] CSV ledger reinitialized with empty headers.")
    except Exception as e:
        print(f"  [!] Failed to reset CSV ledger: {e}")


def cleanup_temp_files(data_dir: Path):
    """Removes any temporary or staged files."""
    print(f"\n[3/4] Cleaning temporary files in: {data_dir}")
    count = 0
    if data_dir.exists():
        for tmp_file in data_dir.glob("**/*.tmp"):
            try:
                tmp_file.unlink(missing_ok=True)
                count += 1
            except Exception:
                pass
    print(f"  [✓] Removed {count} temporary file(s).")


def handle_source_documents(source_dir: Path, delete_source: bool, recreate_samples: bool):
    """Handles source documents directory based on user preferences."""
    print(f"\n[4/4] Checking source PDF documents directory: {source_dir}")

    if delete_source:
        if source_dir.exists():
            shutil.rmtree(source_dir, ignore_errors=True)
            print("  [✓] Deleted all source documents.")
        source_dir.mkdir(parents=True, exist_ok=True)
    elif recreate_samples:
        try:
            from create_sample_pdf import create_all_sample_documents
            create_all_sample_documents(str(source_dir))
            print("  [✓] Generated fresh sample PDF documents in HR, IT, and CustomerSupport.")
        except Exception as e:
            print(f"  [!] Failed to generate sample documents: {e}")
    else:
        pdf_count = len(list(source_dir.glob("**/*.pdf"))) if source_dir.exists() else 0
        print(f"  [i] Preserved {pdf_count} existing source PDF(s) ready for fresh ingestion.")


def main():
    parser = argparse.ArgumentParser(
        description="Wayfinder Clean Slate Utility - Clears all ingested data for fresh testing"
    )
    parser.add_argument(
        "--delete-source-docs",
        action="store_true",
        help="Also delete all files in data/source_documents"
    )
    parser.add_argument(
        "--recreate-samples",
        action="store_true",
        help="Recreate sample frontline PDF documents after cleaning"
    )
    args = parser.parse_args()

    settings = get_settings()
    chroma_dir = settings.chroma_dir_path
    csv_path = settings.csv_ledger_path
    source_dir = settings.source_dir_path
    data_dir = Path("./data").resolve()

    print("=" * 65)
    print(" Wayfinder Clean Slate - Reset Ingestion Data")
    print("=" * 65)
    print(f"ChromaDB Store : {chroma_dir}")
    print(f"CSV Ledger Path: {csv_path}")
    print(f"Source Folder  : {source_dir}")
    print("=" * 65)

    # 1. Clear ChromaDB
    reset_chroma_db(chroma_dir)

    # 2. Reset CSV Ledger
    reset_csv_ledger(csv_path)

    # 3. Clean temporary files
    cleanup_temp_files(data_dir)

    # 4. Handle source documents
    handle_source_documents(source_dir, args.delete_source_docs, args.recreate_samples)

    print("\n" + "=" * 65)
    print(" CLEAN SLATE READY!")
    print(" All vectors, manifest entries, and CSV logs have been cleared.")
    print(" You can now test document ingestion from scratch:")
    print("   1. python -m wayfinder_ingestion.pipeline   (Run batch ingestion)")
    print("   2. python watch_documents.py               (Run live watcher)")
    print("   3. python inspect_chroma.py                (Inspect vector store)")
    print("=" * 65)


if __name__ == "__main__":
    main()
