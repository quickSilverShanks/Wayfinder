#!/usr/bin/env python
"""
Wayfinder ChromaDB Inspection Utility

Inspect, search, and browse document chunks, vectors, and metadata
stored in the persistent ChromaDB vector database.

Usage examples:
    python inspect_chroma.py                         # View summary stats & sample chunks
    python inspect_chroma.py --limit 10              # View 10 recent chunks
    python inspect_chroma.py --query "leave policy"  # Perform semantic search
    python inspect_chroma.py --file "HR/Policies/leave_policy.pdf"  # Filter chunks by exact file path
    python inspect_chroma.py --category "HR"                        # Filter chunks by category
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wayfinder_ingestion.config import get_settings
from wayfinder_ingestion.vector_store import VectorStoreManager


def print_divider(title: str = ""):
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("-" * 65)


def display_chunk(idx: int, chunk_id: str, document: str, metadata: Dict[str, Any], score: Optional[float] = None):
    print(f"\n[{idx}] Chunk ID: {chunk_id}")
    if score is not None:
        print(f"    Similarity Distance: {score:.4f}")
    print(f"    Document Title : {metadata.get('doc_title', 'N/A')}")
    print(f"    Category       : {metadata.get('category', 'N/A')} > {metadata.get('sub_category', 'N/A')}")
    print(f"    Source File    : {metadata.get('file_path', 'N/A')} (Page {metadata.get('page_number', 'N/A')})")
    print(f"    Length         : {len(document)} chars ({metadata.get('char_count', 'N/A')} recorded)")
    
    snippet = document.strip().replace("\r", "")
    lines = snippet.split("\n")
    preview_lines = lines[:6]
    preview = "\n      ".join(preview_lines)
    print(f"    Text Preview   :\n      {preview}")
    if len(lines) > 6 or len(snippet) > 300:
        print("      [... truncated ...]")


def build_chroma_where(file_path: Optional[str] = None, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Build a valid ChromaDB where clause handling single or compound conditions."""
    conditions = []
    if file_path:
        conditions.append({"file_path": file_path})
    if category:
        conditions.append({"category": category})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def inspect_chroma(
    limit: int = 5,
    query_text: Optional[str] = None,
    file_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    persist_dir: Optional[str] = None
):
    settings = get_settings()
    chroma_path = persist_dir or settings.CHROMA_PERSIST_DIR
    
    print_divider("Wayfinder ChromaDB Inspector")
    print(f"Persist Directory : {Path(chroma_path).resolve()}")
    print(f"Collection Name   : {VectorStoreManager.COLLECTION_NAME}")
    print(f"Embedding Model   : {settings.EMBEDDING_MODEL} (from settings/.env)")

    vs = VectorStoreManager(persist_dir=chroma_path)
    total_count = vs.get_count()

    if total_count == 0:
        print("Total Stored Chunks: 0")
        print("\nNo chunks currently stored in ChromaDB.")
        print("Run the ingestion pipeline first:")
        print("    python create_sample_pdf.py")
        print("    python -m wayfinder_ingestion.pipeline")
        return

    # Retrieve all metadatas to analyze categories and documents
    all_data = vs.collection.get(include=["metadatas"])
    all_metas = all_data.get("metadatas", [])
    unique_files = sorted(list(set(m.get("file_path", "unknown") for m in all_metas if m)))
    unique_categories = sorted(list(set(m.get("category", "unknown") for m in all_metas if m)))

    # Check if specified file is present in exact path
    matched_file = None
    if file_filter:
        norm_filter = file_filter.replace("\\", "/").strip()
        for f in unique_files:
            if f.replace("\\", "/").strip() == norm_filter:
                matched_file = f
                break

        if not matched_file:
            print(f"Total Stored Chunks: {total_count}")
            print_divider("File Filter Error")
            print(f"File '{file_filter}' was not found in ChromaDB.\n")
            print(f"Available files ({len(unique_files)}):")
            for f in unique_files:
                print(f"  • {f}")
            print("\nPlease pass one of the exact file paths listed above with --file.")
            return

    # Check if specified category is present
    matched_category = None
    if category_filter:
        norm_cat = category_filter.strip().lower()
        for c in unique_categories:
            if c.strip().lower() == norm_cat:
                matched_category = c
                break

        if not matched_category:
            print(f"Total Stored Chunks: {total_count}")
            print_divider("Category Filter Error")
            print(f"Category '{category_filter}' was not found in ChromaDB.\n")
            print(f"Available categories ({len(unique_categories)}):")
            for c in unique_categories:
                print(f"  • {c}")
            print("\nPlease pass one of the exact categories listed above with --category.")
            return

    where_clause = build_chroma_where(file_path=matched_file, category=matched_category)
    is_filtered = bool(where_clause)

    # Compute matching count and matching metadatas
    if is_filtered:
        def matches_filter(m: Dict[str, Any]) -> bool:
            if not m:
                return False
            if matched_file and m.get("file_path") != matched_file:
                return False
            if matched_category and m.get("category") != matched_category:
                return False
            return True

        matching_metas = [m for m in all_metas if matches_filter(m)]
        matching_count = len(matching_metas)
    else:
        matching_metas = all_metas
        matching_count = total_count

    print(f"Total Stored Chunks: {total_count}")
    if is_filtered:
        filter_parts = []
        if matched_file:
            filter_parts.append(f"file='{matched_file}'")
        if matched_category:
            filter_parts.append(f"category='{matched_category}'")
        print(f"Active Filter     : {', '.join(filter_parts)}")
        print(f"Matching Chunks   : {matching_count} (out of {total_count} total stored)")

    # Semantic search mode
    if query_text:
        print_divider(f"Semantic Search Query: '{query_text}'")
        try:
            # Perform similarity search using LangChain with resolved filter
            results = vs.similarity_search(query=query_text, k=limit, filter=where_clause)
            print(f"Found {len(results)} matching chunks (limit: {limit}):")
            for idx, doc in enumerate(results, 1):
                chunk_id = doc.metadata.get("chunk_id", f"result_{idx}")
                display_chunk(idx, chunk_id, doc.page_content, doc.metadata)
        except Exception as e:
            print(f"Search failed: {e}")
            print("Note: Ensure Ollama is running (`ollama serve`) with the embedding model pulled.")
        return

    # Files Summary Section
    summary_files = sorted(list(set(m.get("file_path", "unknown") for m in matching_metas if m)))
    summary_categories = sorted(list(set(m.get("category", "unknown") for m in matching_metas if m)))

    if is_filtered:
        print_divider("Filtered Files Summary")
        print(f"Matching Categories ({len(summary_categories)}): {', '.join(summary_categories) if summary_categories else 'None'}")
        print(f"Matching Documents  ({len(summary_files)}):")
        if summary_files:
            for f in summary_files:
                count = sum(1 for m in matching_metas if m and m.get("file_path") == f)
                print(f"  • {f} ({count} chunks)")
        else:
            print("  (No matching documents found)")
    else:
        print_divider("Indexed Files Summary")
        print(f"Unique Categories ({len(unique_categories)}): {', '.join(unique_categories)}")
        print(f"Unique Documents  ({len(unique_files)}):")
        for f in unique_files:
            count = sum(1 for m in all_metas if m and m.get("file_path") == f)
            print(f"  • {f} ({count} chunks)")

    # Retrieve chunks for inspection
    get_kwargs = {
        "limit": limit,
        "include": ["documents", "metadatas"]
    }
    if where_clause:
        get_kwargs["where"] = where_clause

    data = vs.collection.get(**get_kwargs)
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    if is_filtered:
        print_divider(f"Displaying Sample Chunks (Showing {len(ids)} of {matching_count})")
    else:
        print_divider(f"Displaying Sample Chunks (Showing {len(ids)} of {total_count})")

    if not ids:
        if is_filtered:
            print("\nNo chunks found matching the specified filter.")
        else:
            print("\nNo chunks found in the database.")
    else:
        for idx in range(len(ids)):
            display_chunk(
                idx=idx + 1,
                chunk_id=ids[idx],
                document=docs[idx] if idx < len(docs) else "",
                metadata=metas[idx] if idx < len(metas) else {}
            )

    print("\n" + "=" * 65)
    print("Tip: Run `python inspect_chroma.py --query \"your question\"` to test vector search.")


def main():
    parser = argparse.ArgumentParser(description="Inspect document chunks and metadata inside ChromaDB")
    parser.add_argument("--limit", type=int, default=5, help="Number of chunks to display (default: 5)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Query text for semantic vector search")
    parser.add_argument("--file", "-f", type=str, default=None, help="Filter by relative file path (e.g. 'HR/leave.pdf')")
    parser.add_argument("--category", "-c", type=str, default=None, help="Filter by category (e.g. 'HR')")
    parser.add_argument("--dir", type=str, default=None, help="Custom ChromaDB persist directory")
    args = parser.parse_args()

    inspect_chroma(
        limit=args.limit,
        query_text=args.query,
        file_filter=args.file,
        category_filter=args.category,
        persist_dir=args.dir
    )


if __name__ == "__main__":
    main()
