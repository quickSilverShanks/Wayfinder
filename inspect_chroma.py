#!/usr/bin/env python
"""
Wayfinder ChromaDB Inspection Utility

Inspect, search, and browse document chunks, vectors, and metadata
stored in the persistent ChromaDB vector database.

Usage examples:
    python inspect_chroma.py                         # View summary stats & sample chunks
    python inspect_chroma.py --limit 10              # View 10 recent chunks
    python inspect_chroma.py --query "leave policy"  # Perform semantic search
    python inspect_chroma.py --file "HR/leave.pdf"   # Filter chunks by specific file
    python inspect_chroma.py --category "HR"         # Filter chunks by category
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
    print(f"Total Stored Chunks: {total_count}")

    if total_count == 0:
        print("\nNo chunks currently stored in ChromaDB.")
        print("Run the ingestion pipeline first:")
        print("    python create_sample_pdf.py")
        print("    python -m wayfinder_ingestion.pipeline")
        return

    # Semantic search mode
    if query_text:
        print_divider(f"Semantic Search Query: '{query_text}'")
        try:
            where_filter = {}
            if file_filter:
                where_filter["file_path"] = file_filter
            if category_filter:
                where_filter["category"] = category_filter

            filter_arg = where_filter if where_filter else None

            # Perform similarity search using LangChain
            results = vs.similarity_search(query=query_text, k=limit, filter=filter_arg)
            print(f"Found {len(results)} matching chunks (limit: {limit}):")
            for idx, doc in enumerate(results, 1):
                chunk_id = doc.metadata.get("chunk_id", f"result_{idx}")
                display_chunk(idx, chunk_id, doc.page_content, doc.metadata)
        except Exception as e:
            print(f"Search failed: {e}")
            print("Note: Ensure Ollama is running (`ollama serve`) with the embedding model pulled.")
        return

    # Filtered or general inspection mode
    where_clause = {}
    if file_filter:
        where_clause["file_path"] = file_filter
    if category_filter:
        where_clause["category"] = category_filter

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

    # Overview of unique files stored
    all_data = vs.collection.get(include=["metadatas"])
    all_metas = all_data.get("metadatas", [])
    unique_files = sorted(list(set(m.get("file_path", "unknown") for m in all_metas if m)))
    unique_categories = sorted(list(set(m.get("category", "unknown") for m in all_metas if m)))

    print_divider("Indexed Files Summary")
    print(f"Unique Categories ({len(unique_categories)}): {', '.join(unique_categories)}")
    print(f"Unique Documents  ({len(unique_files)}):")
    for f in unique_files:
        count = sum(1 for m in all_metas if m and m.get("file_path") == f)
        print(f"  • {f} ({count} chunks)")

    print_divider(f"Displaying Sample Chunks (Showing {len(ids)} of {total_count})")
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
