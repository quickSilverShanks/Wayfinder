"""
Search Audit Runner for Wayfinder Intelligent Document Retrieval.
Runs a batch of queries through the search pipeline and exports all results
into a single, flattened JSON file ready for direct import into Pandas DataFrames.

Usage:
    python run_search_audit.py
    python run_search_audit.py --output data/custom_audit.json --top-k 5
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default 5 benchmark queries covering available document categories
DEFAULT_AUDIT_QUERIES = [
    {
        "query": "where to get reliable documents for operations",
        "category": "SYF",
        "sub_category": "general",
        "description": "Operational guidelines and recommended source of truth (Synchrony)"
    },
    {
        "query": "what is minimum balance in savings account",
        "category": "SYF",
        "sub_category": "general",
        "description": "High-Yield Savings account requirements and deposit rules (Synchrony)"
    },
    {
        "query": "what is the fraud liability protection amount",
        "category": "SYF",
        "sub_category": "general",
        "description": "Authentication standards and password requirements (IT Security)"
    },
    # {
    #     "query": "sams club phone number",
    #     "category": "SYF",
    #     "sub_category": "general",
    #     "description": "Employee leave types and accrual policies (HR Leave Policy)"
    # },
    # {
    #     "query": "synchrony mastercard phone number",
    #     "category": "SYF",
    #     "sub_category": "general",
    #     "description": "Dispute handling and customer rights procedures (Customer Rights Policy)"
    # }
]


def execute_search_via_service(
    queries: List[Dict[str, Any]],
    top_k: int = 5,
    api_url: str = "http://localhost:8000/api/search"
) -> List[Dict[str, Any]]:
    """
    Executes search queries. Tries HTTP POST to FastAPI server first;
    if unavailable, transparently falls back to direct SearchService execution.
    """
    import urllib.request
    import urllib.error

    use_http = False
    try:
        req = urllib.request.Request("http://localhost:8000/api/health", headers={"User-Agent": "Wayfinder-Audit"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                use_http = True
    except Exception:
        use_http = False

    service = None
    if not use_http:
        print("[INFO] Live FastAPI server not detected on port 8000. Using in-process SearchService...")
        from wayfinder_search.service import SearchService
        service = SearchService()
    else:
        print(f"[INFO] Connected to live FastAPI server at {api_url}")

    all_rows: List[Dict[str, Any]] = []

    for q_idx, q_item in enumerate(queries, start=1):
        query_text = q_item["query"]
        category = q_item.get("category")
        sub_category = q_item.get("sub_category")
        desc = q_item.get("description", "")

        print(f"\n[{q_idx}/{len(queries)}] Executing query: '{query_text}' (Category: {category}/{sub_category})...")

        resp_dict = None
        start_time = time.perf_counter()

        if use_http:
            payload = json.dumps({
                "query": query_text,
                "number_of_results": top_k,
                "category": category,
                "sub_category": sub_category
            }).encode("utf-8")
            req = urllib.request.Request(
                api_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "Wayfinder-Audit"}
            )
            with urllib.request.urlopen(req) as response:
                resp_dict = json.loads(response.read().decode("utf-8"))
        else:
            from wayfinder_search.models import SearchRequest
            req_obj = SearchRequest(
                query=query_text,
                number_of_results=top_k,
                category=category,
                sub_category=sub_category
            )
            resp_obj = service.search(req_obj)
            resp_dict = resp_obj.model_dump()

        duration = round(time.perf_counter() - start_time, 4)

        # Extract primary passing results and below-threshold results
        results = resp_dict.get("results", []) or []
        below_results = resp_dict.get("below_threshold_results", []) or []
        threshold_met = resp_dict.get("threshold_met", False)
        rel_threshold = resp_dict.get("relevance_threshold", 0.10)
        th_green = resp_dict.get("threshold_green", 0.40)
        th_amber = resp_dict.get("threshold_amber", 0.10)
        candidates_retrieved = resp_dict.get("candidates_retrieved", 0)

        rank_counter = 1

        # Flatten passing results
        for item in results:
            all_rows.append({
                "query_id": q_idx,
                "query": query_text,
                "query_category": category,
                "query_sub_category": sub_category,
                "query_description": desc,
                "candidates_retrieved": candidates_retrieved,
                "search_duration_seconds": resp_dict.get("search_duration_seconds", duration),
                "threshold_met": threshold_met,
                "relevance_threshold": rel_threshold,
                "threshold_green": th_green,
                "threshold_amber": th_amber,
                "rank": rank_counter,
                "result_type": "primary",
                "is_above_threshold": True,
                "relevance_score": item.get("relevance_score"),
                "confidence_category": item.get("confidence_category", "green"),
                "document_title": item.get("document_title"),
                "category": item.get("category"),
                "sub_category": item.get("sub_category"),
                "source_document": item.get("source_document"),
                "page_number": item.get("page_number"),
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "chunk_text": item.get("chunk_text", "")
            })
            rank_counter += 1

        # Flatten below-threshold results
        for item in below_results:
            all_rows.append({
                "query_id": q_idx,
                "query": query_text,
                "query_category": category,
                "query_sub_category": sub_category,
                "query_description": desc,
                "candidates_retrieved": candidates_retrieved,
                "search_duration_seconds": resp_dict.get("search_duration_seconds", duration),
                "threshold_met": threshold_met,
                "relevance_threshold": rel_threshold,
                "threshold_green": th_green,
                "threshold_amber": th_amber,
                "rank": rank_counter,
                "result_type": "below_threshold",
                "is_above_threshold": False,
                "relevance_score": item.get("relevance_score"),
                "confidence_category": item.get("confidence_category", "red"),
                "document_title": item.get("document_title"),
                "category": item.get("category"),
                "sub_category": item.get("sub_category"),
                "source_document": item.get("source_document"),
                "page_number": item.get("page_number"),
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "chunk_text": item.get("chunk_text", "")
            })
            rank_counter += 1

        print(
            f"   -> Finished in {duration:.2f}s | "
            f"Passing (Primary): {len(results)} | Below-Threshold: {len(below_results)} | "
            f"Total Rows: {len(results) + len(below_results)}"
        )

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Run batch audit queries and export flattened JSON for Pandas DataFrame analysis."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/search_audit_results.json",
        help="Target output path for the flattened JSON results file"
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of combined results to retrieve per query (default: 5)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/api/search",
        help="FastAPI search endpoint URL (default: http://localhost:8000/api/search)"
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WAYFINDER SEARCH BATCH AUDIT RUNNER")
    print(f"Queries to execute : {len(DEFAULT_AUDIT_QUERIES)}")
    print(f"Retrievals / query : {args.top_k}")
    print(f"Target Output JSON : {output_path}")
    print("=" * 70)

    rows = execute_search_via_service(
        queries=DEFAULT_AUDIT_QUERIES,
        top_k=args.top_k,
        api_url=args.api_url
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"AUDIT RUN COMPLETE: Generated {len(rows)} total rows across {len(DEFAULT_AUDIT_QUERIES)} queries.")
    print(f"Saved to: {output_path}")
    print("=" * 70)
    print("\nHow to import into Pandas DataFrame:")
    print("----------------------------------------------------------------------")
    print("  import pandas as pd")
    print(f"  df = pd.read_json('{output_path}')")
    print("  print(df[['query', 'rank', 'result_type', 'confidence_category', 'relevance_score', 'document_title']].head(10))")
    print("----------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
