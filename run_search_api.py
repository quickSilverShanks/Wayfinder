#!/usr/bin/env python
"""
Wayfinder Search API Server Runner.
Launches the FastAPI application via Uvicorn.

Usage:
    python run_search_api.py
    python run_search_api.py --port 8000 --reload
"""

import sys
from pathlib import Path
import argparse
import uvicorn

# Ensure src is at the front of sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Run Wayfinder Search API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")
    args = parser.parse_args()

    print(f"Starting Wayfinder Search API at http://{args.host}:{args.port}")
    print(f"Interactive Swagger Docs available at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "wayfinder_search.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
