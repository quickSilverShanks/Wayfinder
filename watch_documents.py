#!/usr/bin/env python
"""
Wayfinder Document Ingestion - Monitored Folder Watcher

Monitors the data/source_documents folder (and all nested subfolders) for PDF changes
'ONLY' as long as the Prefect server is actively running.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env early
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

if not os.environ.get("PREFECT_API_URL"):
    os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wayfinder_ingestion.watcher import main

if __name__ == "__main__":
    main()
