# Wayfinder — PDF Document Ingestion Pipeline (Part 1)

An enterprise-grade, event-driven PDF document ingestion pipeline for **Wayfinder**, designed for frontline and customer-support teams.

This pipeline reads PDF documents from a configurable folder hierarchy, detects new/modified/unchanged/deleted files, converts PDFs to structured **Markdown (MD)** using **Docling**, redacts PII using standard Python regex patterns, applies **recursive chunking with overlap** via **LangChain**'s `RecursiveCharacterTextSplitter` while preserving structural metadata (`category`, `sub_category`, document title, page numbers, file path, file hash), vectorizes chunks via an **Ollama-hosted embedding model**, persists vectors in **ChromaDB**, maintains a detailed **CSV audit ledger of ingested documents**, and provides a directory watcher that monitors source folders **only while the Prefect server is running**.

---

## 🏗 Pipeline Architecture

```text
PDF Source Directory (e.g. data/source_documents/HR/Policies/leave.pdf)
      ↓
Directory Watcher (Runs ONLY as long as Prefect server is active)
      ↓
Change Detection (SHA-256 state manifest comparison: new/modified/deleted/unchanged)
      ↓
Docling PDF to Markdown Converter (Preserves document title, tables & page numbers)
      ↓
PII Masking (Redact emails, phone numbers, SSNs, credit cards, IPs)
      ↓
LangChain Recursive Chunker (RecursiveCharacterTextSplitter with overlap & metadata preservation)
      ↓
Ollama Embeddings Service (Generate vector embeddings via model configured in .env: EMBEDDING_MODEL)
      ↓
LangChain Chroma Vector Store (Store vectors & metadata idempotently, native retriever support)
      ↓
CSV Ingestion Ledger (data/ingested_documents.csv — logs file location, upload time, vectorization time)
      ↓
Prefect Workflow Orchestrator (Flow & Task error isolation)
```

---

## ⚙️ Configuration Setup

### 1. Create `.env` File

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### 2. Environment Variables

| Variable | Default Value | Description |
|---|---|---|
| `SOURCE_DOCUMENT_DIR` | `./data/source_documents` | Source directory containing PDF documents |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Persistent ChromaDB data directory |
| `INGESTION_LOG_CSV` | `./data/ingested_documents.csv` | Path to CSV ledger of ingested documents |
| `WATCH_INTERVAL_SECONDS`| `5` | Polling interval (seconds) for folder watcher |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for Ollama service |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model name (always picked from settings in `.env`) |
| `CHAT_MODEL` | `llama3.2` | Ollama chat model name |
| `CHUNK_SIZE` | `1000` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap size between adjacent chunks |
| `PII_MASKING_ENABLED` | `True` | Toggle PII redaction (True/False) |
| `LOG_LEVEL` | `INFO` | Application log verbosity (DEBUG, INFO, WARNING, ERROR) |
| `PREFECT_API_URL` | `http://localhost:4200/api` | Prefect API URL for server connection and health checks |

---

## 📊 Ingested Documents CSV Ledger

The pipeline automatically maintains a CSV ledger at `data/ingested_documents.csv` (configurable via `INGESTION_LOG_CSV`). Every time a PDF document is parsed, masked, embedded, and successfully indexed into ChromaDB, its record is created or updated.

### Ledger Fields

| Column | Description | Example |
|---|---|---|
| `file_name` | Name of the PDF file | `leave_policy.pdf` |
| `relative_path` | Path relative to `SOURCE_DOCUMENT_DIR` | `HR/Policies/leave_policy.pdf` |
| `file_location` | Absolute path on disk where file was located | `C:/Users/.../data/source_documents/HR/Policies/leave_policy.pdf` |
| `category` | Derived primary category | `HR` |
| `sub_category` | Derived sub-category | `Policies` |
| `file_hash` | SHA-256 hash of the file content | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `file_size_bytes` | Document file size in bytes | `24580` |
| `file_uploaded_at` | Timestamp when the file was created/uploaded to disk | `2026-09-28T07:30:00+00:00` |
| `vectorized_at` | Timestamp when successfully vectorized into ChromaDB | `2026-09-28T07:35:12+00:00` |
| `file_deleted_at` | Timestamp when the file was deleted from disk/ChromaDB | `2026-09-28T08:00:00+00:00` (or empty string if active) |
| `chunk_count` | Number of chunks vectorized and indexed into ChromaDB | `4` |
| `status` | Current status (`active` or `deleted`) | `active` |
| `last_updated_at` | Timestamp when this ledger entry was last updated | `2026-09-28T07:35:12+00:00` |

### Delete & Re-upload Lifecycle Tracking
- **New File Upload**: Appends a new record with `status="active"`, its `file_uploaded_at` timestamp, and `file_deleted_at=""`.
- **File Deletion**: When an active file is removed from source documents, its vectors are removed from ChromaDB, its existing record in the CSV is updated with `status="deleted"`, and `file_deleted_at` is set while retaining its original `file_uploaded_at` and `vectorized_at` timestamps.
- **Re-uploading at Same Location**: If a file of the same name (e.g. `xyz.pdf`) is re-uploaded at the same location after being deleted, a **new record** is created with `status="active"` and its new `file_uploaded_at` timestamp. **Both the historical `deleted` entry and the new `active` entry remain in the CSV**, preserving an immutable audit trail.

---

## 🚀 Step-by-Step Installation & Running Guide

### ⚡ Quick Start (TL;DR)

For a rapid startup using defaults:

```bash
# 1. Environment & Dependencies
cp .env.example .env
python -m venv venv
.\venv\Scripts\Activate.ps1    # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# 2. Services (Open separate terminals)
ollama serve                   # Terminal 1: Ollama API (with nomic-embed-text pulled)
prefect server start           # Terminal 2: Prefect Server (required for folder watcher)

# 3. Create Sample Documents & Start Watcher (Terminal 3)
python create_sample_pdf.py
python watch_documents.py

# 4. Verify & Run Unit Tests (Terminal 4)
python verify_ingestion.py
pytest -v
```

---

### Step 1: Install Python Dependencies

Create a virtual environment and install all required packages:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2: Start Ollama Service via Docker

Run Ollama as a standalone Docker container.

#### Option A: CPU Execution (Standard Docker)

```bash
docker run -d \
  --name ollama \
  -v ollama_storage:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:latest
```

#### Option B: GPU Execution (NVIDIA GPU Acceleration)

```bash
docker run -d \
  --gpus=all \
  --name ollama \
  -v ollama_storage:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:latest
```

> **Note**: If running Ollama natively on your host machine without Docker, simply launch the Ollama application or run `ollama serve`.

---

### Step 3: Pull Embedding & Chat Models in Ollama

Download the required models into the running Ollama container:

```bash
# Pull the configured embedding model
docker exec -it ollama ollama pull nomic-embed-text

# Pull the configured chat model
docker exec -it ollama ollama pull llama3.2
```

*(Or if using local native Ollama: `ollama pull nomic-embed-text` and `ollama pull llama3.2`)*

---

### Step 4: Start Prefect Server

The folder watcher requires the Prefect server to be actively running. Start the Prefect server in a dedicated terminal window:

```bash
prefect server start
```

Once running, the Prefect dashboard is accessible at `http://localhost:4200` and the API endpoint at `http://localhost:4200/api`.

---

### Step 5: Generate Sample PDF Documents

Generate realistic sample PDF documents with nested category structures (`HR/Policies/`, `IT/Security/`) and embedded PII samples:

```bash
python create_sample_pdf.py
```

---

### Step 6: Run the Ingestion Pipeline

You can run the ingestion pipeline in either of two modes:

#### Option A: Continuous Directory Watcher powered by Watchdog ("ONLY" while Prefect Server is Running)

The folder watcher uses the **Watchdog** library (`Observer` and `FileSystemEventHandler`) to monitor `data/source_documents` (and all nested subfolders) for real-time filesystem events (new, modified, or deleted PDF documents) with debouncing. It enforces that watching occurs **"ONLY" as long as the Prefect server is active**:

```bash
python watch_documents.py
```

*(Alternatively: `python src/wayfinder_ingestion/pipeline.py --watch`)*

- **Watchdog Event-Driven**: Detects file creation, modification, and deletion events immediately at the OS level instead of requiring high-frequency disk polling.
- **If Prefect server is down at launch**: The watcher checks server health, refuses to start, and instructs you to start Prefect server first.
- **If Prefect server stops while watching**: The watcher detects the stoppage immediately on the next health check, shuts down the Watchdog observer, and terminates the watch loop.
- **When a document is dropped or modified in a folder/subfolder**: The watcher triggers `ingest_documents_flow` automatically, indexes vectors in ChromaDB, and updates `data/ingested_documents.csv`.

#### Option B: Single Batch Run

To execute a single batch ingestion run:

```bash
python src/wayfinder_ingestion/pipeline.py
```

---

### Step 7: Verify Indexed Data, CSV Ledger & Idempotency

Run the automated verification script to validate ChromaDB indexing, metadata preservation, and the CSV ledger:

```bash
python verify_ingestion.py
```

This script will:
1. Ensure sample PDFs exist in `data/source_documents`.
2. Run document ingestion and index chunks into ChromaDB.
3. Inspect the persistent ChromaDB collection and display sample vectorized chunks.
4. Inspect the **CSV Ledger** (`data/ingested_documents.csv`) and print each document's upload time and vectorization timestamp.
5. Run ingestion a second time to verify that **no duplicate chunks are created** (idempotency test).

---

### Step 8: Inspect and Query ChromaDB Data

Inspect, browse, and search vectors and chunks stored in ChromaDB using the dedicated CLI tool:

```bash
# View collection summary, indexed documents breakdown & sample chunks
python inspect_chroma.py

# View top 10 chunks with metadata
python inspect_chroma.py --limit 10

# Perform semantic vector similarity search
python inspect_chroma.py --query "frontline leave policy and entitlements"

# Filter chunks by specific file or category
python inspect_chroma.py --file "HR/Policies/leave_policy.pdf"
python inspect_chroma.py --category "HR"
```

---

### Step 9: Run Unit Tests

Execute the automated pytest suite covering Docling PDF conversion, folder category derivation, PII masking, LangChain recursive chunking with overlap, ChromaDB vector store operations, CSV audit ledger tracking, and Watchdog Prefect server health checks:

```bash
# Run all unit tests
pytest -v

# Run individual test suites
pytest tests/test_pdf_parser.py -v         # Docling PDF to Markdown parsing
pytest tests/test_chunker.py -v            # LangChain recursive text chunking with overlap & metadata
pytest tests/test_embedding_service.py -v  # Ollama embedding service picking model from .env/settings
pytest tests/test_ledger.py -v             # CSV audit ledger (active/deleted/reupload lifecycle)
pytest tests/test_watcher.py -v            # Watchdog & Prefect server health monitoring
pytest tests/test_change_detector.py -v    # State manifest & change detection
pytest tests/test_metadata_extraction.py -v# Directory hierarchy category derivation
pytest tests/test_pii_masker.py -v         # Regex PII redaction
pytest tests/test_vector_store.py -v       # ChromaDB upsert, query, and delete operations
```

---

### Step 10: Development & Maintenance Utilities

#### 1. Stop Prefect Server & Kill Watcher Processes
Gracefully terminates any running Prefect server (frees port 4200) and halts running watcher background processes:

```bash
python stop_services.py
```

#### 2. Reset Ingested Data for a Clean Slate
Clears all ChromaDB vectors, wipes the manifest, and resets the CSV audit ledger back to clean headers, allowing you to re-test document ingestion experiments from scratch:

```bash
# Clean slate (keeps existing source PDFs ready to be re-ingested)
python clean_slate.py

# Clean slate + regenerates fresh sample PDFs
python clean_slate.py --recreate-samples

# Clean slate + deletes source PDFs as well
python clean_slate.py --delete-source-docs
```

---

## 📁 Repository Folder Structure

```text
Wayfinder/
├── .env.example                # Template configuration file
├── .gitignore                  # Git ignore rules for venv, data, and cache
├── requirements.txt            # Python dependencies (Docling, LangChain, ChromaDB, Ollama, Prefect, Watchdog, Pytest)
├── README.md                   # Full documentation and exact commands
├── create_sample_pdf.py        # Helper script to create test PDFs with folder hierarchy
├── watch_documents.py          # Monitored directory watcher (runs ONLY while Prefect server is active)
├── verify_ingestion.py         # End-to-end pipeline verification & CSV ledger inspection script
├── src/
│   └── wayfinder_ingestion/
│       ├── __init__.py
│       ├── config.py           # Pydantic settings management
│       ├── logging_config.py   # Secure structured logging
│       ├── change_detector.py  # Change detection & category extraction with file upload stats
│       ├── ledger.py           # CSV ledger tracking ingested documents, upload/vectorize timestamps
│       ├── watcher.py          # Folder watcher with continuous Prefect server health monitoring
│       ├── pdf_parser.py       # Docling PDF to Markdown parser & metadata preservation
│       ├── pii_masker.py       # PII redaction component (Emails, Phone, SSN, Credit Card, IP)
│       ├── chunker.py          # LangChain recursive character text splitting with overlap
│       ├── embedding_service.py# Ollama vector embedding client (reads model from .env)
│       ├── vector_store.py     # Persistent ChromaDB client manager
│       └── pipeline.py         # Prefect tasks, flows, and CLI runner
└── tests/
    ├── __init__.py
    ├── test_change_detector.py # Tests for file state detection
    ├── test_metadata_extraction.py # Tests for directory hierarchy parsing
    ├── test_pii_masker.py     # Tests for PII regex masking
    ├── test_pdf_parser.py     # Tests for Docling PDF to Markdown formatting
    ├── test_chunker.py        # Tests for LangChain recursive text chunking and metadata preservation
    ├── test_embedding_service.py # Tests for embedding model selection from .env/settings
    ├── test_vector_store.py   # Tests for ChromaDB storage, query, and deletion
    ├── test_ledger.py         # Tests for CSV ledger creation, upload/vectorize timestamps, updates, deletions
    └── test_watcher.py        # Tests for watcher Prefect server health checks and lifecycle termination
```

---

## 🔒 Security & Logging

- **PII Masking**: Redacts sensitive customer/employee data before generating embeddings or storing text in ChromaDB.
- **Privacy Logging**: Log statements record file names, relative paths, chunk counts, execution durations, and error details, **never** logging raw document contents or sensitive values.
- **Auditable Ledger**: Maintains an accurate, timestamped CSV ledger of all ingested and vectorized documents for auditing and frontline traceability.
