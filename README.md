# Wayfinder — Intelligent Document Retrieval System

An enterprise-grade document ingestion and intelligent semantic search platform for **Wayfinder**, designed for frontline and customer-support teams to rapidly find relevant policies, guidelines, and procedures.

---

## 📑 Part 1 — PDF Document Ingestion Pipeline

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

## 🔍 Part 2 — Search & Retrieval Backend

The Search & Retrieval Backend provides an intelligent search API enabling frontline workers to query enterprise policies, guidelines, and procedures. It combines dense semantic embeddings with lexical BM25 matching, reranks merged candidates with a cross-encoder, and filters results using a configurable relevance threshold before returning structured, UI-ready payloads through FastAPI.

---

### 🏗 Retrieval Flow Architecture

```text
User Query (e.g. "How should I handle a customer disputing a transaction?")
    ↓
Query Processing & Input Validation
    ↓
Hybrid Retrieval
 ┌─────────────────────────────┐
 │ Dense Semantic Search       │ (ChromaDB + Ollama Embedding: EMBEDDING_MODEL)
 │ +                           │
 │ BM25 Lexical Keyword Search │ (Pure-Python Inverted Index)
 └──────────────┬──────────────┘
                ↓
    Candidate Document Chunks (Reciprocal Rank Fusion RRF Merge & Deduplication)
                ↓
    BGE Reranker v2 M3 (BAAI/bge-reranker-v2-m3 Cross-Encoder)
                ↓
    Relevance Threshold (score >= RELEVANCE_THRESHOLD)
                ↓
    Top-K Results (number_of_results requested or DEFAULT_TOP_K)
                ↓
    FastAPI Response (POST /api/search)
```

---

### 🧩 Architectural Separation

The search subsystem maintains a strict layered architecture:

```text
FastAPI Endpoints (wayfinder_search/api.py)
   ↓
Search Service (wayfinder_search/service.py)
   ↓
Hybrid Retriever (wayfinder_search/hybrid_retriever.py)
   ├─ Dense Retriever (wayfinder_ingestion/vector_store.py + embedding_service.py)
   └─ Lexical Retriever (wayfinder_search/bm25.py)
   ↓
BGE Cross-Encoder Reranker (wayfinder_search/reranker.py)
   ↓
ChromaDB Collection (`wayfinder_documents`)
```

- **FastAPI Layer (`api.py`)**: Exposes REST routes (`POST /api/search`, `GET /api/health`, `GET /`), manages CORS, handles request validation exceptions, and serializes responses. Contains zero retrieval logic.
- **Search Service Layer (`service.py`)**: Central orchestrator. Manages input validation, candidate pool sizing, threshold filtering, duration measurement, and result packaging. Exposes clean interfaces reusable by future MCP servers.
- **Hybrid Retriever Layer (`hybrid_retriever.py`)**: Queries ChromaDB for vector similarity and the in-memory BM25 index for keyword matches, merging candidates using Reciprocal Rank Fusion (RRF).
- **Reranker Layer (`reranker.py`)**: Cross-encoder scoring that evaluates query-chunk pairs and generates normalized `[0.0, 1.0]` relevance probabilities. Model identifier is strictly loaded from settings.

---

### 🔗 How Part 2 Connects to Part 1

1. **Shared ChromaDB Collection**: Part 2 connects directly to the persistent ChromaDB collection (`wayfinder_documents` in `data/chroma_db`) populated during the Part 1 ingestion pipeline.
2. **Unified Embedding Service**: Dense query embeddings are generated using the exact same Ollama embedding model (`EMBEDDING_MODEL` e.g. `qwen3-embedding:0.6b`) hosted at `OLLAMA_BASE_URL` that was used to embed chunks in Part 1.
3. **Preserved Metadata**: Structural metadata extracted in Part 1 (`doc_title`, `category`, `sub_category`, `file_path`, `page_number`, `file_hash`, `chunk_id`) is surfaced directly in each search result item for rich UI display and citation.

---

### ⚙️ Search Configuration Setup

#### Environment Variables (`.env`)

Add the following Part 2 settings to your `.env` file:

| Variable | Default Value | Description |
|---|---|---|
| `DEFAULT_TOP_K` | `5` | Default number of search results returned when omitted from API request |
| `MIN_TOP_K` | `1` | Minimum allowed `number_of_results` in API requests |
| `MAX_TOP_K` | `50` | Maximum allowed `number_of_results` in API requests |
| `DENSE_TOP_K` | `15` | Number of candidate chunks retrieved via ChromaDB dense vector search |
| `BM25_TOP_K` | `15` | Number of candidate chunks retrieved via BM25 lexical search |
| `RERANK_TOP_K` | `25` | Maximum number of merged candidates sent to the BGE cross-encoder reranker |
| `RELEVANCE_THRESHOLD`| `0.20` | Minimum score threshold for qualifying results (0.0 to 1.0) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model identifier (never hard-coded in Python code) |
| `RERANKER_DEVICE` | `auto` | Device for reranker execution (`auto`, `cuda`, or `cpu`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service base URL (used for query embeddings) |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model hosted on Ollama (must match Part 1) |

---

### 🚀 Step-by-Step Running Guide

#### Step 1: Ensure Conda Environment is Activated

```bash
conda activate wayfinder
```

#### Step 2: Start Ollama Service

Ensure your Ollama service is running (in Docker or natively):

```bash
# If using Docker:
docker run -d \
  --name ollama \
  -v ollama_storage:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:latest

# Pull the configured embedding model (used for query embeddings):
docker exec -it ollama ollama pull qwen3-embedding:0.6b
```

*(If running native Ollama: `ollama pull qwen3-embedding:0.6b`)*

#### Step 3: Reranker Model Setup

The reranker model is configured via `RERANKER_MODEL=BAAI/bge-reranker-v2-m3` in `.env`.
The weights (~1.1 GB) are automatically downloaded and cached on first startup to `~/.cache/huggingface/hub/`.

If you prefer to pull the model to Ollama in Docker:
```bash
docker exec -it ollama ollama pull qllama/bge-reranker-v2-m3
```

#### Step 4: Start the FastAPI Search Server

Launch the search backend using the runner script:

```bash
python run_search_api.py --port 8000 --reload
```

Or using Uvicorn directly:

```bash
uvicorn wayfinder_search.api:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **Health Check Endpoint**: `http://localhost:8000/api/health`

---

### 📡 API Usage & Examples

#### Endpoint: `POST /api/search`

##### Request Schema

```json
{
  "query": "How should I handle a customer disputing a transaction?",
  "number_of_results": 5,
  "category": "Payments",
  "sub_category": "Disputes"
}
```

- `query` (string, required): Natural-language question or search term.
- `number_of_results` (integer, optional): Desired top results count. Defaults to `DEFAULT_TOP_K` (5) if omitted. Validated between `MIN_TOP_K` (1) and `MAX_TOP_K` (50).
- `category` (string, optional): Filter by primary folder category (e.g. `HR`, `IT`, `samples`).
- `sub_category` (string, optional): Filter by sub-category (e.g. `Policies`, `Security`, `Disputes`).

---

#### Example 1: cURL Request with Default Result Count

```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "How should I handle a customer disputing a transaction?"}'
```

#### Example 2: PowerShell Request with Custom `number_of_results` & Filter

```powershell
$body = @{
    query = "frontline leave entitlements and annual vacation"
    number_of_results = 3
    category = "HR"
    sub_category = "Policies"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/search" `
    -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

#### Example 3: Python Client Request

```python
import requests

payload = {
    "query": "How should I handle a customer disputing a transaction?",
    "number_of_results": 3
}

response = requests.post("http://localhost:8000/api/search", json=payload)
data = response.json()

print(f"Threshold Met: {data['threshold_met']}")
print(f"Candidates Retrieved: {data['candidates_retrieved']}")
print(f"Search Duration: {data['search_duration_seconds']}s")

for result in data["results"]:
    print(f"\n[{result['relevance_score']}] {result['document_title']} (Page {result['page_number']})")
    print(f"Source: {result['source_document']}")
    print(f"Snippet: {result['chunk_text'][:150]}...")
```

---

#### Example API Response (High Relevance Match)

```json
{
  "query": "How should I handle a customer disputing a transaction?",
  "requested_results": 3,
  "candidates_retrieved": 25,
  "search_duration_seconds": 0.8421,
  "threshold_met": true,
  "relevance_threshold": 0.20,
  "total_results": 1,
  "results": [
    {
      "document_title": "customer-rights-policy",
      "category": "samples",
      "sub_category": "General",
      "source_document": "samples/customer-rights-policy.pdf",
      "relevance_score": 0.2265,
      "chunk_text": "- (xi) Clearly spell out, at the time of establishing a customer relationship, the liability for losses, as well as the grievance redressal and dispute procedures...",
      "page_number": 1,
      "document_id": "02bede9fa4f38b5171673655ec50ffb6c8b5c7c956b7cfa1b01e58204fd2a83e",
      "chunk_id": "02bede9fa4f38b5171673655ec50ffb6c8b5c7c956b7cfa1b01e58204fd2a83e_p1_c1"
    }
  ],
  "below_threshold_results": null,
  "message": "Successfully retrieved 1 document chunk(s) exceeding relevance threshold 0.2."
}
```

#### Example API Response (Below Threshold / No-Result Scenario)

When a query has no semantically relevant documents in ChromaDB, the API returns `threshold_met: false` and `results: []` rather than hallucinating low-quality matches:

```json
{
  "query": "quantum physics entanglement superstring dimensions",
  "requested_results": 5,
  "candidates_retrieved": 15,
  "search_duration_seconds": 0.7255,
  "threshold_met": false,
  "relevance_threshold": 0.20,
  "total_results": 0,
  "results": [],
  "below_threshold_results": [
    {
      "document_title": "password_guidelines",
      "category": "IT",
      "sub_category": "Security",
      "source_document": "IT/Security/password_guidelines.pdf",
      "relevance_score": 0.0004,
      "chunk_text": "Wayfinder IT Security Standards...",
      "page_number": 1,
      "document_id": "afebd037ee02fcc1b1d738ce0b34e6f390ca0e1c5935e7618955e22422775a88",
      "chunk_id": "afebd037ee02fcc1b1d738ce0b34e6f390ca0e1c5935e7618955e22422775a88_p1_c1"
    }
  ],
  "message": "No retrieved document chunk met the minimum relevance threshold of 0.2. Retrieved 15 candidate(s) but none demonstrated sufficient semantic confidence."
}
```

---

### 🧪 Automated Verification & Testing

#### 1. End-to-End Verification Script

Execute the automated verification script that tests health checks, default result counts, dynamic top-k (k=1, k=3), category filtering, irrelevant query rejection, and input validation:

```bash
python verify_search_backend.py
```

#### 2. Run Full Automated PyTest Suite

Run the complete 50-test automated test suite:

```bash
pytest -v
```

Or run individual Part 2 search test suites:

```bash
pytest tests/test_bm25.py -v            # BM25 lexical indexing & keyword retrieval
pytest tests/test_reranker.py -v        # BGE cross-encoder reranking & model configuration
pytest tests/test_hybrid_retriever.py -v# Dense + Lexical candidate merging & RRF fusion
pytest tests/test_search_service.py -v  # Search service orchestration & relevance thresholding
pytest tests/test_search_api.py -v      # FastAPI HTTP requests, validation, & response schemas
```

---

## 📁 Repository Folder Structure

```text
Wayfinder/
├── .env.example                # Template configuration file with Ingestion & Search settings
├── .gitignore                  # Git ignore rules for venv, data, and cache
├── requirements.txt            # Python dependencies (Docling, LangChain, ChromaDB, FastAPI, Uvicorn, Torch, Transformers)
├── README.md                   # Complete system documentation
├── run_search_api.py           # CLI launcher for FastAPI Search Backend
├── verify_search_backend.py    # Automated end-to-end verification script for Search Backend
├── create_sample_pdf.py        # Helper script to create test PDFs with folder hierarchy
├── watch_documents.py          # Monitored directory watcher (runs ONLY while Prefect server is active)
├── verify_ingestion.py         # End-to-end ingestion pipeline verification script
├── inspect_chroma.py           # ChromaDB interactive CLI inspection tool
├── clean_slate.py              # Clean slate reset utility
├── src/
│   ├── wayfinder_ingestion/    # Part 1: Document Ingestion Pipeline
│   │   ├── __init__.py
│   │   ├── config.py           # Unified Pydantic settings management (.env)
│   │   ├── logging_config.py   # Secure structured privacy-safe logging
│   │   ├── change_detector.py  # File state manifest comparison & category derivation
│   │   ├── ledger.py           # CSV audit ledger tracking uploads & vectorizations
│   │   ├── watcher.py          # Watchdog folder watcher with Prefect health monitoring
│   │   ├── pdf_parser.py       # Docling PDF to structured Markdown converter
│   │   ├── pii_masker.py       # PII redaction component (Regex patterns)
│   │   ├── chunker.py          # Recursive text chunking with overlap & metadata
│   │   ├── embedding_service.py# Ollama vector embedding client (reads EMBEDDING_MODEL)
│   │   ├── vector_store.py     # Persistent ChromaDB client manager
│   │   └── pipeline.py         # Prefect flows and CLI execution
│   └── wayfinder_search/       # Part 2: Search & Retrieval Backend
│       ├── __init__.py
│       ├── models.py           # Pydantic request and response schemas
│       ├── bm25.py             # Pure-Python BM25Okapi lexical retrieval index
│       ├── hybrid_retriever.py # Hybrid retriever (Dense + BM25 with RRF merge)
│       ├── reranker.py         # BGE cross-encoder reranker (BAAI/bge-reranker-v2-m3)
│       ├── service.py          # Core search orchestration & threshold filtering service
│       └── api.py              # FastAPI app with POST /api/search & health routes
└── tests/
    ├── __init__.py
    ├── conftest.py             # Pytest root path configuration
    ├── test_bm25.py            # Tests for BM25 lexical search and category filtering
    ├── test_reranker.py        # Tests for BGE cross-encoder reranking & settings
    ├── test_hybrid_retriever.py# Tests for hybrid candidate merging and RRF fusion
    ├── test_search_service.py  # Tests for search business logic & threshold handling
    ├── test_search_api.py      # Tests for FastAPI search endpoints and validation
    ├── test_change_detector.py # Tests for file state detection
    ├── test_metadata_extraction.py # Tests for folder category extraction
    ├── test_pii_masker.py      # Tests for PII regex masking
    ├── test_pdf_parser.py      # Tests for Docling PDF to Markdown formatting
    ├── test_chunker.py         # Tests for recursive text chunking and overlap
    ├── test_embedding_service.py # Tests for embedding model configuration
    ├── test_vector_store.py    # Tests for ChromaDB storage, query, and deletion
    ├── test_ledger.py          # Tests for CSV ledger lifecycle tracking
    └── test_watcher.py         # Tests for watcher Prefect server health checks
```

---

## 🔮 Roadmap & Future Phases

The Wayfinder Document Retrieval platform is being constructed in modular phases:

- **Part 1 — Document Ingestion Pipeline** *(Completed)*: PDF parsing via Docling, PII redaction, LangChain recursive chunking with overlap, Ollama vectorization, ChromaDB persistence, CSV audit ledger, and Watchdog directory watching.
- **Part 2 — Search & Retrieval Backend** *(Completed)*: Hybrid retrieval (Dense Vector + BM25), BGE Cross-Encoder reranking (`bge-reranker-v2-m3`), relevance threshold filtering, and FastAPI search endpoints (`POST /api/search`).
- **Part 3 — Frontend Portal & MCP Server** *(Upcoming)*:
  - Responsive web dashboard for frontline workers to search guidelines with snippet highlighting and category filters.
  - Model Context Protocol (MCP) server exposing search tools to AI agents using the decoupled `SearchService`.
- **Part 4 — RAG Synthesis & Monitoring** *(Upcoming)*:
  - Generative answering with explicit source citations.
  - Latency monitoring and frontline query analytics dashboard.

---

## 🔒 Security & Logging

- **PII Masking**: Redacts sensitive customer and employee data before generating embeddings or persisting vectors in ChromaDB.
- **Privacy-Safe Logging**: Logs only record request IDs, chunk counts, execution latencies, and sanitized queries, **never** logging raw sensitive values or credentials.
- **Auditable Ledger**: Maintains an accurate, timestamped CSV ledger of all ingested and vectorized documents for frontline traceability.

