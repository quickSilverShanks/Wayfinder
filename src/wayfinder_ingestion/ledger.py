import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from wayfinder_ingestion.change_detector import DocumentMetadataInfo
from wayfinder_ingestion.logging_config import setup_logger

logger = setup_logger(__name__)


class IngestionLedger:
    """
    Maintains a CSV ledger tracking all PDF documents that have been ingested,
    vectorized, and stored in ChromaDB.
    
    Retains full audit history:
    - If a file is uploaded, a record is created with status "active", upload time, and vectorization time.
    - If a file is deleted, its existing record is updated with status "deleted" and its deletion time.
    - If a file with the same name is re-uploaded at the same location, a NEW record is added with its new
      upload time and status "active", preserving the historical "deleted" record.
    """

    FIELDNAMES = [
        "file_name",
        "relative_path",
        "file_location",
        "category",
        "sub_category",
        "file_hash",
        "file_size_bytes",
        "file_uploaded_at",
        "vectorized_at",
        "file_deleted_at",
        "chunk_count",
        "status",
        "last_updated_at"
    ]

    def __init__(self, csv_path: str = "./data/ingested_documents.csv"):
        self.csv_path = Path(csv_path).resolve()
        self._ensure_csv_file()

    def _ensure_csv_file(self) -> None:
        """Ensures parent directory and CSV header row exist."""
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                    writer.writeheader()
                logger.debug(f"Created new ingestion CSV ledger at: {self.csv_path}")
            else:
                # Check if existing CSV has all headers; if not, migrate
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                if "file_deleted_at" not in header:
                    self._migrate_headers()
        except Exception as e:
            logger.error(f"Failed to initialize CSV ledger at {self.csv_path}: {e}")

    def _migrate_headers(self) -> None:
        """Migrates CSV if columns are missing."""
        try:
            records = self._read_records()
            self._write_records(records)
            logger.info("Migrated CSV ledger headers to include file_deleted_at.")
        except Exception as e:
            logger.error(f"Failed migrating CSV ledger headers: {e}")

    def _read_records(self) -> List[Dict[str, str]]:
        """Reads all records from the CSV file as an ordered list."""
        records: List[Dict[str, str]] = []
        if not self.csv_path.exists():
            return records

        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("relative_path"):
                        records.append(row)
        except Exception as e:
            logger.error(f"Failed to read CSV ledger at {self.csv_path}: {e}")
        return records

    def _write_records(self, records: List[Dict[str, str]]) -> None:
        """Writes all records back to the CSV file safely using a temporary file."""
        try:
            temp_path = self.csv_path.with_suffix(".tmp")
            with open(temp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
                for row in records:
                    filtered_row = {k: row.get(k, "") for k in self.FIELDNAMES}
                    writer.writerow(filtered_row)

            # Atomic replace
            temp_path.replace(self.csv_path)
            logger.debug(f"Saved {len(records)} records to CSV ledger: {self.csv_path}")
        except Exception as e:
            logger.error(f"Failed to write CSV ledger to {self.csv_path}: {e}")

    def record_success(
        self,
        doc_info: DocumentMetadataInfo,
        chunk_count: int,
        vectorized_at: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Records a successfully vectorized document in the CSV ledger.
        
        - If an 'active' record already exists for this relative_path, updates it in place.
        - If no 'active' record exists (new document OR previously deleted document being re-uploaded),
          appends a brand new record with status 'active' and its new upload/vectorized timestamps,
          preserving any previous 'deleted' history.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        vec_iso = vectorized_at or now_iso

        records = self._read_records()

        # Check if there is an existing 'active' record for this relative_path
        active_index = None
        for idx, rec in enumerate(records):
            if rec.get("relative_path") == doc_info.relative_path and rec.get("status") in ("active", "INGESTED"):
                active_index = idx
                break

        if active_index is not None:
            # Update existing active record in place (e.g. modified content)
            entry = records[active_index]
            entry["file_name"] = doc_info.file_name
            entry["file_location"] = doc_info.absolute_path.replace("\\", "/")
            entry["category"] = doc_info.category
            entry["sub_category"] = doc_info.sub_category
            entry["file_hash"] = doc_info.file_hash
            entry["file_size_bytes"] = str(doc_info.file_size_bytes)
            entry["file_uploaded_at"] = doc_info.file_uploaded_at or entry.get("file_uploaded_at") or now_iso
            entry["vectorized_at"] = vec_iso
            entry["file_deleted_at"] = ""
            entry["chunk_count"] = str(chunk_count)
            entry["status"] = "active"
            entry["last_updated_at"] = now_iso
            logger.info(f"Updated active document in CSV ledger: {doc_info.relative_path}")
        else:
            # New upload OR re-upload of a previously deleted file: append a NEW record
            entry = {
                "file_name": doc_info.file_name,
                "relative_path": doc_info.relative_path,
                "file_location": doc_info.absolute_path.replace("\\", "/"),
                "category": doc_info.category,
                "sub_category": doc_info.sub_category,
                "file_hash": doc_info.file_hash,
                "file_size_bytes": str(doc_info.file_size_bytes),
                "file_uploaded_at": doc_info.file_uploaded_at or now_iso,
                "vectorized_at": vec_iso,
                "file_deleted_at": "",
                "chunk_count": str(chunk_count),
                "status": "active",
                "last_updated_at": now_iso
            }
            records.append(entry)
            logger.info(
                f"Added new active document to CSV ledger: {doc_info.relative_path} "
                f"(Uploaded: {entry['file_uploaded_at']}, Vectorized: {entry['vectorized_at']}, Chunks: {chunk_count})"
            )

        self._write_records(records)
        return entry

    def record_deletion(
        self,
        relative_path: str,
        deleted_at: Optional[str] = None
    ) -> bool:
        """
        Marks an ingested document as 'deleted' in the CSV ledger when its source file
        and ChromaDB vectors are removed.
        
        Preserves original file_uploaded_at, sets file_deleted_at timestamp, and updates status to 'deleted'.
        """
        records = self._read_records()
        now_iso = deleted_at or datetime.now(timezone.utc).isoformat()
        found = False

        # Find the latest 'active' record for this path
        for rec in reversed(records):
            if rec.get("relative_path") == relative_path and rec.get("status") in ("active", "INGESTED"):
                rec["status"] = "deleted"
                rec["file_deleted_at"] = now_iso
                rec["last_updated_at"] = now_iso
                found = True
                break

        # Fallback if no active record was found: update the last matching entry
        if not found:
            for rec in reversed(records):
                if rec.get("relative_path") == relative_path:
                    rec["status"] = "deleted"
                    rec["file_deleted_at"] = now_iso
                    rec["last_updated_at"] = now_iso
                    found = True
                    break

        if found:
            self._write_records(records)
            logger.info(f"Marked document as 'deleted' in CSV ledger: {relative_path} (Deleted: {now_iso})")
            return True

        return False

    def get_active_records(self) -> List[Dict[str, str]]:
        """Returns all records currently marked as active."""
        records = self._read_records()
        return [r for r in records if r.get("status") in ("active", "INGESTED")]

    def get_ingested_records(self) -> List[Dict[str, str]]:
        """Alias for get_active_records for backwards compatibility."""
        return self.get_active_records()

    def get_deleted_records(self) -> List[Dict[str, str]]:
        """Returns all records marked as deleted."""
        records = self._read_records()
        return [r for r in records if r.get("status") in ("deleted", "DELETED")]

    def get_all_records(self) -> List[Dict[str, str]]:
        """Returns all records in chronological order, including active and deleted."""
        return self._read_records()

    def get_records_for_path(self, relative_path: str) -> List[Dict[str, str]]:
        """Returns all history records for a given relative_path (e.g. deleted and reuploaded)."""
        records = self._read_records()
        return [r for r in records if r.get("relative_path") == relative_path]

    def get_record(self, relative_path: str, status: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Returns the latest record for relative_path (optionally filtered by status).
        """
        records = self._read_records()
        for rec in reversed(records):
            if rec.get("relative_path") == relative_path:
                if status is None or rec.get("status") == status:
                    return rec
        return None
