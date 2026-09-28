import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

from wayfinder_ingestion.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class DocumentMetadataInfo:
    relative_path: str
    absolute_path: str
    file_hash: str
    category: str
    sub_category: str
    file_name: str
    file_size_bytes: int = 0
    file_uploaded_at: str = ""


@dataclass
class ChangeSet:
    new_files: List[DocumentMetadataInfo] = field(default_factory=list)
    modified_files: List[DocumentMetadataInfo] = field(default_factory=list)
    unchanged_files: List[DocumentMetadataInfo] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)  # list of relative paths

    @property
    def has_changes(self) -> bool:
        return bool(self.new_files or self.modified_files or self.deleted_files)


class ChangeDetector:
    """
    Scans the source directory for PDF files, compares SHA-256 hashes against an
    ingestion manifest file and CSV ledger, and categorizes files into new, modified,
    unchanged, or deleted. Also extracts category and sub_category metadata from folder hierarchy.
    """

    def __init__(self, source_dir: str, manifest_dir: str, csv_path: Optional[str] = None):
        self.source_dir = Path(source_dir).resolve()
        self.manifest_dir = Path(manifest_dir).resolve()
        self.manifest_file = self.manifest_dir / "ingestion_manifest.json"
        self.csv_path = Path(csv_path).resolve() if csv_path else None
        self._manifest_data: Dict[str, Dict] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Loads manifest file from disk if present, and synchronizes with CSV ledger if available."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    self._manifest_data = json.load(f)
                logger.debug(f"Loaded ingestion manifest with {len(self._manifest_data)} entries.")
            except Exception as e:
                logger.error(f"Failed to load manifest at {self.manifest_file}: {e}. Starting fresh.")
                self._manifest_data = {}
        else:
            self._manifest_data = {}

        # Synchronize / backfill with CSV ledger to ensure already imported files are never re-imported
        csv_target = self.csv_path
        if not csv_target:
            try:
                from wayfinder_ingestion.config import get_settings
                csv_target = Path(get_settings().INGESTION_LOG_CSV).resolve()
            except Exception:
                csv_target = None

        if csv_target and csv_target.exists():
            try:
                with open(csv_target, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rel_path = row.get("relative_path")
                        status = row.get("status")
                        file_hash = row.get("file_hash")
                        if rel_path and status in ("active", "INGESTED") and file_hash:
                            if rel_path not in self._manifest_data:
                                self._manifest_data[rel_path] = {
                                    "file_hash": file_hash,
                                    "category": row.get("category", "General"),
                                    "sub_category": row.get("sub_category", "General"),
                                    "file_name": row.get("file_name", Path(rel_path).name),
                                    "file_size_bytes": int(row.get("file_size_bytes") or 0),
                                    "file_uploaded_at": row.get("file_uploaded_at", ""),
                                    "chunk_ids": []
                                }
            except Exception as e:
                logger.debug(f"Could not load ledger sync in change detector: {e}")

    def save_manifest(self) -> None:
        """Persists current manifest state to disk."""
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(self._manifest_data, f, indent=2)
            logger.debug(f"Saved ingestion manifest with {len(self._manifest_data)} entries.")
        except Exception as e:
            logger.error(f"Failed to save manifest to {self.manifest_file}: {e}")

    @staticmethod
    def calculate_sha256(file_path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def derive_categories(source_root: Path, file_path: Path) -> Tuple[str, str]:
        """
        Derives category and sub_category from folder hierarchy relative to source_root.
        Examples:
            source_root/HR/Policies/leave.pdf -> category='HR', sub_category='Policies'
            source_root/Finance/tax.pdf -> category='Finance', sub_category='General'
            source_root/doc.pdf -> category='General', sub_category='General'
        """
        try:
            rel_path = file_path.relative_to(source_root)
            parts = rel_path.parts[:-1]  # Exclude file name

            if not parts:
                return "General", "General"
            elif len(parts) == 1:
                return parts[0], "General"
            else:
                category = parts[0]
                sub_category = "/".join(parts[1:])
                return category, sub_category
        except ValueError:
            return "General", "General"

    def scan_source_directory(self) -> Dict[str, DocumentMetadataInfo]:
        """
        Scans source directory for all .pdf files.
        Returns dict mapping relative_path -> DocumentMetadataInfo.
        """
        current_files: Dict[str, DocumentMetadataInfo] = {}

        if not self.source_dir.exists():
            logger.warning(f"Source document directory does not exist: {self.source_dir}")
            return current_files

        for pdf_path in self.source_dir.glob("**/*.pdf"):
            if not pdf_path.is_file():
                continue

            try:
                rel_path = str(pdf_path.relative_to(self.source_dir)).replace("\\", "/")
                file_hash = self.calculate_sha256(pdf_path)
                category, sub_category = self.derive_categories(self.source_dir, pdf_path)

                stat = pdf_path.stat()
                file_size_bytes = stat.st_size
                file_uploaded_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

                doc_info = DocumentMetadataInfo(
                    relative_path=rel_path,
                    absolute_path=str(pdf_path.resolve()).replace("\\", "/"),
                    file_hash=file_hash,
                    category=category,
                    sub_category=sub_category,
                    file_name=pdf_path.name,
                    file_size_bytes=file_size_bytes,
                    file_uploaded_at=file_uploaded_at
                )
                current_files[rel_path] = doc_info
            except Exception as e:
                logger.error(f"Error reading file info for {pdf_path}: {e}")

        return current_files

    def detect_changes(self) -> ChangeSet:
        """
        Compares current source files on disk against the saved manifest.
        Returns a ChangeSet with new, modified, unchanged, and deleted PDF files.
        """
        current_files = self.scan_source_directory()
        change_set = ChangeSet()

        current_rel_paths = set(current_files.keys())
        manifest_rel_paths = set(self._manifest_data.keys())

        # Detect deleted files
        for rel_path in manifest_rel_paths - current_rel_paths:
            change_set.deleted_files.append(rel_path)

        # Detect new, modified, unchanged
        for rel_path, doc_info in current_files.items():
            if rel_path not in self._manifest_data:
                change_set.new_files.append(doc_info)
            else:
                stored_hash = self._manifest_data[rel_path].get("file_hash")
                if stored_hash != doc_info.file_hash:
                    change_set.modified_files.append(doc_info)
                else:
                    change_set.unchanged_files.append(doc_info)

        logger.info(
            f"Change Detection Summary: "
            f"New={len(change_set.new_files)}, "
            f"Modified={len(change_set.modified_files)}, "
            f"Unchanged={len(change_set.unchanged_files)}, "
            f"Deleted={len(change_set.deleted_files)}"
        )
        return change_set

    def record_ingested(self, doc_info: DocumentMetadataInfo, chunk_ids: List[str]) -> None:
        """Records a successfully ingested document into the manifest."""
        self._manifest_data[doc_info.relative_path] = {
            "file_hash": doc_info.file_hash,
            "category": doc_info.category,
            "sub_category": doc_info.sub_category,
            "file_name": doc_info.file_name,
            "file_size_bytes": doc_info.file_size_bytes,
            "file_uploaded_at": doc_info.file_uploaded_at,
            "chunk_ids": chunk_ids
        }
        self.save_manifest()

    def record_deleted(self, rel_path: str) -> Optional[List[str]]:
        """Removes a deleted document from manifest and returns its chunk_ids if present."""
        if rel_path in self._manifest_data:
            entry = self._manifest_data.pop(rel_path)
            self.save_manifest()
            return entry.get("chunk_ids", [])
        return []

    def get_manifest_entry(self, rel_path: str) -> Optional[Dict]:
        return self._manifest_data.get(rel_path)
