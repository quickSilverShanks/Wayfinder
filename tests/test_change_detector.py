import sys
from pathlib import Path
import pytest

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.change_detector import ChangeDetector, DocumentMetadataInfo


def test_detect_new_and_unchanged_files(tmp_path):
    source_dir = tmp_path / "source"
    manifest_dir = tmp_path / "manifest"
    source_dir.mkdir()
    manifest_dir.mkdir()

    # Create dummy pdf
    pdf_file = source_dir / "HR" / "leave.pdf"
    pdf_file.parent.mkdir(parents=True)
    pdf_file.write_bytes(b"%PDF-1.4 dummy pdf content for testing")

    detector = ChangeDetector(source_dir=str(source_dir), manifest_dir=str(manifest_dir))
    change_set = detector.detect_changes()

    assert len(change_set.new_files) == 1
    assert change_set.new_files[0].relative_path == "HR/leave.pdf"
    assert change_set.new_files[0].category == "HR"
    assert change_set.new_files[0].sub_category == "General"

    # Record ingested
    doc_info = change_set.new_files[0]
    detector.record_ingested(doc_info, ["hash1_p1_c1"])

    # Re-run detection -> should be unchanged
    detector2 = ChangeDetector(source_dir=str(source_dir), manifest_dir=str(manifest_dir))
    change_set2 = detector2.detect_changes()

    assert len(change_set2.new_files) == 0
    assert len(change_set2.modified_files) == 0
    assert len(change_set2.unchanged_files) == 1
    assert len(change_set2.deleted_files) == 0


def test_detect_modified_file(tmp_path):
    source_dir = tmp_path / "source"
    manifest_dir = tmp_path / "manifest"
    source_dir.mkdir()
    manifest_dir.mkdir()

    pdf_file = source_dir / "doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 initial content")

    detector = ChangeDetector(source_dir=str(source_dir), manifest_dir=str(manifest_dir))
    change_set = detector.detect_changes()
    doc_info = change_set.new_files[0]
    detector.record_ingested(doc_info, ["chunk_1"])

    # Modify file content
    pdf_file.write_bytes(b"%PDF-1.4 updated content modified")

    detector2 = ChangeDetector(source_dir=str(source_dir), manifest_dir=str(manifest_dir))
    change_set2 = detector2.detect_changes()

    assert len(change_set2.new_files) == 0
    assert len(change_set2.modified_files) == 1
    assert change_set2.modified_files[0].relative_path == "doc.pdf"


def test_detect_deleted_file(tmp_path):
    source_dir = tmp_path / "source"
    manifest_dir = tmp_path / "manifest"
    source_dir.mkdir()
    manifest_dir.mkdir()

    pdf_file = source_dir / "temp.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 temp content")

    detector = ChangeDetector(source_dir=str(source_dir), manifest_dir=str(manifest_dir))
    change_set = detector.detect_changes()
    detector.record_ingested(change_set.new_files[0], ["chunk_temp"])

    # Delete file from disk
    pdf_file.unlink()

    detector2 = ChangeDetector(source_dir=str(source_dir), manifest_dir=str(manifest_dir))
    change_set2 = detector2.detect_changes()

    assert len(change_set2.deleted_files) == 1
    assert change_set2.deleted_files[0] == "temp.pdf"


def test_detect_already_imported_via_csv_ledger(tmp_path):
    source_dir = tmp_path / "source"
    manifest_dir = tmp_path / "manifest"
    csv_file = tmp_path / "ingested_documents.csv"
    source_dir.mkdir()
    manifest_dir.mkdir()

    pdf_file = source_dir / "HR" / "policy.pdf"
    pdf_file.parent.mkdir(parents=True)
    pdf_file.write_bytes(b"%PDF-1.4 dummy policy content")

    # Pre-populate CSV ledger with this file as active
    from wayfinder_ingestion.ledger import IngestionLedger
    ledger = IngestionLedger(csv_path=str(csv_file))
    doc_hash = ChangeDetector.calculate_sha256(pdf_file)
    doc_info = DocumentMetadataInfo(
        relative_path="HR/policy.pdf",
        absolute_path=str(pdf_file),
        file_hash=doc_hash,
        category="HR",
        sub_category="General",
        file_name="policy.pdf",
        file_size_bytes=len(pdf_file.read_bytes()),
        file_uploaded_at="2026-01-01T00:00:00Z"
    )
    ledger.record_success(doc_info, chunk_count=3)

    # Initialize ChangeDetector with empty manifest dir but referencing csv_file
    detector = ChangeDetector(
        source_dir=str(source_dir),
        manifest_dir=str(manifest_dir),
        csv_path=str(csv_file)
    )
    change_set = detector.detect_changes()

    # The file should be recognized as already imported / unchanged!
    assert len(change_set.new_files) == 0
    assert len(change_set.unchanged_files) == 1
    assert change_set.unchanged_files[0].relative_path == "HR/policy.pdf"

