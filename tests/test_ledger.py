import sys
import csv
from pathlib import Path
import pytest

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.ledger import IngestionLedger
from wayfinder_ingestion.change_detector import DocumentMetadataInfo


def test_ledger_creation_and_headers(tmp_path):
    csv_file = tmp_path / "ingested_test.csv"
    ledger = IngestionLedger(str(csv_file))

    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert "relative_path" in headers
        assert "file_location" in headers
        assert "file_uploaded_at" in headers
        assert "vectorized_at" in headers
        assert "file_deleted_at" in headers
        assert "chunk_count" in headers
        assert "status" in headers


def test_ledger_record_success_active(tmp_path):
    csv_file = tmp_path / "ingested_test.csv"
    ledger = IngestionLedger(str(csv_file))

    doc = DocumentMetadataInfo(
        relative_path="HR/Policies/leave.pdf",
        absolute_path=str(tmp_path / "data/HR/Policies/leave.pdf"),
        file_hash="hash_v1_123456",
        category="HR",
        sub_category="Policies",
        file_name="leave.pdf",
        file_size_bytes=1024,
        file_uploaded_at="2026-09-28T10:00:00+00:00"
    )

    vec_time = "2026-09-28T10:05:00+00:00"
    ledger.record_success(doc, chunk_count=3, vectorized_at=vec_time)

    records = ledger.get_active_records()
    assert len(records) == 1
    assert records[0]["relative_path"] == "HR/Policies/leave.pdf"
    assert records[0]["file_uploaded_at"] == "2026-09-28T10:00:00+00:00"
    assert records[0]["vectorized_at"] == vec_time
    assert records[0]["file_deleted_at"] == ""
    assert records[0]["status"] == "active"


def test_ledger_record_deletion_has_delete_time(tmp_path):
    csv_file = tmp_path / "ingested_test.csv"
    ledger = IngestionLedger(str(csv_file))

    doc = DocumentMetadataInfo(
        relative_path="IT/Security/wifi.pdf",
        absolute_path=str(tmp_path / "data/IT/Security/wifi.pdf"),
        file_hash="hash_wifi",
        category="IT",
        sub_category="Security",
        file_name="wifi.pdf",
        file_size_bytes=512,
        file_uploaded_at="2026-09-28T09:00:00+00:00"
    )

    ledger.record_success(doc, chunk_count=2, vectorized_at="2026-09-28T09:05:00+00:00")
    assert len(ledger.get_active_records()) == 1

    del_time = "2026-09-28T09:30:00+00:00"
    del_success = ledger.record_deletion("IT/Security/wifi.pdf", deleted_at=del_time)
    assert del_success is True

    # Active records should be 0
    assert len(ledger.get_active_records()) == 0

    # Total audit records should still include it as deleted
    all_recs = ledger.get_all_records()
    assert len(all_recs) == 1
    assert all_recs[0]["status"] == "deleted"
    assert all_recs[0]["file_uploaded_at"] == "2026-09-28T09:00:00+00:00"
    assert all_recs[0]["vectorized_at"] == "2026-09-28T09:05:00+00:00"
    assert all_recs[0]["file_deleted_at"] == del_time


def test_delete_and_reupload_creates_both_entries(tmp_path):
    """
    Validates user requirement:
    If a file named 'xyz' is deleted and then reuploaded in the same location,
    BOTH entries exist in the CSV:
    - Deleted record has its original upload time, status='deleted', and delete time.
    - Reuploaded file has a new record with status='active' and its new upload time.
    """
    csv_file = tmp_path / "ingested_test.csv"
    ledger = IngestionLedger(str(csv_file))

    # 1. First upload of xyz.pdf
    doc_v1 = DocumentMetadataInfo(
        relative_path="HR/xyz.pdf",
        absolute_path=str(tmp_path / "data/HR/xyz.pdf"),
        file_hash="hash_v1",
        category="HR",
        sub_category="General",
        file_name="xyz.pdf",
        file_size_bytes=1000,
        file_uploaded_at="2026-09-28T10:00:00+00:00"
    )
    ledger.record_success(doc_v1, chunk_count=2, vectorized_at="2026-09-28T10:02:00+00:00")

    # Verify initial active record
    recs_v1 = ledger.get_all_records()
    assert len(recs_v1) == 1
    assert recs_v1[0]["status"] == "active"
    assert recs_v1[0]["file_uploaded_at"] == "2026-09-28T10:00:00+00:00"
    assert recs_v1[0]["file_deleted_at"] == ""

    # 2. Delete xyz.pdf
    del_time = "2026-09-28T10:30:00+00:00"
    ledger.record_deletion("HR/xyz.pdf", deleted_at=del_time)

    recs_after_delete = ledger.get_all_records()
    assert len(recs_after_delete) == 1
    assert recs_after_delete[0]["status"] == "deleted"
    assert recs_after_delete[0]["file_uploaded_at"] == "2026-09-28T10:00:00+00:00"
    assert recs_after_delete[0]["file_deleted_at"] == del_time

    # 3. Re-upload xyz.pdf at the exact same location with new content/time
    doc_v2 = DocumentMetadataInfo(
        relative_path="HR/xyz.pdf",
        absolute_path=str(tmp_path / "data/HR/xyz.pdf"),
        file_hash="hash_v2_new_content",
        category="HR",
        sub_category="General",
        file_name="xyz.pdf",
        file_size_bytes=1500,
        file_uploaded_at="2026-09-28T11:00:00+00:00"
    )
    ledger.record_success(doc_v2, chunk_count=3, vectorized_at="2026-09-28T11:02:00+00:00")

    # 4. Verify BOTH entries are in the CSV!
    all_recs = ledger.get_all_records()
    assert len(all_recs) == 2, f"Expected 2 records in CSV, but found {len(all_recs)}"

    # Record 1 (historical deleted file)
    rec1 = all_recs[0]
    assert rec1["relative_path"] == "HR/xyz.pdf"
    assert rec1["status"] == "deleted"
    assert rec1["file_uploaded_at"] == "2026-09-28T10:00:00+00:00"
    assert rec1["vectorized_at"] == "2026-09-28T10:02:00+00:00"
    assert rec1["file_deleted_at"] == del_time
    assert rec1["file_hash"] == "hash_v1"

    # Record 2 (newly reuploaded active file)
    rec2 = all_recs[1]
    assert rec2["relative_path"] == "HR/xyz.pdf"
    assert rec2["status"] == "active"
    assert rec2["file_uploaded_at"] == "2026-09-28T11:00:00+00:00"
    assert rec2["vectorized_at"] == "2026-09-28T11:02:00+00:00"
    assert rec2["file_deleted_at"] == ""
    assert rec2["file_hash"] == "hash_v2_new_content"

    # Verify helpers
    active_recs = ledger.get_active_records()
    assert len(active_recs) == 1
    assert active_recs[0]["file_uploaded_at"] == "2026-09-28T11:00:00+00:00"

    history = ledger.get_records_for_path("HR/xyz.pdf")
    assert len(history) == 2
