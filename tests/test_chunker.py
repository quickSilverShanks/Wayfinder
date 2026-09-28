import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.chunker import DocumentChunker
from wayfinder_ingestion.pdf_parser import ParsedDocument, ParsedPage
from wayfinder_ingestion.change_detector import DocumentMetadataInfo


def test_chunking_and_metadata():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    doc = ParsedDocument(
        file_path="/tmp/test.pdf",
        doc_title="Test Policy Document",
        pages=[
            ParsedPage(page_number=1, text="This is sentence one of page one. This is sentence two. " * 5),
            ParsedPage(page_number=2, text="This is page two content for chunking test.")
        ]
    )

    doc_info = DocumentMetadataInfo(
        relative_path="HR/test.pdf",
        absolute_path="/tmp/test.pdf",
        file_hash="abcdef123456",
        category="HR",
        sub_category="General",
        file_name="test.pdf"
    )

    chunks = chunker.chunk_document(doc, doc_info)

    assert len(chunks) > 1
    # Check deterministic chunk ID format: {file_hash}_p{page_number}_c{chunk_counter}
    assert chunks[0].chunk_id == "abcdef123456_p1_c1"
    assert chunks[0].metadata["category"] == "HR"
    assert chunks[0].metadata["sub_category"] == "General"
    assert chunks[0].metadata["doc_title"] == "Test Policy Document"
    assert chunks[0].metadata["file_hash"] == "abcdef123456"
    assert chunks[0].metadata["file_path"] == "HR/test.pdf"
    assert chunks[0].metadata["file_name"] == "test.pdf"
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["chunk_index"] == 1
    assert "char_count" in chunks[0].metadata


def test_recursive_markdown_chunking_with_overlap():
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)

    markdown_text = (
        "# Leave Policy Overview\n\n"
        "## Annual Entitlement\n\n"
        "Employees receive 20 days annual leave per year. "
        "Leave requests should be booked 14 days in advance.\n\n"
        "## Sick Leave Protocol\n\n"
        "Emergency sick leave must be notified before 9:00 AM on the day of absence. "
        "Doctor notes are mandatory after three consecutive sick days."
    )

    doc = ParsedDocument(
        file_path="/tmp/leave.pdf",
        doc_title="Leave Policy",
        pages=[ParsedPage(page_number=1, text=markdown_text)]
    )

    doc_info = DocumentMetadataInfo(
        relative_path="HR/Policies/leave.pdf",
        absolute_path="/tmp/leave.pdf",
        file_hash="hash_leave_123",
        category="HR",
        sub_category="Policies",
        file_name="leave.pdf"
    )

    chunks = chunker.chunk_document(doc, doc_info)

    assert len(chunks) >= 2
    # Verify metadata on all chunks
    for idx, chunk in enumerate(chunks, start=1):
        assert chunk.metadata["category"] == "HR"
        assert chunk.metadata["sub_category"] == "Policies"
        assert chunk.metadata["doc_title"] == "Leave Policy"
        assert chunk.metadata["page_number"] == 1
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.chunk_id == f"hash_leave_123_p1_c{idx}"
        assert len(chunk.text) > 0


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=150)
