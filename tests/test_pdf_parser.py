import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.pdf_parser import PDFParser, ParsedDocument, ParsedPage


def test_format_text_as_markdown():
    raw_text = (
        "1. Overview\n"
        "This is paragraph one explaining company policies.\n"
        "• Bullet point item one\n"
        "• Bullet point item two\n\n"
        "2. Contact\n"
        "Reach out to HR team."
    )

    md = PDFParser._format_text_as_markdown(raw_text, page_number=1, doc_title="Employee Handbook")

    assert "# Employee Handbook" in md
    assert "## Page 1" in md
    assert "- Bullet point item one" in md
    assert "- Bullet point item two" in md
    assert "1. Overview" in md


def test_format_empty_page():
    md = PDFParser._format_text_as_markdown("", page_number=3)
    assert "## Page 3" in md
    assert "*Empty page*" in md


def test_fallback_parse_plain_text(tmp_path):
    mock_pdf = tmp_path / "sample.pdf"
    mock_pdf.write_text("Company IT Security Policy\n• Use strong passwords\n• Enable MFA", encoding="utf-8")

    parser = PDFParser()
    doc = parser._fallback_parse(mock_pdf)

    assert isinstance(doc, ParsedDocument)
    assert len(doc.pages) >= 1
    assert "Sample" in doc.doc_title
    assert "## Page 1" in doc.pages[0].text


def test_docling_parser_flow(tmp_path):
    from unittest.mock import MagicMock

    mock_pdf = tmp_path / "docling_test.pdf"
    mock_pdf.write_bytes(b"%PDF-1.4 dummy")

    parser = PDFParser()
    mock_converter = MagicMock()
    mock_doc = MagicMock()
    mock_doc.name = "Custom Policy Title"
    mock_page_1 = MagicMock()
    mock_page_1.export_to_markdown.return_value = "Section 1: Leave entitlement details."
    mock_doc.pages = {1: mock_page_1}

    mock_result = MagicMock()
    mock_result.document = mock_doc
    mock_converter.convert.return_value = mock_result
    parser._docling_converter = mock_converter

    parsed = parser._parse_with_docling(mock_pdf)
    assert parsed is not None
    assert parsed.doc_title == "Custom Policy Title"
    assert len(parsed.pages) == 1
    assert "## Page 1" in parsed.pages[0].text
    assert "Leave entitlement details" in parsed.pages[0].text

