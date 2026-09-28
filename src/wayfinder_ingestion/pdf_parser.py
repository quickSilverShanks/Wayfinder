from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import re

from wayfinder_ingestion.logging_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class ParsedPage:
    page_number: int
    text: str  # Markdown-formatted text of the page


@dataclass
class ParsedDocument:
    file_path: str
    doc_title: str
    pages: List[ParsedPage] = field(default_factory=list)
    markdown_content: str = ""

    @property
    def full_text(self) -> str:
        if self.markdown_content:
            return self.markdown_content
        return "\n\n".join(page.text for page in self.pages)


class PDFParser:
    """
    Converts PDF documents into structured Markdown (MD) using Docling,
    preserving document titles, section headers, tables, and page numbers.
    
    Includes a graceful fallback parser if Docling is unavailable.
    """

    def __init__(self):
        self._docling_converter = None
        self._init_docling()

    def _init_docling(self):
        """Initializes Docling DocumentConverter."""
        try:
            from docling.document_converter import DocumentConverter
            self._docling_converter = DocumentConverter()
            logger.debug("Initialized Docling DocumentConverter successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize Docling DocumentConverter: {e}. Will use fallback parser if needed.")
            self._docling_converter = None

    @staticmethod
    def _format_text_as_markdown(raw_text: str, page_number: int, doc_title: Optional[str] = None) -> str:
        """
        Converts raw extracted page text into clean, structured Markdown (MD).
        Adds page headers, standardizes paragraph breaks, and formats bullet lists.
        """
        if not raw_text or not raw_text.strip():
            return f"## Page {page_number}\n\n*Empty page*"

        lines = [line.strip() for line in raw_text.splitlines()]
        md_lines: List[str] = [f"## Page {page_number}\n"]

        if page_number == 1 and doc_title:
            md_lines.insert(0, f"# {doc_title}\n")

        current_para: List[str] = []

        for line in lines:
            if not line:
                if current_para:
                    md_lines.append(" ".join(current_para))
                    current_para = []
                continue

            # Detect existing markdown headers or bullet items
            if re.match(r"^#{1,6}\s+", line) or re.match(r"^[-*•]\s+", line) or re.match(r"^\d+\.\s+", line):
                if current_para:
                    md_lines.append(" ".join(current_para))
                    current_para = []
                cleaned_line = re.sub(r"^[•]\s*", "- ", line)
                md_lines.append(cleaned_line)
            else:
                current_para.append(line)

        if current_para:
            md_lines.append(" ".join(current_para))

        return "\n\n".join(md_lines).strip()

    def _parse_with_docling(self, path: Path) -> Optional[ParsedDocument]:
        """
        Primary converter: Uses Docling to parse PDF and export to structured Markdown.
        """
        if self._docling_converter is None:
            return None

        try:
            logger.info(f"Converting PDF to Markdown using Docling: {path.name}")
            result = self._docling_converter.convert(str(path))
            doc = result.document

            doc_title = path.stem.replace("_", " ").replace("-", " ").title()
            if hasattr(doc, "name") and doc.name:
                doc_title = doc.name

            pages: List[ParsedPage] = []

            # Extract markdown content by pages if available in Docling document
            if hasattr(doc, "pages") and doc.pages:
                for page_no, page_obj in doc.pages.items():
                    page_text = ""
                    if hasattr(page_obj, "export_to_markdown"):
                        page_text = page_obj.export_to_markdown()
                    elif hasattr(page_obj, "text"):
                        page_text = page_obj.text

                    if page_text and page_text.strip():
                        page_md = f"## Page {page_no}\n\n" + page_text.strip()
                        if int(page_no) == 1:
                            page_md = f"# {doc_title}\n\n" + page_md
                        pages.append(ParsedPage(page_number=int(page_no), text=page_md))

            # If pages dictionary was not directly iterable, export full markdown and split
            if not pages:
                full_md = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""
                if full_md and full_md.strip():
                    page_chunks = full_md.split("<!-- pagebreak -->")
                    for idx, chunk in enumerate(page_chunks, start=1):
                        if chunk.strip():
                            page_md = f"## Page {idx}\n\n" + chunk.strip()
                            if idx == 1:
                                page_md = f"# {doc_title}\n\n" + page_md
                            pages.append(ParsedPage(page_number=idx, text=page_md))

            if pages:
                full_markdown = "\n\n---\n\n".join(p.text for p in pages)
                logger.info(f"Successfully converted PDF to Markdown via Docling: {path.name} ({len(pages)} pages)")
                return ParsedDocument(
                    file_path=str(path),
                    doc_title=doc_title,
                    pages=pages,
                    markdown_content=full_markdown
                )
        except Exception as e:
            logger.warning(f"Docling conversion encountered an issue for {path.name}: {e}. Falling back to standard parser.")

        return None

    def _fallback_parse(self, path: Path) -> ParsedDocument:
        """
        Fallback parser using pypdf, formatting extracted text into structured Markdown.
        """
        doc_title = path.stem.replace("_", " ").replace("-", " ").title()
        pages: List[ParsedPage] = []

        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            if reader.metadata and reader.metadata.title:
                doc_title = reader.metadata.title

            for idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                page_md = self._format_text_as_markdown(
                    raw_text=text,
                    page_number=idx,
                    doc_title=doc_title if idx == 1 else None
                )
                pages.append(ParsedPage(page_number=idx, text=page_md))
        except Exception as e:
            logger.error(f"Fallback PyPDF parsing failed for {path.name}: {e}")
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if content.strip():
                        page_md = self._format_text_as_markdown(content, page_number=1, doc_title=doc_title)
                        pages.append(ParsedPage(page_number=1, text=page_md))
            except Exception:
                pass

        if not pages:
            pages.append(ParsedPage(page_number=1, text=f"# {doc_title}\n\n## Page 1\n\n*Empty document*"))

        full_markdown = "\n\n---\n\n".join(p.text for p in pages)
        return ParsedDocument(
            file_path=str(path),
            doc_title=doc_title,
            pages=pages,
            markdown_content=full_markdown
        )

    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parses a PDF file and converts it to structured Markdown (MD) using Docling.
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at {path}")

        # 1. Primary: Docling PDF to Markdown conversion
        parsed_doc = self._parse_with_docling(path)
        if parsed_doc is not None and parsed_doc.pages:
            return parsed_doc

        # 2. Fallback: PyPDF converted to Markdown
        return self._fallback_parse(path)
