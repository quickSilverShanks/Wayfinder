from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from wayfinder_ingestion.pdf_parser import ParsedDocument
from wayfinder_ingestion.change_detector import DocumentMetadataInfo
from wayfinder_ingestion.logging_config import setup_logger

logger = setup_logger(__name__)

# LangChain RecursiveCharacterTextSplitter integration with graceful fallback
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
    LANGCHAIN_SPLITTER_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
        LANGCHAIN_SPLITTER_AVAILABLE = True
    except ImportError:
        LANGCHAIN_SPLITTER_AVAILABLE = False


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """
    Splits ParsedDocument Markdown pages into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter while preserving structural and folder metadata
    (category, sub_category, doc_title, page_number, file_path, file_hash) before
    storing vectors in ChromaDB.
    """

    # Standard Markdown separators for recursive splitting with hierarchy awareness
    MARKDOWN_SEPARATORS = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n\n",
        "\n- ",
        "\n* ",
        "\n",
        " ",
        ""
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"CHUNK_OVERLAP ({chunk_overlap}) must be less than CHUNK_SIZE ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = self._init_langchain_splitter()

    def _init_langchain_splitter(self) -> Optional[Any]:
        """Initializes LangChain's RecursiveCharacterTextSplitter with Markdown awareness."""
        if not LANGCHAIN_SPLITTER_AVAILABLE:
            logger.debug("LangChain text splitter not installed. Using native recursive fallback.")
            return None

        try:
            # First attempt: Language.MARKDOWN specialized splitter
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.debug(f"Initialized LangChain RecursiveCharacterTextSplitter for Markdown (size={self.chunk_size}, overlap={self.chunk_overlap}).")
            return splitter
        except Exception:
            try:
                # Second attempt: custom markdown separators list
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.MARKDOWN_SEPARATORS
                )
                logger.debug("Initialized LangChain RecursiveCharacterTextSplitter with custom Markdown separators.")
                return splitter
            except Exception as e:
                logger.warning(f"Could not initialize LangChain RecursiveCharacterTextSplitter: {e}")
                return None

    def _fallback_recursive_split(self, text: str) -> List[str]:
        """
        Native recursive chunking fallback preserving boundary hierarchy and overlap
        when LangChain is not installed in the environment.
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)

            if end < text_length:
                # Seek best markdown boundary near the end window
                best_cut = -1
                for sep in self.MARKDOWN_SEPARATORS:
                    cut = text.rfind(sep, start + (self.chunk_size // 2), end)
                    if cut != -1:
                        best_cut = cut + len(sep)
                        break

                if best_cut != -1:
                    end = best_cut

            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append(chunk_str)

            step = max(1, (end - start) - self.chunk_overlap)
            if start + step >= text_length:
                break
            start += step

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Splits Markdown text using LangChain's RecursiveCharacterTextSplitter with overlap.
        """
        if not text or not text.strip():
            return []

        # 1. Primary: LangChain RecursiveCharacterTextSplitter
        if self.text_splitter is not None:
            try:
                split_chunks = self.text_splitter.split_text(text)
                return [c.strip() for c in split_chunks if c.strip()]
            except Exception as e:
                logger.warning(f"LangChain recursive splitting failed: {e}. Falling back to native splitter.")

        # 2. Fallback: recursive splitting
        return self._fallback_recursive_split(text)

    def chunk_document(
        self,
        doc: ParsedDocument,
        doc_info: DocumentMetadataInfo
    ) -> List[Chunk]:
        """
        Recursively chunks all Markdown pages in ParsedDocument and constructs
        complete metadata for ChromaDB vector indexing.
        
        Preserves:
        - file_path: Source document relative path
        - file_hash: SHA-256 hash for versioning & idempotency
        - file_name: Base PDF filename
        - doc_title: Extracted document title
        - category: Folder category (e.g. 'HR')
        - sub_category: Folder sub-category (e.g. 'Policies')
        - page_number: Specific page number where chunk originated
        - chunk_index: Sequential chunk counter for this document
        - char_count: Length of the chunk in characters
        """
        all_chunks: List[Chunk] = []
        chunk_counter = 0

        # Chunk per-page so page_number metadata is preserved accurately
        for page in doc.pages:
            page_text_chunks = self._split_text(page.text)

            for idx, text_chunk in enumerate(page_text_chunks):
                if not text_chunk.strip():
                    continue

                chunk_counter += 1
                # Deterministic chunk ID for ChromaDB idempotency
                chunk_id = f"{doc_info.file_hash}_p{page.page_number}_c{chunk_counter}"

                metadata = {
                    "file_path": doc_info.relative_path,
                    "file_hash": doc_info.file_hash,
                    "file_name": doc_info.file_name,
                    "doc_title": doc.doc_title,
                    "category": doc_info.category,
                    "sub_category": doc_info.sub_category,
                    "page_number": page.page_number,
                    "chunk_index": chunk_counter,
                    "char_count": len(text_chunk)
                }

                all_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=text_chunk,
                    metadata=metadata
                ))

        logger.debug(f"Created {len(all_chunks)} recursive chunks with overlap for document: {doc_info.relative_path}")
        return all_chunks
