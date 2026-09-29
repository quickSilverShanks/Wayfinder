"""
Pure-Python BM25Okapi Lexical Search Engine.
Provides robust lexical keyword matching over document chunks without external C-extensions.
"""

import math
import re
from typing import Dict, List, Tuple, Any, Optional
from wayfinder_ingestion.logging_config import setup_logger

logger = setup_logger(__name__)


def default_tokenize(text: str) -> List[str]:
    """
    Standard regex-based alphanumeric tokenization with lowercasing.
    Handles alphanumeric tokens, hyphens, and common punctuation.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """
    In-memory BM25Okapi index over document chunks.
    Calculates relevance scores based on term frequency (TF), inverted document frequency (IDF),
    and document length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.doc_records: List[Dict[str, Any]] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing term
        self.idf: Dict[str, float] = {}       # term -> IDF weight
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, term_freq)]

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of document chunk records.
        Each document record must have at least 'chunk_id', 'text', and 'metadata'.
        """
        self.doc_records = documents
        self.corpus_size = len(documents)
        self.doc_lens = []
        self.doc_freqs = {}
        self.inverted_index = {}
        self.idf = {}

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        total_length = 0

        for doc_idx, doc in enumerate(documents):
            text = doc.get("text", "")
            tokens = default_tokenize(text)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_length += doc_len

            # Calculate term frequencies for this document
            tf_map: Dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for token, freq in tf_map.items():
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((doc_idx, freq))

        self.avg_doc_len = total_length / self.corpus_size

        # Precompute Robertson-Spärck Jones IDF for all terms
        for token, df in self.doc_freqs.items():
            # BM25Okapi IDF with smoothing (+0.5) to guarantee non-negative scores
            self.idf[token] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

        logger.debug(
            f"Built BM25 index over {self.corpus_size} document chunks "
            f"(unique terms: {len(self.doc_freqs)}, avg_len: {self.avg_doc_len:.1f})."
        )

    def search(
        self,
        query: str,
        top_k: int = 15,
        category: Optional[str] = None,
        sub_category: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes BM25 lexical scoring for the query.
        Optionally filters candidate records by category and sub_category.
        Returns list of (doc_record, score) sorted descending by score.
        """
        query_tokens = default_tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        doc_scores: Dict[int, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf.get(token, 0.0)
            postings = self.inverted_index[token]

            for doc_idx, tf in postings:
                # Apply metadata filters if specified
                if category or sub_category:
                    meta = self.doc_records[doc_idx].get("metadata", {})
                    if category and meta.get("category") != category:
                        continue
                    if sub_category and meta.get("sub_category") != sub_category:
                        continue

                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                term_score = idf * (numerator / denominator)

                doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + term_score

        if not doc_scores:
            return []

        sorted_scored = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.doc_records[idx], score) for idx, score in sorted_scored]
