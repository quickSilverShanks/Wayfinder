"""
BGE Cross-Encoder Reranker Service.
Scores and ranks retrieved candidate document chunks using a configurable cross-encoder model.
"""

from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from wayfinder_ingestion.config import get_settings
from wayfinder_ingestion.logging_config import setup_logger

logger = setup_logger(__name__)


class BGEReranker:
    """
    Reranks candidate document chunks against a user query using a cross-encoder model.
    The model identifier is loaded strictly from settings (.env: RERANKER_MODEL)
    and is never hard-coded in logic.
    """

    _instance: Optional["BGEReranker"] = None

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        settings = get_settings()
        # Strictly load model name from settings (.env)
        self.model_name = (model_name or settings.RERANKER_MODEL).strip()
        configured_device = (device or settings.RERANKER_DEVICE).strip().lower()

        if configured_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif configured_device in ("cuda", "cpu"):
            self.device = configured_device if (configured_device != "cuda" or torch.cuda.is_available()) else "cpu"
        else:
            self.device = "cpu"

        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        logger.info(
            f"Initialized BGEReranker configured with model='{self.model_name}' on device='{self.device}'"
        )

    def _ensure_loaded(self) -> None:
        """
        Lazily loads tokenizer and sequence classification weights into memory/device.
        """
        if self._is_loaded:
            return

        logger.info(f"Loading reranker model '{self.model_name}' onto {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info(f"Successfully loaded reranker model '{self.model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load reranker model '{self.model_name}': {e}")
            raise RuntimeError(f"Could not load reranker model '{self.model_name}': {e}") from e

    def compute_scores(self, query: str, texts: List[str], batch_size: int = 16) -> List[float]:
        """
        Computes relevance scores for a query paired with a list of chunk texts.
        Returns a list of float scores in range [0.0, 1.0] using sigmoid normalization.
        """
        if not texts:
            return []

        self._ensure_loaded()

        all_scores: List[float] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            pairs = [[query, text] for text in batch_texts]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs.logits.view(-1).float()
                # Apply sigmoid normalization to yield intuitive [0.0, 1.0] relevance scores
                probabilities = torch.sigmoid(logits).cpu().tolist()
                if isinstance(probabilities, float):
                    probabilities = [probabilities]
                all_scores.extend(probabilities)

        return all_scores

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunk dictionaries.
        Each dictionary must contain 'text' and 'chunk_id'.
        Attaches 'relevance_score' to each candidate and returns them sorted descending by score.
        """
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]
        scores = self.compute_scores(query, texts)

        reranked: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            # Create a shallow copy with relevance_score attached
            item = dict(candidate)
            item["relevance_score"] = round(float(score), 4)
            reranked.append(item)

        # Sort strictly by reranker score descending
        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)

        if top_k is not None and top_k > 0:
            return reranked[:top_k]

        return reranked


def get_reranker() -> BGEReranker:
    """Singleton getter for BGEReranker."""
    if BGEReranker._instance is None:
        BGEReranker._instance = BGEReranker()
    return BGEReranker._instance
