import re
from typing import Tuple, Dict
from wayfinder_ingestion.logging_config import setup_logger

logger = setup_logger(__name__)


class PIIMasker:
    """
    Normal Python component responsible for redacting sensitive PII from document text
    prior to vector embedding and storage in ChromaDB.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

        # Regex patterns for common PII categories
        self.patterns = {
            "EMAIL": (
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                "[EMAIL_REDACTED]"
            ),
            "PHONE": (
                r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                "[PHONE_REDACTED]"
            ),
            "SSN": (
                r'\b\d{3}-\d{2}-\d{4}\b',
                "[SSN_REDACTED]"
            ),
            "CREDIT_CARD": (
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                "[CREDIT_CARD_REDACTED]"
            ),
            "IPV4": (
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                "[IP_ADDRESS_REDACTED]"
            )
        }

    def mask(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Masks PII in the given text if masking is enabled.

        Returns:
            Tuple of (masked_text, redaction_counts_dict)
        """
        if not self.enabled or not text:
            return text, {}

        masked_text = text
        stats: Dict[str, int] = {}

        for pii_type, (pattern, replacement) in self.patterns.items():
            matches = re.findall(pattern, masked_text)
            count = len(matches)
            if count > 0:
                stats[pii_type] = count
                masked_text = re.sub(pattern, replacement, masked_text)

        if stats:
            logger.debug(f"Redacted PII from text block: {stats}")

        return masked_text, stats
