import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.pii_masker import PIIMasker


def test_pii_masking_enabled():
    masker = PIIMasker(enabled=True)
    sample_text = (
        "Contact Alice at alice.smith@enterprise.com or call 555-123-4567. "
        "SSN is 123-45-6789. Card is 4532-1234-5678-9010. IP is 192.168.0.1."
    )

    masked, stats = masker.mask(sample_text)

    assert "alice.smith@enterprise.com" not in masked
    assert "[EMAIL_REDACTED]" in masked
    assert "555-123-4567" not in masked
    assert "[PHONE_REDACTED]" in masked
    assert "123-45-6789" not in masked
    assert "[SSN_REDACTED]" in masked
    assert "4532-1234-5678-9010" not in masked
    assert "[CREDIT_CARD_REDACTED]" in masked
    assert "192.168.0.1" not in masked
    assert "[IP_ADDRESS_REDACTED]" in masked

    assert stats.get("EMAIL") == 1
    assert stats.get("PHONE") == 1
    assert stats.get("SSN") == 1
    assert stats.get("CREDIT_CARD") == 1
    assert stats.get("IPV4") == 1


def test_pii_masking_disabled():
    masker = PIIMasker(enabled=False)
    sample_text = "Contact Alice at alice.smith@enterprise.com or call 555-123-4567."

    masked, stats = masker.mask(sample_text)

    assert masked == sample_text
    assert stats == {}
