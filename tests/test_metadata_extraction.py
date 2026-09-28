import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.change_detector import ChangeDetector


def test_derive_categories_depth():
    root = Path("/data/docs")

    # Case 1: Root level document -> General, General
    cat, sub_cat = ChangeDetector.derive_categories(root, root / "policy.pdf")
    assert cat == "General"
    assert sub_cat == "General"

    # Case 2: One folder level -> HR, General
    cat, sub_cat = ChangeDetector.derive_categories(root, root / "HR" / "benefits.pdf")
    assert cat == "HR"
    assert sub_cat == "General"

    # Case 3: Two folder levels -> HR, Policies
    cat, sub_cat = ChangeDetector.derive_categories(root, root / "HR" / "Policies" / "leave.pdf")
    assert cat == "HR"
    assert sub_cat == "Policies"

    # Case 4: Deep subfolders -> CustomerSupport, Tier1/Refunds
    cat, sub_cat = ChangeDetector.derive_categories(root, root / "CustomerSupport" / "Tier1" / "Refunds" / "guide.pdf")
    assert cat == "CustomerSupport"
    assert sub_cat == "Tier1/Refunds"
