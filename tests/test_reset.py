import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from reset_ingestion_data import reset_chroma_db, reset_csv_ledger, cleanup_temp_files
from wayfinder_ingestion.ledger import IngestionLedger


def test_reset_chroma_db(tmp_path):
    chroma_dir = tmp_path / "chroma_db"
    chroma_dir.mkdir()
    (chroma_dir / "test_file.sqlite3").write_text("dummy")
    (chroma_dir / "ingestion_manifest.json").write_text("{}")

    reset_chroma_db(chroma_dir)

    assert chroma_dir.exists()
    assert list(chroma_dir.iterdir()) == []


def test_reset_csv_ledger(tmp_path):
    csv_file = tmp_path / "ingested_documents.csv"
    csv_file.write_text("file_name,relative_path\nfoo.pdf,foo.pdf\n")

    reset_csv_ledger(csv_file)

    assert csv_file.exists()
    ledger = IngestionLedger(csv_path=str(csv_file))
    assert ledger.get_all_records() == []


def test_cleanup_temp_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.tmp").write_text("temp")
    (data_dir / "keep.pdf").write_text("pdf")

    cleanup_temp_files(data_dir)

    assert not (data_dir / "test.tmp").exists()
    assert (data_dir / "keep.pdf").exists()
