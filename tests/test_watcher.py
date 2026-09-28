import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Ensure src is in import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wayfinder_ingestion.watcher import is_prefect_server_running, DocumentFolderWatcher


def test_is_prefect_server_running_true():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        running = is_prefect_server_running("http://localhost:4200/api")
        assert running is True


def test_is_prefect_server_running_false():
    import requests
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        running = is_prefect_server_running("http://localhost:4200/api")
        assert running is False


def test_watcher_refuses_to_start_when_server_is_down(tmp_path):
    source_dir = tmp_path / "source"
    chroma_dir = tmp_path / "chroma"
    csv_file = tmp_path / "ledger.csv"
    source_dir.mkdir()
    chroma_dir.mkdir()

    with patch("wayfinder_ingestion.watcher.is_prefect_server_running", return_value=False):
        watcher = DocumentFolderWatcher(
            source_dir=str(source_dir),
            chroma_dir=str(chroma_dir),
            csv_path=str(csv_file),
            exit_on_server_down=False
        )

        with pytest.raises(RuntimeError) as exc_info:
            watcher.watch()

        assert "Prefect server is NOT running" in str(exc_info.value)


def test_watcher_stops_when_server_goes_down(tmp_path):
    source_dir = tmp_path / "source"
    chroma_dir = tmp_path / "chroma"
    csv_file = tmp_path / "ledger.csv"
    source_dir.mkdir()
    chroma_dir.mkdir()

    # Sequence: first check on startup returns True, second check in loop returns False
    server_status_sequence = [True, False]

    def mock_server_health(*args, **kwargs):
        if server_status_sequence:
            return server_status_sequence.pop(0)
        return False

    with patch("wayfinder_ingestion.watcher.is_prefect_server_running", side_effect=mock_server_health), \
         patch("time.sleep", return_value=None):

        watcher = DocumentFolderWatcher(
            source_dir=str(source_dir),
            chroma_dir=str(chroma_dir),
            csv_path=str(csv_file),
            interval_seconds=1
        )

        watcher.watch(max_iterations=5)

        # Loop must have terminated as soon as server went down
        assert watcher._running is False


def test_watchdog_pdf_event_triggers_scan():
    from wayfinder_ingestion.watcher import WATCHDOG_AVAILABLE
    if not WATCHDOG_AVAILABLE:
        pytest.skip("watchdog is not installed in the test environment")

    from wayfinder_ingestion.watcher import PDFChangeEventHandler
    from watchdog.events import FileModifiedEvent

    mock_watcher = MagicMock()
    mock_watcher.check_server.return_value = True

    handler = PDFChangeEventHandler(watcher=mock_watcher, debounce_seconds=0.01)

    # Event for non-pdf should not trigger scan
    handler.on_any_event(FileModifiedEvent("test.txt"))
    mock_watcher.scan_and_process.assert_not_called()

    # Event for PDF should trigger scan when server is up
    handler.on_any_event(FileModifiedEvent("policy.pdf"))
    mock_watcher.scan_and_process.assert_called_once()

