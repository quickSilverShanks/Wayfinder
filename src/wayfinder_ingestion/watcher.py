import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import requests

from wayfinder_ingestion.config import Settings, get_settings
from wayfinder_ingestion.change_detector import ChangeDetector
from wayfinder_ingestion.logging_config import setup_logger
from wayfinder_ingestion.pipeline import ingest_documents_flow

# Watchdog integration with graceful fallback
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

logger = setup_logger(__name__)


def is_prefect_server_running(api_url: Optional[str] = None, timeout: float = 2.0) -> bool:
    """
    Checks if a Prefect server is actively running and responding to health check requests.
    Validates against candidate endpoints:
    - {api_url}/health
    - {base_url}/health
    - Equivalent 127.0.0.1 addresses
    """
    settings = get_settings()
    target_url = (api_url or settings.effective_prefect_api_url).rstrip("/")

    # Derive candidate URLs
    candidate_urls = [
        f"{target_url}/health",
        f"{target_url.replace('/api', '')}/health" if target_url.endswith("/api") else f"{target_url}/health"
    ]

    # Add 127.0.0.1 variations if localhost is used (avoids Windows IPv6 resolution latency)
    if "localhost" in target_url:
        target_ip_url = target_url.replace("localhost", "127.0.0.1")
        candidate_urls.extend([
            f"{target_ip_url}/health",
            f"{target_ip_url.replace('/api', '')}/health" if target_ip_url.endswith("/api") else f"{target_ip_url}/health"
        ])

    for url in candidate_urls:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue
        except Exception as e:
            logger.debug(f"Prefect server probe to {url} failed: {e}")
            continue

    return False


def check_prefect_server_status(api_url: Optional[str] = None) -> Tuple[bool, str]:
    """
    Returns boolean indicating whether Prefect server is active along with a descriptive message.
    """
    settings = get_settings()
    target_url = (api_url or settings.effective_prefect_api_url).rstrip("/")
    running = is_prefect_server_running(target_url)
    if running:
        return True, f"Prefect server is running and reachable at {target_url}"
    else:
        return False, (
            f"Prefect server is NOT running at {target_url}. "
            f"Start it with: 'prefect server start'"
        )


if WATCHDOG_AVAILABLE:
    class PDFChangeEventHandler(FileSystemEventHandler):
        """
        Watchdog event handler for PDF filesystem events in the source directory hierarchy.
        Debounces rapid OS write events and checks Prefect server health before processing.
        """

        def __init__(self, watcher: "DocumentFolderWatcher", debounce_seconds: float = 1.0):
            super().__init__()
            self.watcher = watcher
            self.debounce_seconds = debounce_seconds
            self._last_event_time = 0.0

        def _is_pdf(self, path: str) -> bool:
            return path.lower().endswith(".pdf")

        def on_any_event(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                return

            affects_pdf = False
            if hasattr(event, "src_path") and self._is_pdf(event.src_path):
                affects_pdf = True
            if hasattr(event, "dest_path") and self._is_pdf(event.dest_path):
                affects_pdf = True

            if affects_pdf:
                now = time.time()
                # Debounce to prevent multiple triggers while a file is actively being copied/written
                if now - self._last_event_time > self.debounce_seconds:
                    self._last_event_time = now
                    logger.info(f"[Watchdog] File event '{event.event_type}' detected: {getattr(event, 'src_path', '')}")

                    # Enforce server health check on each event
                    if self.watcher.check_server():
                        self.watcher.scan_and_process()
                    else:
                        logger.warning(
                            "[Watchdog] Event detected but Prefect server is NOT RUNNING. "
                            "Refusing to process and halting watcher."
                        )
                        self.watcher.stop()


class DocumentFolderWatcher:
    """
    Watches the source document directory and all nested subdirectories for PDF changes
    'ONLY' as long as the Prefect server is actively running.
    
    Uses the 'watchdog' library for real-time OS filesystem event monitoring.
    If the Prefect server is not running at start, watching will refuse to begin.
    If the Prefect server stops during execution, watching terminates immediately.
    """

    def __init__(
        self,
        source_dir: Optional[str] = None,
        chroma_dir: Optional[str] = None,
        csv_path: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        prefect_api_url: Optional[str] = None,
        exit_on_server_down: bool = True
    ):
        self.settings: Settings = get_settings()

        if source_dir:
            self.settings.SOURCE_DOCUMENT_DIR = source_dir
        if chroma_dir:
            self.settings.CHROMA_PERSIST_DIR = chroma_dir
        if csv_path:
            self.settings.INGESTION_LOG_CSV = csv_path
        if interval_seconds is not None:
            self.settings.WATCH_INTERVAL_SECONDS = interval_seconds
        if prefect_api_url:
            self.settings.PREFECT_API_URL = prefect_api_url

        self.source_dir = self.settings.source_dir_path
        self.chroma_dir = self.settings.chroma_dir_path
        self.csv_path = self.settings.csv_ledger_path
        self.interval = self.settings.WATCH_INTERVAL_SECONDS
        self.api_url = self.settings.effective_prefect_api_url
        self.exit_on_server_down = exit_on_server_down
        self._running = False
        self._observer = None

    def check_server(self) -> bool:
        """Verifies if the Prefect server is currently operational."""
        return is_prefect_server_running(self.api_url)

    def scan_and_process(self) -> Optional[Dict[str, Any]]:
        """
        Runs change detection and triggers ingestion flow if any changes are detected.
        Returns the ingestion summary if triggered, or None if no changes were found.
        """
        detector = ChangeDetector(
            source_dir=str(self.source_dir),
            manifest_dir=str(self.chroma_dir),
            csv_path=str(self.csv_path)
        )
        change_set = detector.detect_changes()

        if change_set.has_changes:
            logger.info(
                f"[Watcher] Change detected in {self.source_dir}: "
                f"New={len(change_set.new_files)}, "
                f"Modified={len(change_set.modified_files)}, "
                f"Deleted={len(change_set.deleted_files)}"
            )
            summary = ingest_documents_flow(
                source_dir_override=str(self.source_dir),
                chroma_dir_override=str(self.chroma_dir),
                csv_path_override=str(self.csv_path)
            )
            return summary
        else:
            logger.debug(f"[Watcher] No changes detected in {self.source_dir}. Monitoring...")
            return None

    def watch(self, max_iterations: Optional[int] = None) -> None:
        """
        Enters continuous watch loop using the watchdog library.
        Crucial requirement: Runs 'ONLY' as long as the Prefect server is running.
        """
        is_up, status_msg = check_prefect_server_status(self.api_url)
        if not is_up:
            logger.error("=" * 60)
            logger.error(" CANNOT START FOLDER WATCH: PREFECT SERVER IS NOT RUNNING")
            logger.error(f" Target API: {self.api_url}")
            logger.error(" The data folder will be under watch 'ONLY' as long as")
            logger.error(" Prefect server is running.")
            logger.error(" Please start Prefect server in a separate terminal:")
            logger.error("     prefect server start")
            logger.error("=" * 60)
            if self.exit_on_server_down:
                return
            else:
                raise RuntimeError(status_msg)

        logger.info("=" * 60)
        logger.info(" Wayfinder Document Folder Watcher Active (Watchdog Powered)")
        logger.info(f" Monitored Directory: {self.source_dir}")
        logger.info(f" ChromaDB Directory: {self.chroma_dir}")
        logger.info(f" CSV Ledger:         {self.csv_path}")
        logger.info(f" Embedding Model:    {self.settings.EMBEDDING_MODEL} (from .env)")
        logger.info(f" Server Heartbeat:   every {self.interval}s")
        logger.info(f" Prefect Server:     {self.api_url} [HEALTHY]")
        logger.info(" Watching 'ONLY' as long as Prefect server remains running.")
        logger.info(" Press Ctrl+C to stop.")
        logger.info("=" * 60)

        self._running = True

        # Run initial catch-up scan to process any pending documents
        try:
            self.scan_and_process()
        except Exception as e:
            logger.error(f"[Watcher] Initial scan failed: {e}", exc_info=True)

        # Start Watchdog Observer if available
        if WATCHDOG_AVAILABLE:
            event_handler = PDFChangeEventHandler(watcher=self)
            self._observer = Observer()
            self._observer.schedule(event_handler, str(self.source_dir), recursive=True)
            self._observer.start()
            logger.info(f"[Watcher] Watchdog observer started monitoring '{self.source_dir}' recursively.")

        iteration_count = 0

        try:
            while self._running:
                # Continuous Heartbeat: Verify Prefect server is STILL running
                if not self.check_server():
                    logger.warning("=" * 60)
                    logger.warning(" [STOPPING WATCH] PREFECT SERVER HAS STOPPED RUNNING!")
                    logger.warning(f" Target API: {self.api_url}")
                    logger.warning(" As configured, the data folder is watched 'ONLY' while")
                    logger.warning(" Prefect server is active.")
                    logger.warning(" Halting document folder watch immediately.")
                    logger.warning("=" * 60)
                    self.stop()
                    break

                # If watchdog is not installed, fallback to periodic polling
                if not WATCHDOG_AVAILABLE:
                    try:
                        self.scan_and_process()
                    except Exception as e:
                        logger.error(f"[Watcher] Error during polling cycle: {e}", exc_info=True)

                iteration_count += 1
                if max_iterations is not None and iteration_count >= max_iterations:
                    logger.info(f"[Watcher] Reached max iterations ({max_iterations}). Stopping watch.")
                    self.stop()
                    break

                time.sleep(self.interval)

        except KeyboardInterrupt:
            logger.info("\n[Watcher] Folder watch stopped by user (Ctrl+C).")
        finally:
            self.stop()
            logger.info("[Watcher] Document folder watch concluded.")

    def stop(self) -> None:
        """Stops the watch loop and shuts down watchdog observer."""
        self._running = False
        if self._observer is not None:
            try:
                if self._observer.is_alive():
                    self._observer.stop()
                    self._observer.join(timeout=2.0)
            except Exception as e:
                logger.debug(f"[Watcher] Error stopping observer: {e}")
            self._observer = None


def main():
    parser = argparse.ArgumentParser(
        description="Wayfinder Document Folder Watcher (Watches 'ONLY' while Prefect server is running)"
    )
    parser.add_argument("--source-dir", type=str, default=None, help="Directory containing source PDF files")
    parser.add_argument("--chroma-dir", type=str, default=None, help="Directory for ChromaDB persistence")
    parser.add_argument("--csv-path", type=str, default=None, help="Path for CSV ledger")
    parser.add_argument("--interval", type=int, default=None, help="Polling interval in seconds (default: 5)")
    parser.add_argument("--prefect-url", type=str, default=None, help="Prefect server API URL")
    args = parser.parse_args()

    watcher = DocumentFolderWatcher(
        source_dir=args.source_dir,
        chroma_dir=args.chroma_dir,
        csv_path=args.csv_path,
        interval_seconds=args.interval,
        prefect_api_url=args.prefect_url
    )
    watcher.watch()


if __name__ == "__main__":
    main()
