import logging
import sys
from typing import Optional


def setup_logger(name: str = "wayfinder_ingestion", level_name: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a logger for the Wayfinder ingestion pipeline.
    Ensures log outputs contain diagnostic metadata without leaking document body or PII.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format includes timestamp, log level, logger name, module/line, and message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
