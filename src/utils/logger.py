"""
Logging utility for the backtesting framework.
Provides a standardized way to configure logging for both console and file outputs.
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    capture_all_loggers: Optional[bool] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        name: Logger name.
        log_file: Optional path to log file.
        level: Base logging level.
        console_level: Level for console output (info level usually).
        file_level: Level for file output (debug level usually).
        capture_all_loggers: If True, configure root logger so all module logs
            go to this script's handlers. If False, capture only this logger.
            If None, reads env LOG_CAPTURE_ALL (default: "1").

    Returns:
        Configured logger.
    """
    # Ensure stdout can handle Unicode characters (e.g. emoji) on Windows cp1252 consoles.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logger = logging.getLogger(name)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    if capture_all_loggers is None:
        capture_all_loggers = os.getenv("LOG_CAPTURE_ALL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if capture_all_loggers:
        # Configure ROOT logger so all module loggers propagate to the same sinks.
        # This also neutralizes any prior basicConfig() side effects.
        root_logger = logging.getLogger()
        root_level = min(level, console_level, file_level)
        root_logger.setLevel(root_level)

        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        logger.setLevel(level)
        logger.handlers.clear()
        logger.propagate = True
        return logger

    # Script-only mode: only this logger writes to handlers.
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
