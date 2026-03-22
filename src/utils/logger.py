"""
Logging configuration for trading system.

Usage:
    # On each run_*.py script, call setup_logging once at the start:
    logger = setup_logging("run_backtest", log_file="logs/backtest.log")

    # In src/ modules, use logging.getLogger(__name__) to get a logger:
    logger = logging.getLogger(__name__)
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── JSON Formatter ───


class JsonFormatter(logging.Formatter):
    """
    Structured JSON log formatter for production.

    Output each log record as a JSON object on a single line:
        {"ts": "2024-01-02T09:00:00Z", "level": "INFO", "logger": "...", "msg": "..."}
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # Include extra fields if any (e.g. logger.info("...", extra={"trade_id": 1}))
        skip = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
            "message",
        }
        for key, val in record.__dict__.items():
            if key not in skip:
                payload[key] = val

        return json.dumps(payload, ensure_ascii=False, default=str)


# ── Setup ───

# Tag used to track handlers created by setup_logging
# Avoid removing handlers from third-party libraries
_HANDLER_TAG = "_trading_v2"


def setup_logging(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    capture_all_loggers: bool | None = None,
    log_format: str | None = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Call once at the start of each run_*.py script - do not call in src/ modules.
    src/ modules should only use logging.getLogger(__name__).

    Args:
        name:
            Logger name, used in log output. Typically set to __name__ of the main script.
        log_file:
            Path to log file. None = log to console only.
        level:
            Base level for the logger (default: INFO).
            This is the minimum level that will be processed by the logger itself.
        console_level:
            Level for console handler (default: INFO).
            Logs below this level will not be printed to console.
        file_level:
            Level for file handler (default: DEBUG).
        capture_all_loggers:
            True  = Configure root logger, all loggers (including Optuna, psycopg2, etc.)
                    will go through these handlers.
            False = only configure logger with the given name, other loggers will be unaffected.
            None  = read from env LOG_CAPTURE_ALL (default "0" = False).
        log_format:
            "text" = human-readable (default for development).
            "json" = structured JSON (used for production/paper trading).
            None   = read from env LOG_FORMAT (default "text").

    Returns:
        Configured logger.
    """
    _fix_windows_encoding()

    # Resolve defaults from environment variables
    if capture_all_loggers is None:
        capture_all_loggers = os.getenv("LOG_CAPTURE_ALL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if log_format is None:
        log_format = os.getenv("LOG_FORMAT", "text").strip().lower()

    formatter = _make_formatter(log_format)

    if capture_all_loggers:
        _setup_root_logger(
            formatter=formatter,
            console_level=console_level,
            file_level=file_level,
            log_file=log_file,
            min_level=min(level, console_level, file_level),
        )
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()
        logger.propagate = True
        return logger

    # Script-only mode
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console_h = _make_console_handler(formatter, console_level)
    logger.addHandler(console_h)

    if log_file:
        file_h = _make_file_handler(formatter, file_level, log_file)
        logger.addHandler(file_h)

    return logger


# ── Private helpers ───


def _setup_root_logger(
    formatter: logging.Formatter,
    console_level: int,
    file_level: int,
    log_file: str | None,
    min_level: int,
) -> None:
    """
    Configure root logger.
    """
    root = logging.getLogger()
    root.setLevel(min_level)

    # Only remove handlers that we added in a previous setup_logging call
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)

    console_h = _make_console_handler(formatter, console_level)
    setattr(console_h, _HANDLER_TAG, True)
    root.addHandler(console_h)

    if log_file:
        file_h = _make_file_handler(formatter, file_level, log_file)
        setattr(file_h, _HANDLER_TAG, True)
        root.addHandler(file_h)


def _make_formatter(log_format: str) -> logging.Formatter:
    if log_format == "json":
        return JsonFormatter()
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _make_console_handler(
    formatter: logging.Formatter,
    level: int,
) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _make_file_handler(
    formatter: logging.Formatter,
    level: int,
    log_file: str,
) -> logging.FileHandler:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _fix_windows_encoding() -> None:
    """Ensure stdout/stderr handle Unicode on Windows cp1252 consoles."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
