"""Helpers for consistent CLI output formatting."""

from collections.abc import Mapping

_STATUS_PREFIX = {
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
}


def print_section(title: str, width: int = 55, border: str = "=") -> None:
    """Print section header with top and bottom border."""
    line = border * width
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def print_section_end(width: int = 55, border: str = "=") -> None:
    """Print section footer border."""
    print(f"{border * width}\n")


def print_kv(label: str, value: object, label_width: int = 12) -> None:
    """Print a key/value row with aligned label column."""
    print(f"  {label + ':':<{label_width}} {value}")


def print_kv_rows(rows: Mapping[str, object], label_width: int = 12) -> None:
    """Print multiple key/value rows."""
    for label, value in rows.items():
        print_kv(label, value, label_width=label_width)


def print_rule(title: str | None = None, width: int = 40, border: str = "-") -> None:
    """Print separator line, optionally with title."""
    line = border * width
    print(f"\n  {line}")
    if title:
        print(f"  {title}")
        print(f"  {line}")


def print_status(message: str, status: str = "info") -> None:
    """Print status line with emoji prefix."""
    prefix = _STATUS_PREFIX.get(status, _STATUS_PREFIX["info"])
    print(f"{prefix}  {message}")


def print_exception(context: str, error: Exception) -> None:
    """Print standardized exception message for CLI command failures."""
    print_status(f"{context} failed: {error}", status="error")
