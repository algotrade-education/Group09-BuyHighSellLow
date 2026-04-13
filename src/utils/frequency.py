"""Frequency parsing and conversion utilities.

Centralized handling of frequency strings across the codebase.
Supports formats: '5min', '5', '1H', '1D', etc.
"""

from typing import Literal

# Type alias for supported frequency strings
ResampleFreq = Literal["1min", "5min", "15min", "30min", "1H", "1D", "1W", "1M"]


def parse_frequency_to_minutes(freq: str | int) -> int:
    """Parse frequency string or integer to minutes.

    Args:
        freq: Frequency as string ('5min', '1H', '1D') or integer (5).

    Returns:
        Frequency in minutes.

    Raises:
        ValueError: If frequency format is invalid or unsupported.

    Examples:
        >>> parse_frequency_to_minutes("5min")
        5
        >>> parse_frequency_to_minutes("1H")
        60
        >>> parse_frequency_to_minutes(5)
        5
    """
    if isinstance(freq, int):
        if freq <= 0:
            raise ValueError(f"Frequency must be positive, got {freq}")
        return freq

    freq_str = freq.strip().lower()
    if not freq_str:
        raise ValueError("Frequency string cannot be empty")

    # Handle pure digit strings (e.g., "5")
    if freq_str.isdigit():
        minutes = int(freq_str)
        if minutes <= 0:
            raise ValueError(f"Frequency must be positive, got {minutes}")
        return minutes

    # Handle minute format (e.g., "5min")
    if freq_str.endswith("min"):
        try:
            minutes = int(freq_str[:-3])
            if minutes <= 0:
                raise ValueError(f"Frequency must be positive, got {minutes}")
            return minutes
        except ValueError as e:
            raise ValueError(f"Invalid minute format '{freq}': {e}") from e

    # Handle hour format (e.g., "1H", "1h")
    if freq_str.endswith("h"):
        try:
            hours = int(freq_str[:-1])
            if hours <= 0:
                raise ValueError(f"Frequency must be positive, got {hours}")
            return hours * 60
        except ValueError as e:
            raise ValueError(f"Invalid hour format '{freq}': {e}") from e

    # Handle day format (e.g., "1D", "1d")
    if freq_str.endswith("d"):
        try:
            days = int(freq_str[:-1])
            if days <= 0:
                raise ValueError(f"Frequency must be positive, got {days}")
            # 255 trading minutes/day for VN30 (150 morning + 105 afternoon, excludes ATC)
            # Source of truth: VN30SessionConfig.TRADING_MINUTES_PER_DAY
            return days * 255
        except ValueError as e:
            raise ValueError(f"Invalid day format '{freq}': {e}") from e

    # Handle week format (e.g., "1W", "1w")
    if freq_str.endswith("w"):
        try:
            weeks = int(freq_str[:-1])
            if weeks <= 0:
                raise ValueError(f"Frequency must be positive, got {weeks}")
            # 5 trading days * 255 minutes = 1275 minutes/week
            return weeks * 1275
        except ValueError as e:
            raise ValueError(f"Invalid week format '{freq}': {e}") from e

    # Handle month format (e.g., "1M", "1m")
    if freq_str.endswith("m") and len(freq_str) > 1:
        # Check if it's "min" format first
        if freq_str.endswith("min"):
            try:
                minutes = int(freq_str[:-3])
                if minutes <= 0:
                    raise ValueError(f"Frequency must be positive, got {minutes}")
                return minutes
            except ValueError as e:
                raise ValueError(f"Invalid minute format '{freq}': {e}") from e
        # Otherwise treat as month
        try:
            months = int(freq_str[:-1])
            if months <= 0:
                raise ValueError(f"Frequency must be positive, got {months}")
            # 20 trading days * 255 minutes = 5100 minutes/month
            return months * 5100
        except ValueError as e:
            raise ValueError(f"Invalid month format '{freq}': {e}") from e

    raise ValueError(
        f"Unsupported frequency format '{freq}'. "
        f"Supported formats: '5min', '5', '1H', '1D', '1W', '1M'"
    )


def format_minutes_to_frequency(minutes: int) -> str:
    """Convert minutes to frequency string.

    Args:
        minutes: Frequency in minutes.

    Returns:
        Frequency string (e.g., '5min', '1H').

    Raises:
        ValueError: If minutes is not positive.

    Examples:
        >>> format_minutes_to_frequency(5)
        '5min'
        >>> format_minutes_to_frequency(60)
        '1H'
        >>> format_minutes_to_frequency(255)
        '1D'
    """
    if minutes <= 0:
        raise ValueError(f"Minutes must be positive, got {minutes}")

    # Check for common conversions
    if minutes == 255:
        return "1D"
    elif minutes == 60:
        return "1H"
    elif minutes % 60 == 0:
        return f"{minutes // 60}H"
    else:
        return f"{minutes}min"


def to_pandas_offset(freq: str | int) -> str:
    """Convert frequency to pandas offset alias.

    Args:
        freq: Frequency as string or integer.

    Returns:
        Pandas offset alias string.

    Raises:
        ValueError: If frequency is invalid or unsupported.

    Examples:
        >>> to_pandas_offset("5min")
        '5min'
        >>> to_pandas_offset("1H")
        '1H'
        >>> to_pandas_offset(5)
        '5min'
    """
    # Mapping of common frequency strings to pandas offset aliases
    mapping = {
        "1min": "1min",
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "1h": "1H",
        "1d": "1D",
        "1w": "1W",
        "1m": "1ME",  # Month end frequency in pandas
    }

    if isinstance(freq, int):
        return f"{freq}min"

    freq_lower = freq.strip().lower()

    # Direct mapping lookup
    if freq_lower in mapping:
        return mapping[freq_lower]

    # Handle minute formats
    if freq_lower.endswith("min"):
        return freq_lower

    # Handle hour formats
    if freq_lower.endswith("h"):
        return freq_lower.upper()

    # Handle day formats
    if freq_lower.endswith("d"):
        return freq_lower.upper()

    # Handle week formats
    if freq_lower.endswith("w"):
        return freq_lower.upper()

    # Handle month formats (convert to month-end)
    if freq_lower.endswith("m") and not freq_lower.endswith("min"):
        value = freq_lower[:-1]
        return f"{value}ME"

    raise ValueError(
        f"Unsupported frequency '{freq}' for pandas conversion. Supported: {list(mapping.keys())}"
    )


def validate_frequency(freq: str | int) -> bool:
    """Validate frequency format.

    Args:
        freq: Frequency to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_frequency("5min")
        True
        >>> validate_frequency("invalid")
        False
    """
    try:
        parse_frequency_to_minutes(freq)
        return True
    except (ValueError, TypeError):
        return False


def normalize_frequency(freq: str | int) -> str:
    """Normalize frequency to standard string format.

    Args:
        freq: Frequency in any supported format.

    Returns:
        Normalized frequency string (e.g., '5min', '1H').

    Examples:
        >>> normalize_frequency(5)
        '5min'
        >>> normalize_frequency("5")
        '5min'
        >>> normalize_frequency("5min")
        '5min'
    """
    minutes = parse_frequency_to_minutes(freq)
    return format_minutes_to_frequency(minutes)
