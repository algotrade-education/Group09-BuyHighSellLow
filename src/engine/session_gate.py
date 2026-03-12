"""Session gate helpers shared by paper and backtest execution layers."""

from datetime import datetime, time
from typing import Optional

MORNING_START = time(9, 0)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(14, 30)
ATC_END = time(14, 45)


def vn30_is_trading_time(dt: datetime) -> bool:
    """Return True when dt is inside VN30 tradable windows."""
    t = dt.time()
    return (
        MORNING_START <= t < MORNING_END
        or AFTERNOON_START <= t < AFTERNOON_END
        or AFTERNOON_END <= t < ATC_END
    )


def vn30_active_window_end(dt: datetime) -> Optional[datetime]:
    """Return datetime end of the current VN30 sub-session window, else None."""
    t = dt.time()

    if MORNING_START <= t < MORNING_END:
        return dt.replace(
            hour=MORNING_END.hour,
            minute=MORNING_END.minute,
            second=0,
            microsecond=0,
        )

    if AFTERNOON_START <= t < AFTERNOON_END:
        return dt.replace(
            hour=AFTERNOON_END.hour,
            minute=AFTERNOON_END.minute,
            second=0,
            microsecond=0,
        )

    if AFTERNOON_END <= t < ATC_END:
        return dt.replace(
            hour=ATC_END.hour,
            minute=ATC_END.minute,
            second=0,
            microsecond=0,
        )

    return None


def vn30_seconds_to_window_end(dt: datetime) -> Optional[float]:
    """Return seconds until active sub-session end, else None if outside trading."""
    window_end = vn30_active_window_end(dt)
    if window_end is None:
        return None

    return max(0.0, (window_end - dt).total_seconds())


def vn30_is_entry_blocked(
    dt: datetime,
    entry_cutoff_seconds: float,
    allow_late_entry: bool,
) -> bool:
    """Return True when new entries should be blocked at dt for VN30."""
    if allow_late_entry:
        return False

    if entry_cutoff_seconds <= 0:
        return False

    seconds_to_end = vn30_seconds_to_window_end(dt)
    if seconds_to_end is None:
        return True

    return seconds_to_end <= entry_cutoff_seconds
