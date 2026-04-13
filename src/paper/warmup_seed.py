"""Warmup seeding helpers for paper trading startup.

This module owns the transformation from the warmup OHLCV DataFrame to a
seed payload for ``BarAggregator.seed_current_live_bar``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def extract_incomplete_bar(raw_df: pd.DataFrame) -> dict[str, Any] | None:
    """Extract the last warmup bar as seed payload for current live bucket.

    Args:
        raw_df: Historical OHLCV DataFrame.

    Returns:
        Seed dict with keys ``datetime``, ``open``, ``high``, ``low``, ``close``,
        and ``volume``; ``None`` when ``raw_df`` is empty.
    """
    if raw_df.empty:
        return None

    last_row = raw_df.iloc[-1].copy()
    dt_val = last_row["datetime"]
    if isinstance(dt_val, pd.Timestamp):
        dt_val = dt_val.to_pydatetime()
    elif not isinstance(dt_val, datetime):
        dt_val = pd.Timestamp(dt_val).to_pydatetime()

    return {
        "datetime": dt_val,
        "open": float(last_row["open"]),
        "high": float(last_row["high"]),
        "low": float(last_row["low"]),
        "close": float(last_row["close"]),
        "volume": float(last_row["volume"]),
    }
