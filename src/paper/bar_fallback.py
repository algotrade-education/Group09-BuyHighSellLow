"""
DB fallback bar lookup for paper trading bar generation.

Pure function module for loading historical bars from database when live data
is insufficient. Uses dependency injection for testability and follows the
callback pattern established in data_quality.py.
"""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from src.database.base import DataServiceBase

logger = logging.getLogger(__name__)


def load_fallback_bar_for_bucket(
    data_service: DataServiceBase,
    *,
    symbol: str,
    bucket_dt: datetime,
    freq_minutes: int,
) -> dict[str, Any] | None:
    """Load a closed bar from DB when no reliable live trades are available.

    Parameters
    ----------
    data_service:
        DataServiceBase instance for fetching bar data from database.
    symbol:
        Full instrument symbol (e.g. "HOSE:VN30F1M").
    bucket_dt:
        Start datetime of the bar bucket.
    freq_minutes:
        Bar frequency in minutes.

    Returns
    -------
    dict[str, Any] | None
        Bar dict with keys: datetime, open, high, low, close, volume, rows.
        None if no data available or error occurred.
    """
    contract = symbol.split(":")[-1]
    db_symbol = "VN30F1M" if contract.startswith("VN30F") else contract
    bucket_end = bucket_dt + timedelta(minutes=freq_minutes)

    try:
        bar_df = data_service.fetch_bucket_bar(
            contract_name=db_symbol,
            bucket_start=bucket_dt,
            bucket_end=bucket_end,
        )
        if bar_df.empty:
            return None

        row = bar_df.iloc[-1]
        return {
            "datetime": bucket_dt,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0) or 0.0),
            "rows": int(row.get("rows", 0) or 0),
        }
    except Exception as exc:
        logger.warning(
            "DB bar fallback lookup failed for %s (%s): %s",
            bucket_dt.strftime("%Y-%m-%d %H:%M"),
            symbol,
            exc,
        )
        return None


def create_fallback_provider(
    data_service: DataServiceBase,
    symbol: str,
    freq_minutes: int,
    enabled: bool = True,
) -> Callable[[datetime], dict[str, Any] | None]:
    """Factory function to create a fallback bar provider callback.

    This follows the callback pattern used in data_quality.py's maybe_merge_db_bar,
    allowing the bar aggregator to request fallback bars without knowing about
    database implementation details.

    Parameters
    ----------
    data_service:
        DataServiceBase instance for database access.
    symbol:
        Full instrument symbol (e.g. "HOSE:VN30F1M").
    freq_minutes:
        Bar frequency in minutes.
    enabled:
        Whether fallback is enabled. If False, returns a no-op callback.

    Returns
    -------
    Callable[[datetime], dict[str, Any] | None]
        Callback that takes bucket_start datetime and returns bar dict or None.

    Example
    -------
    >>> from src.database import get_data_service
    >>> provider = create_fallback_provider(
    ...     data_service=get_data_service(),
    ...     symbol="HOSE:VN30F1M",
    ...     freq_minutes=5,
    ...     enabled=True
    ... )
    >>> bar = provider(datetime(2024, 1, 1, 9, 0))
    """
    if not enabled:
        return lambda bucket_dt: None

    def provider(bucket_dt: datetime) -> dict[str, Any] | None:
        return load_fallback_bar_for_bucket(
            data_service=data_service,
            symbol=symbol,
            bucket_dt=bucket_dt,
            freq_minutes=freq_minutes,
        )

    return provider
