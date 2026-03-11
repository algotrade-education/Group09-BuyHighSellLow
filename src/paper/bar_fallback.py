"""DB fallback bar lookup for paper trading bar generation."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional


def load_fallback_bar_for_bucket(
    *,
    symbol: str,
    bucket_dt: datetime,
    freq_minutes: int,
    enabled: bool,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Load a closed bar from DB when no reliable live trades are available."""
    if not enabled:
        return None

    contract = symbol.split(":")[-1]
    db_symbol = "VN30F1M" if contract.startswith("VN30F") else contract
    bucket_end = bucket_dt + timedelta(minutes=freq_minutes)

    try:
        from src.database.data_service import fetch_bucket_bar

        bar_df = fetch_bucket_bar(
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
