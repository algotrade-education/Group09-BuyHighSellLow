"""
Warmup data cache for live paper trading.

Historical tick data (merged close + matched) is expensive to re-fetch on every
server start.  This module caches one Parquet file per calendar day so that
only *today's* data (which is still accumulating) must be fetched from the DB;
everything before today is served from disk.

Cache layout
------------
data/cache/warmup/{db_symbol}/{YYYY-MM-DD}.parquet

Invalidation strategy
---------------------
- Past days are immutable → cached indefinitely (optional TTL via ``max_age_days``).
- Today is always re-fetched from the DB so the engine gets up-to-the-minute ticks.
- Files for dates outside the requested window are left on disk (cheap storage)
  and simply ignored.  Call ``evict_old_files()`` to clean up if desired.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.database.base import DataServiceBase

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_ROOT = Path("data/cache/warmup")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _day_path(cache_root: Path, db_symbol: str, day: date) -> Path:
    return cache_root / db_symbol / f"{day.isoformat()}.parquet"


def _load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # corrupt / version mismatch
        logger.warning("Warmup cache: could not read %s (%s) - will re-fetch.", path, exc)
        return pd.DataFrame()


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        logger.debug("Warmup cache: saved %d rows to %s", len(df), path)
    except Exception as exc:
        logger.warning("Warmup cache: could not write %s (%s).", path, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_with_cache(
    data_service: DataServiceBase,
    db_symbol: str,
    n_days: int = 5,
    *,
    cache_root: Path = _DEFAULT_CACHE_ROOT,
    today: date | None = None,
) -> pd.DataFrame:
    """Return merged tick data for the last ``n_days`` calendar days.

    Past days are loaded from the on-disk Parquet cache when available.
    Today's data is always fetched fresh from the DB.

    Parameters
    ----------
    data_service:
        DataServiceBase instance for fetching tick data from database.
    db_symbol:
        The contract identifier (e.g. ``"VN30F1M"``).
    n_days:
        How many calendar days back to look (inclusive of today).
    cache_root:
        Root directory for Parquet day-files.
    today:
        Override the current date (useful for testing).

    Returns
    -------
    pd.DataFrame
        Merged tick data with columns: datetime, tickersymbol, price, quantity.
        Empty DataFrame if no data available.
    """
    today = today or date.today()
    start_date = today - timedelta(days=n_days - 1)

    dates: list[date] = [start_date + timedelta(days=i) for i in range(n_days)]

    frames: list[pd.DataFrame] = []
    missing_past_days: list[date] = []

    # ── Separate cached / uncached past days ──────────────────────────────
    for day in dates:
        if day >= today:
            continue  # today handled separately below

        path = _day_path(cache_root, db_symbol, day)
        if path.exists():
            df = _load_parquet(path)
            if not df.empty:
                frames.append(df)
                logger.debug("Warmup cache: loaded %d rows for %s from cache.", len(df), day)
                continue
        missing_past_days.append(day)

    # ── Fetch missing past days in one DB call (if any) ───────────────────
    if missing_past_days:
        min_day = min(missing_past_days)
        max_day = max(missing_past_days)
        logger.info(
            "Warmup cache: fetching %d uncached past day(s) [%s - %s] from DB…",
            len(missing_past_days),
            min_day,
            max_day,
        )
        try:
            raw = data_service.get_matched_data(
                contract_name=db_symbol,
                from_date=min_day.isoformat(),
                to_date=max_day.isoformat(),
            )
            if not raw.empty:
                _save_days(raw, missing_past_days, cache_root, db_symbol)
                frames.append(raw)
        except Exception as exc:
            logger.error(
                "Warmup cache: failed to fetch past days [%s - %s]: %s",
                min_day,
                max_day,
                exc,
            )

    # ── Always fetch today fresh ──────────────────────────────────────────
    logger.info("Warmup cache: fetching today (%s) from DB…", today)
    try:
        today_raw = data_service.get_matched_data(
            contract_name=db_symbol,
            from_date=today.isoformat(),
            to_date=today.isoformat(),
        )
        if not today_raw.empty:
            frames.append(today_raw)
        else:
            logger.debug("Warmup cache: no data for today yet.")
    except Exception as exc:
        logger.error("Warmup cache: failed to fetch today (%s): %s", today, exc)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    # De-duplicate rows that might appear in both cache and a fresh fetch
    if "datetime" in merged.columns and "tickersymbol" in merged.columns:
        merged = (
            merged.drop_duplicates(subset=["datetime", "tickersymbol"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    return merged


def evict_old_files(
    db_symbol: str,
    max_age_days: int = 30,
    *,
    cache_root: Path = _DEFAULT_CACHE_ROOT,
) -> int:
    """Delete day-files older than ``max_age_days`` for ``db_symbol``.

    Returns the number of files removed.
    """
    cutoff = date.today() - timedelta(days=max_age_days)
    symbol_dir = cache_root / db_symbol
    if not symbol_dir.exists():
        return 0

    removed = 0
    for path in symbol_dir.glob("*.parquet"):
        try:
            file_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if file_date < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
            logger.debug("Warmup cache: evicted %s", path)

    if removed:
        logger.info("Warmup cache: evicted %d old file(s) for %s.", removed, db_symbol)
    return removed


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _save_days(
    df: pd.DataFrame,
    days: Sequence[date],
    cache_root: Path,
    db_symbol: str,
) -> None:
    """Persist each day's slice of *df* to its own Parquet file."""
    if "datetime" not in df.columns:
        return
    df = df.copy()
    df["_date"] = pd.to_datetime(df["datetime"]).dt.date
    for day in days:
        day_df = df[df["_date"] == day].drop(columns=["_date"])
        if day_df.empty:
            logger.debug("Warmup cache: no rows for %s - skipping cache write.", day)
            continue
        _save_parquet(day_df, _day_path(cache_root, db_symbol, day))
