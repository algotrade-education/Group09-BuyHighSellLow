"""
Load OHLCV data from Database or CSV, validate, cache as parquet.
This module handles 1-minute bars.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.data.validators import DataValidator
from src.database.base import DataServiceBase

logger = logging.getLogger(__name__)

DATETIME_COL = "datetime"
_CACHE_VERSION = "v2"
_REQUIRED_COLS = [DATETIME_COL, "open", "high", "low", "close", "volume"]


# --- Manifest ---


class CacheManifest:
    """
    Per-symbol JSON manifest tracking which months are cached and their metadata.

    Stored at: data/cache/<symbol>/manifest.json

    Schema:
    {
        "symbol": "VN30F1M",
        "cache_version": "v2",
        "months": {
            "2023_01": {
                "row_count": 12852,
                "source": "database",
                "complete": true,
                "last_synced_timestamp": "2024-01-31T23:59:00+00:00",
                "created_at": "2024-01-15T09:30:00+00:00",
            }
        }
    }
    """

    def __init__(self, path: Path, symbol: str) -> None:
        self._path = path
        self._symbol = symbol
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                parsed = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    return cast(dict[str, Any], parsed)
                logger.warning("Manifest at %s is not a JSON object - resetting.", self._path)
            except Exception as e:
                logger.warning("Manifest corrupt at %s: %s - resetting.", self._path, e)

        return {
            "symbol": self._symbol,
            "cache_version": _CACHE_VERSION,
            "months": {},
        }

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_cached(self, month_key: str) -> bool:
        """Return True if month is present and marked complete."""
        entry = self._data["months"].get(month_key)
        return entry is not None and entry.get("complete", False)

    def is_stale(self, month_key: str, max_age_days: int) -> bool:
        """Return True if cache entry is older than max_age_days."""
        entry = self._data["months"].get(month_key)
        if not entry:
            return True

        created = entry.get("created_at")
        if not created:
            return True

        try:
            age = (datetime.now(UTC) - datetime.fromisoformat(created)).days
            return age > max_age_days
        except Exception:
            return True

    def record(
        self,
        month_key: str,
        row_count: int,
        source: str = "database",
        complete: bool | None = None,
        last_synced_timestamp: str | None = None,
    ) -> None:
        """
        Record or update cache metadata for a month.

        Args:
            month_key: Month identifier in "YYYY_MM" format (e.g., "2024_03").
            row_count: Number of rows/bars in the cached data.
            source: Data source identifier ("database", "tick_csv", etc.).
            complete: Whether the month is complete and won't need updates.
                      None = auto-detect (past months are complete, current month is not).
            last_synced_timestamp: ISO format timestamp of the last data point synced.
                                   Used for incremental updates of current month.
                                   Example: "2024-03-23T15:30:00+00:00"

        Notes:
            - Past months are marked complete=True by default (won't refetch).
            - Current month is marked complete=False (allows incremental updates).
            - last_synced_timestamp enables efficient incremental fetching by
              tracking the latest data point, so next fetch only pulls new data
              from that timestamp forward instead of re-fetching the entire month.
        """
        # Auto-detect: past months are complete, current month is not.
        # Explicit True/False overrides auto-detection.
        if complete is None:
            complete = not _is_current_month(month_key)

        self._data["months"][month_key] = {
            "row_count": row_count,
            "created_at": datetime.now(UTC).isoformat(),
            "source": source,
            "complete": complete,
        }

        # Store last synced timestamp for incremental updates
        if last_synced_timestamp:
            self._data["months"][month_key]["last_synced_timestamp"] = last_synced_timestamp

        self.save()

    def invalidate(self, month_key: str | None = None) -> None:
        """Remove one month entry or clear all months."""
        if month_key is None:
            self._data["months"] = {}
        else:
            self._data["months"].pop(month_key, None)

        self.save()

    @property
    def cached_months(self) -> list[str]:
        return [k for k, v in self._data["months"].items() if v.get("complete")]

    def get_last_synced_timestamp(self, month_key: str) -> str | None:
        """
        Get the last synced timestamp for a cached month.

        Args:
            month_key: Month identifier in "YYYY_MM" format.

        Returns:
            ISO format timestamp string of the last data point, or None if:
            - Month is not cached yet
            - Month entry exists but has no last_synced_timestamp field
            - Used for incremental fetching to avoid re-downloading existing data

        Example:
            >>> manifest.get_last_synced_timestamp("2024_03")
            "2024-03-23T15:30:00+00:00"
        """
        entry = self._data["months"].get(month_key)
        return entry.get("last_synced_timestamp") if entry else None

    def summary(self) -> str:
        months = self._data["months"]
        complete = sum(1 for v in months.values() if v.get("complete"))
        return f"{self._symbol}: {complete}/{len(months)} months cached"


# --- DataLoader ---


class DataLoader:
    """
    Load 1-minute OHLCV data with month-chunk parquet cache.

    Always returns 1-minute bars.
    Resampling to 5min/15min/30min is DataPreprocessor's responsibility.

    Usage:
        loader = DataLoader(data_service)

        # Load from DB with monthly cache
        df = loader.load("VN30F1M", "2023-01-01", "2024-12-31")

        # Force refresh specific months
        df = loader.load("VN30F1M", "2023-01-01", "2023-03-31",
                         force_months=["2023_01"])

        # One-time migration: load from legacy tick CSVs
        df = loader.load_tick_csv(
            "data/ticks/ticks_2023_*.csv",
            symbol="VN30F1M",
        )
    """

    def __init__(
        self,
        data_service: DataServiceBase,
        cache_dir: str = "data/cache",
        chunk_size_days: int = 30,
        cache_max_age_days: int = 7,
    ) -> None:
        """
        Args:
            data_service:       DB adapter implementing DataServiceBase.
            cache_dir:          Root directory for monthly parquet cache.
            chunk_size_days:    DB fetch chunk size forwarded to DataServiceBase.
            cache_max_age_days: Refetch month from DB if cache is older than this.
                                Set 0 to disable staleness check.
        """
        self._svc = data_service
        self._cache_root = Path(cache_dir)
        self._validator = DataValidator()
        self._chunk_size = chunk_size_days
        self._max_age_days = cache_max_age_days

    # --- Public API ---

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        use_cache: bool = True,
        force_months: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load 1-minute OHLCV data for symbol in [start, end].

        Fetches from monthly cache if available and fresh; otherwise pulls
        from DB and caches the result as monthly parquet files.

        Args:
            symbol:       Contract symbol, e.g. "VN30F1M".
            start:        Start date "YYYY-MM-DD" (inclusive).
            end:          End date "YYYY-MM-DD" (inclusive).
            use_cache:    Try monthly cache first; fall back to DB on miss.
            force_months: Month keys to force-refresh regardless of cache state,
                          e.g. ["2023_01", "2023_02"]. Useful when DB is
                          reprocessed for specific months.

        Returns:
            DataFrame sorted by datetime with columns:
            datetime, open, high, low, close, volume.

        Raises:
            ValueError:   No data found after fetching, or validation fails.
            RuntimeError: DB fetch fails with no cached fallback available.
        """
        force_set = set(force_months or [])
        months = _months_in_range(start, end)
        manifest = self._get_manifest(symbol)
        chunks: list[pd.DataFrame] = []

        for month_key in months:
            parquet_path = self._month_path(symbol, month_key)
            needs_fetch = (
                not use_cache
                or month_key in force_set
                or not manifest.is_cached(month_key)
                or not parquet_path.exists()
                or _is_current_month(month_key)  # always refetch - DB has more data today
                or (self._max_age_days > 0 and manifest.is_stale(month_key, self._max_age_days))
            )

            if needs_fetch:
                month_df = self._fetch_and_cache_month(symbol, month_key, manifest)
            else:
                month_df = self._read_month_cache(symbol, month_key, manifest)

            if month_df is not None and not month_df.empty:
                chunks.append(month_df)

        if not chunks:
            raise ValueError(f"No data found for {symbol} [{start} -> {end}].")

        df = pd.concat(chunks, ignore_index=True)
        df = _dedup_sort(df)

        # Slice to exact requested range
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
        mask = (df[DATETIME_COL] >= start_ts) & (df[DATETIME_COL] < end_ts)
        df = df.loc[mask].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"No data in [{start} -> {end}] after slicing.")

        logger.info("Loaded %d bars for %s [%s -> %s].", len(df), symbol, start, end)
        return df

    def load_tick_csv(
        self,
        path_pattern: str,
        symbol: str,
        datetime_col: str = DATETIME_COL,
        price_col: str = "price",
        volume_col: str = "quantity",
        cache_result: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load legacy tick CSV files and aggregate to 1-minute OHLCV bars.

        Designed for one-time migration from V1 tick data.
        Results are cached as monthly parquet chunks so subsequent
        calls use the fast parquet path instead of re-aggregating.

        CSV files must contain at minimum: a datetime/timestamp column,
        a price column, and a volume/quantity column.

        File naming convention for correct month detection:
            ticks_YYYY_MM.csv  ->  "2023_01"
            VN30F1M_YYYYMM.csv ->  "2023_01"

        Args:
            path_pattern: Path or glob, e.g. "data/ticks/ticks_2023_*.csv".
                          Files are sorted lexicographically - use YYYY_MM
                          naming convention to ensure chronological order.
            symbol:       Symbol used for cache directory.
            datetime_col: Timestamp column name in the CSV.
            price_col:    Last trade price column name.
            volume_col:   Volume / quantity column name.
            cache_result: Write aggregated 1min bars to monthly parquet cache.
            force_refresh: Re-aggregate even if parquet already exists.

        Returns:
            1-minute OHLCV DataFrame.
        """
        import glob as glob_mod

        raw_paths = sorted(glob_mod.glob(path_pattern))
        if not raw_paths:
            raise FileNotFoundError(f"No files match: {path_pattern!r}")

        logger.info("Aggregating %d tick CSV file(s) -> 1min...", len(raw_paths))
        manifest = self._get_manifest(symbol)
        all_chunks: list[pd.DataFrame] = []

        for csv_path in raw_paths:
            month_key = _month_key_from_path(csv_path)
            parquet_path = self._month_path(symbol, month_key)

            # Use cache if available and not forcing refresh
            if not force_refresh and parquet_path.exists() and manifest.is_cached(month_key):
                logger.debug("Tick cache hit: %s", month_key)
                cached = self._read_month_cache(symbol, month_key, manifest)
                if cached is not None:
                    all_chunks.append(cached)
                    continue

            # Aggregate ticks -> 1min
            logger.info("Aggregating: %s", csv_path)
            try:
                tick_df = pd.read_csv(csv_path, parse_dates=[datetime_col])
            except Exception as e:
                logger.error("Failed to read %s: %s - skipping.", csv_path, e)
                continue

            month_df = _aggregate_ticks_to_1min(
                tick_df,
                datetime_col=datetime_col,
                price_col=price_col,
                volume_col=volume_col,
            )
            if month_df.empty:
                logger.warning("No bars produced from %s.", csv_path)
                continue

            if cache_result:
                self._save_month_parquet(month_df, symbol, month_key, manifest, source="tick_csv")
            all_chunks.append(month_df)

        if not all_chunks:
            raise ValueError(f"No 1min data produced from {path_pattern!r}.")

        df = pd.concat(all_chunks, ignore_index=True)
        df = _dedup_sort(df)
        logger.info("Tick aggregation complete: %d 1min bars.", len(df))
        return df

    def invalidate_cache(
        self,
        symbol: str | None = None,
        month_key: str | None = None,
    ) -> int:
        """
        Delete cached parquet files and update the manifest.

        Args:
            symbol:    Delete only this symbol. None = all symbols.
            month_key: Delete only this month. None = all months for symbol.

        Returns:
            Number of parquet files deleted.
        """
        if symbol is None:
            count = 0
            for sym_dir in self._cache_root.iterdir():
                if sym_dir.is_dir():
                    count += self.invalidate_cache(sym_dir.name)
            return count

        manifest = self._get_manifest(symbol)

        if month_key is not None:
            path = self._month_path(symbol, month_key)
            if path.exists():
                path.unlink()
                manifest.invalidate(month_key)
                logger.info("Invalidated %s/%s.", symbol, month_key)
                return 1
            return 0

        count = 0
        for f in self._month_dir(symbol).glob("*.parquet"):
            f.unlink()
            count += 1
        manifest.invalidate()
        if count:
            logger.info("Invalidated %d months for %s.", count, symbol)
        return count

    def list_cached_months(self, symbol: str) -> list[str]:
        """Return sorted list of cached month keys for a symbol."""
        return sorted(self._get_manifest(symbol).cached_months)

    # --- Private: month cache ---

    def _fetch_and_cache_month(
        self,
        symbol: str,
        month_key: str,
        manifest: CacheManifest,
    ) -> pd.DataFrame | None:
        """
        Fetch one calendar month from DB, validate, and write to parquet.

        Implements intelligent incremental fetching for current month:
        - If month is current and has cached data with last_synced_timestamp,
          only fetches new data from that timestamp forward
        - Merges new data with existing cache and deduplicates
        - Falls back to full month fetch if incremental fetch fails
        - Past months always do full fetch (they should be cached already)

        Args:
            symbol: Contract symbol (e.g., "VN30F1M").
            month_key: Month identifier in "YYYY_MM" format.
            manifest: Cache manifest for tracking metadata.

        Returns:
            DataFrame with validated OHLCV data, or None if fetch/validation fails.

        Incremental Fetch Logic:
            1. Check if month is current and has last_synced_timestamp
            2. Load existing cached data
            3. Fetch only from last_synced_timestamp to month_end
            4. Merge and deduplicate: existing + new data
            5. Validate and save merged result

        Full Fetch Logic (fallback):
            1. Fetch entire month from month_start to month_end
            2. Validate and save

        Example:
            Current month is 2024-03, cache has data up to 2024-03-20T15:00:00
            -> Incremental fetch: 2024-03-20 to 2024-03-31
            -> Merge with existing data, deduplicate, save
        """
        month_start, month_end = _month_bounds(month_key)
        is_current = _is_current_month(month_key)

        # --- Incremental Fetch Path (Current Month Only) ---
        last_synced = manifest.get_last_synced_timestamp(month_key)
        existing_df = None

        if is_current and last_synced:
            # Try to load existing cache for incremental update
            existing_df = self._read_month_cache(symbol, month_key, manifest)
            if existing_df is not None and not existing_df.empty:
                # Extract date part from ISO timestamp for DB query using robust parsing
                fetch_start = pd.Timestamp(last_synced).strftime(
                    "%Y-%m-%d"
                )  # "2024-03-20T15:00:00" -> "2024-03-20"
                logger.info(
                    "Incremental fetch %s/%s from %s to %s...",
                    symbol,
                    month_key,
                    fetch_start,
                    month_end,
                )
                try:
                    # Fetch only new data since last sync
                    new_df = self._svc.fetch_ohlcv(
                        contract_name=symbol,
                        from_date=fetch_start,
                        to_date=month_end,
                        chunk_size_days=self._chunk_size,
                    )
                    if new_df is not None and not new_df.empty:
                        # Merge existing and new data, remove duplicates
                        df = pd.concat([existing_df, new_df], ignore_index=True)
                        df = _dedup_sort(df)  # Dedup by datetime, keep last
                    else:
                        # No new data available, return existing cache as-is
                        logger.info("No new data for %s/%s.", symbol, month_key)
                        return existing_df
                except Exception as e:
                    # Incremental fetch failed, fall back to full fetch
                    logger.warning(
                        "Incremental fetch failed for %s/%s: %s - falling back to full fetch",
                        symbol,
                        month_key,
                        e,
                    )
                    existing_df = None  # Clear flag to trigger full fetch below

        # --- Full Fetch Path ---
        if existing_df is None:
            logger.info("Fetching %s/%s from DB...", symbol, month_key)
            try:
                df = self._svc.fetch_ohlcv(
                    contract_name=symbol,
                    from_date=month_start,
                    to_date=month_end,
                    chunk_size_days=self._chunk_size,
                )
            except Exception as e:
                logger.error("DB fetch failed for %s/%s: %s", symbol, month_key, e)
                return None

            if df is None or df.empty:
                logger.warning("No data from DB for %s/%s.", symbol, month_key)
                return None

            df = _dedup_sort(df)

        # --- Validation ---
        result = self._validator.validate_ohlcv(df)

        if not result.is_valid:
            logger.error(
                "Validation failed for %s/%s: %s",
                symbol,
                month_key,
                result.summary(),
            )
            return None
        if result.warnings:
            logger.warning(
                "Data warnings for %s/%s: %s",
                symbol,
                month_key,
                result.summary(),
            )

        # --- Save and Return ---
        self._save_month_parquet(df, symbol, month_key, manifest)
        return df

    def _read_month_cache(
        self,
        symbol: str,
        month_key: str,
        manifest: CacheManifest,
    ) -> pd.DataFrame | None:
        """Read one month from parquet. Invalidates entry if corrupt or schema mismatch."""
        path = self._month_path(symbol, month_key)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning("Corrupt parquet %s: %s - refetching.", path.name, e)
            path.unlink(missing_ok=True)
            manifest.invalidate(month_key)
            return None

        schema_ok = self._validator.validate_schema(df, required_cols=_REQUIRED_COLS)
        if not schema_ok.is_valid:
            logger.warning("Schema mismatch in %s - refetching.", path.name)
            path.unlink(missing_ok=True)
            manifest.invalidate(month_key)
            return None

        return df

    def _save_month_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        month_key: str,
        manifest: CacheManifest,
        source: str = "database",
    ) -> None:
        """
        Write one month of 1min bars to parquet and update manifest metadata.

        Automatically extracts and records the last timestamp from the data
        to enable incremental fetching on subsequent loads.

        Args:
            df: DataFrame with OHLCV data to cache.
            symbol: Contract symbol for directory structure.
            month_key: Month identifier in "YYYY_MM" format.
            manifest: Cache manifest to update with metadata.
            source: Data source identifier ("database", "tick_csv", etc.).

        Side Effects:
            - Creates parquet file at: data/cache/{symbol}/1min/{month_key}.parquet
            - Updates manifest.json with row count, timestamp, and metadata
            - Logs success/failure messages

        Manifest Update:
            - row_count: Number of bars saved
            - created_at: Current UTC timestamp
            - source: Where data came from
            - complete: Auto-detected (past=true, current=false)
            - last_synced_timestamp: Latest datetime from the data (ISO format)
              -> Used for next incremental fetch
        """
        path = self._month_path(symbol, month_key)
        try:
            # Ensure cache directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write data to parquet
            df.to_parquet(path, index=False)

            # --- Extract Last Timestamp for Incremental Updates ---
            # This timestamp will be used as the starting point for the next
            # incremental fetch, avoiding re-downloading existing data
            last_timestamp = None
            if not df.empty and DATETIME_COL in df.columns:
                df_sorted = df.sort_values(DATETIME_COL)
                last_ts = df_sorted[DATETIME_COL].iloc[-1]
                if pd.notna(last_ts):
                    # Convert to ISO format: "2024-03-23T15:30:00+00:00"
                    last_timestamp = pd.Timestamp(last_ts).isoformat()

            # --- Update Manifest ---
            manifest.record(
                month_key,
                row_count=len(df),
                source=source,
                last_synced_timestamp=last_timestamp,
            )
            logger.debug("Cached %d bars -> %s", len(df), path.name)
        except Exception as e:
            logger.warning("Failed to save parquet %s: %s", path.name, e)

    # --- Private: path helpers ---

    def _month_dir(self, symbol: str) -> Path:
        return self._cache_root / symbol / "1min"

    def _month_path(self, symbol: str, month_key: str) -> Path:
        return self._month_dir(symbol) / f"{month_key}.parquet"

    def _get_manifest(self, symbol: str) -> CacheManifest:
        return CacheManifest(self._cache_root / symbol / "manifest.json", symbol)


# --- Pure functions ---


def _months_in_range(start: str, end: str) -> list[str]:
    """Return list of month keys ("YYYY_MM") covering the full [start, end] range."""
    s = pd.Timestamp(start).replace(day=1)
    e = pd.Timestamp(end)
    months = []
    current = s
    while current <= e:
        months.append(current.strftime("%Y_%m"))
        # Advance to first day of next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    return months


def _month_bounds(month_key: str) -> tuple[str, str]:
    """Return (first_day, last_day) date strings for a "YYYY_MM" month key."""
    year, month = int(month_key[:4]), int(month_key[5:])
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _month_key_from_path(path: str) -> str:
    """
    Infer "YYYY_MM" month key from a CSV filename.

    Recognises patterns: YYYY_MM, YYYY-MM, YYYYMM anywhere in the stem.
    Falls back to the file's modification time if no pattern is found.
    """
    import re

    stem = Path(path).stem

    for pattern in (r"(\d{4})[_-](\d{2})", r"(\d{4})(\d{2})"):
        m = re.search(pattern, stem)
        if m:
            return f"{m.group(1)}_{m.group(2)}"

    mtime = Path(path).stat().st_mtime
    dt = datetime.fromtimestamp(mtime, tz=UTC)
    logger.warning("Cannot infer month from %r - using mtime %s.", stem, dt.strftime("%Y_%m"))
    return dt.strftime("%Y_%m")


def _aggregate_ticks_to_1min(
    df: pd.DataFrame,
    datetime_col: str = "timestamp",
    price_col: str = "price",
    volume_col: str = "quantity",
) -> pd.DataFrame:
    """
    Aggregate a tick DataFrame into 1-minute OHLCV bars.

    Handles cumulative volume correctly by computing diff within each trading day.

    Args:
        df:           Raw tick DataFrame.
        datetime_col: Column containing tick timestamps.
        price_col:    Column with last-trade price.
        volume_col:   Column with trade volume / quantity (cumulative within day).

    Returns:
        1-minute OHLCV DataFrame with columns:
        datetime, open, high, low, close, volume.
        Bars are sorted ascending by datetime.

    Note:
        Volume is assumed to be cumulative within each trading day and resets
        at day boundaries. This function computes the diff to get actual traded volume.
    """
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col)

    # --- Compute volume diff (handle cumulative) ---
    # Volume is cumulative within each day, resets at day boundaries
    if volume_col in df.columns:
        qty = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
        day_keys = df[datetime_col].dt.normalize()  # Group by date

        # Compute diff within each day
        tick_vols = qty.groupby(day_keys).diff()

        # First tick of each day: use the cumulative value as-is
        first_tick_of_day = tick_vols.isna()
        tick_vols.loc[first_tick_of_day] = qty.loc[first_tick_of_day]

        # Clip negative values (can happen if data is corrupted)
        tick_vols = tick_vols.clip(lower=0)

        df["_volume_diff"] = tick_vols
    else:
        df["_volume_diff"] = 0.0

    # --- Aggregate to 1min bars ---
    df["_bar"] = df[datetime_col].dt.floor("1min")

    ohlcv = (
        df.groupby("_bar")
        .agg(
            open=(price_col, "first"),
            high=(price_col, "max"),
            low=(price_col, "min"),
            close=(price_col, "last"),
            volume=("_volume_diff", "sum"),  # Sum the diff, not cumulative
        )
        .reset_index()
        .rename(columns={"_bar": DATETIME_COL})
    )
    return ohlcv.sort_values(DATETIME_COL, ignore_index=True)


def _dedup_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows by datetime (keep last) and sort ascending."""
    return df.drop_duplicates(subset=[DATETIME_COL], keep="last").sort_values(
        DATETIME_COL, ignore_index=True
    )


def _is_current_month(month_key: str) -> bool:
    """
    Return True if month_key matches the current calendar month.

    Used to decide whether a cached month is "complete":
    - Past months:    complete=True  -> cache hit forever, never refetch
    - Current month:  complete=False -> always refetch to get today's bars

    Examples (if today is 2026-03-23):
        _is_current_month("2026_03") -> True
        _is_current_month("2026_02") -> False
        _is_current_month("2025_12") -> False
    """
    return month_key == datetime.now(UTC).strftime("%Y_%m")
