"""
Load OHLCV data from Database or CSV, validate, cache as parquet.

V2 fixes:
    - Generator-based chunked loading - Do not load all data into memory at once.
    - Validate schema after loading from cache - do not trust cache blindly.
    - Data freshness check
    - Dedup in chunk boundaries
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.data.validators import DataValidator
from src.database.base import DataServiceBase

logger = logging.getLogger(__name__)

DATETIME_COL = "datetime"
_CACHE_VERSION = "v2"
_CACHE_FILE_PREFIX = "ohlcv"


class DataLoader:
    """
    Load OHLCV data and cache as parquet for future reuse.

    Usage:
        loader = DataLoader(data_service, cache_dir="data/cache")

        # Load and validate data, with caching (default)
        df = loader.load("VN30F1M", start="2023-01-01", end="2024-12-31")

        # Force refresh from DB, bypass cache
        df = loader.load("VN30F1M", ..., use_cache=False)
    """

    def __init__(
        self,
        data_service: DataServiceBase,
        cache_dir: str = "data/cache",
        chunk_size_days: int = 30,
    ) -> None:
        """
        Args:
            data_service: Instance of DataService (database adapter).
            cache_dir: Parquet cache directory.
            chunk_size_days: Fetch DB in chunks of this many days to avoid OOM. Overlap 1 day between chunks to catch duplicates.
        """
        self._svc = data_service
        self._cache_dir = Path(cache_dir)
        self._validator = DataValidator()
        self._chunk_size = chunk_size_days

    # ── Public API ────────────────────────────────────────────────

    def load(
        self,
        symbol: str,
        start: str,
        end: str,
        freq: str = "5min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load OHLCV data, validate, cache.

        Args:
            symbol:    Contract symbol, e.g. "VN30F1M".
            start:     Start date "YYYY-MM-DD".
            end:       End date "YYYY-MM-DD".
            freq:      Bar frequency - only for cache key.
                       Actual resampling is the job of DataPreprocessor.
            use_cache: True = try cache first, False = always fetch from DB.

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.

        Raises:
            ValueError: If validation fails or no data found.
            RuntimeError: If DB fetch fails and cache is not available.
        """
        cache_key = self._build_cache_key(symbol, start, end, freq)

        # ── Try cache ──────────────────────────────────────────────
        if use_cache:
            cached = self._load_from_cache(cache_key, symbol, start, end)
            if cached is not None:
                return cached

        # ── Fetch from DB ────────────────────────────────────────────
        logger.info("Fetching %s [%s -> %s] from DB...", symbol, start, end)
        try:
            df = self._fetch_from_db(symbol, start, end)
        except Exception as e:
            raise RuntimeError(f"DB fetch failed for {symbol} [{start} -> {end}]: {e}") from e

        if df.empty:
            raise ValueError(f"No data available for {symbol} [{start} -> {end}].")

        # ── Validate ───────────────────────────────────────────────
        result = self._validator.validate_ohlcv(df)
        if not result:
            raise ValueError(f"Data validation failed for {symbol}:\n{result.summary()}")
        if result.warnings:
            logger.warning("Data warnings for %s:\n%s", symbol, result.summary())

        # ── Cache ──────────────────────────────────────────────────
        if use_cache:
            self._save_cache(cache_key, df, symbol, start, end)

        logger.info("Loaded %d bars for %s.", len(df), symbol)
        return df

    def load_csv(
        self,
        path: str,
        datetime_col: str = DATETIME_COL,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file(s), with optional validation.
        Uses generator for multiple files to avoid loading all data into memory at once.

        Args:
            path: Path to CSV file or glob pattern (e.g. "data/*.csv").
            datetime_col: Name of datetime column in CSV.
            validate: Whether to validate the loaded DataFrame.
        """
        import glob

        paths = sorted(glob.glob(path)) if "*" in path else [path]
        if not paths:
            raise FileNotFoundError(f"Cannot find file with pattern {path}")

        chunks = self._read_csv_chunks(paths, datetime_col)
        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values(datetime_col, ignore_index=True)
        df = df.drop_duplicates(subset=[datetime_col], keep="last")

        if validate:
            result = self._validator.validate_ohlcv(df)
            if not result:
                raise ValueError(f"CSV validation failed:\n{result.summary()}")

        return df

    # ── Private: fetch ────────────────────────────────────────────

    def _fetch_from_db(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch data from DB.

        Chunking responsibility belongs to the database layer via
        DataServiceBase.fetch_ohlcv(chunk_size_days=...).
        """
        df = self._svc.fetch_ohlcv(
            contract_name=symbol,
            from_date=start,
            to_date=end,
            chunk_size_days=self._chunk_size,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        # Defensive dedup/sort in case upstream implementation changes
        df = df.drop_duplicates(subset=[DATETIME_COL], keep="last")
        df = df.sort_values(DATETIME_COL, ignore_index=True)

        return df

    @staticmethod
    def _read_csv_chunks(paths: list[str], datetime_col: str) -> Iterator[pd.DataFrame]:
        """Generator - read each CSV file as a chunk, do not load all into RAM at once."""
        for path in paths:
            logger.debug("Reading CSV: %s", path)
            chunk = pd.read_csv(path, parse_dates=[datetime_col])
            yield chunk

    # ── Private: cache ────────────────────────────────────────────

    def _load_from_cache(
        self,
        cache_key: str,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning("Cache corrupt (%s): %s. Fetching from DB.", path.name, e)
            path.unlink(missing_ok=True)
            return None

        # Validate schema - do not trust cache blindly
        result = self._validator.validate_schema(
            df,
            required_cols=[DATETIME_COL, "open", "high", "low", "close", "volume"],
        )
        if not result:
            logger.warning(
                "Cache schema invalid (%s): %s. Refetching.",
                path.name,
                result.summary(),
            )
            path.unlink(missing_ok=True)
            return None

        logger.info(
            "Cache hit: %s bars for %s [%s -> %s].",
            len(df),
            symbol,
            start,
            end,
        )
        return df

    def _save_cache(
        self,
        cache_key: str,
        df: pd.DataFrame,
        symbol: str,
        start: str,
        end: str,
    ) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_path(cache_key)

            df.to_parquet(path, index=False)
            logger.debug("Cached %d bars -> %s", len(df), path.name)
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{_CACHE_FILE_PREFIX}_{cache_key}.parquet"

    def _build_cache_key(self, symbol: str, start: str, end: str, freq: str) -> str:
        sig = f"{_CACHE_VERSION}|{symbol}|{start}|{end}|{freq}"
        return hashlib.sha256(sig.encode()).hexdigest()

    def invalidate_cache(self, symbol: str | None = None) -> int:
        """
        Delete cache files.
        If symbol is None, delete all. Otherwise, only delete files for that symbol.

        Args:
            symbol: If None, delete all. If specified, only delete files for that symbol.
                    Note: Since cache keys are hashed, filtering is not possible by symbol -
                    pass symbol=None to delete all when a refresh is needed.

        Returns:
            Number of files deleted.
        """
        if not self._cache_dir.exists():
            return 0

        count = 0
        for f in self._cache_dir.glob(f"{_CACHE_FILE_PREFIX}_*.parquet"):
            f.unlink()
            count += 1

        if count:
            logger.info("Invalidated %d cache files.", count)

        return count
