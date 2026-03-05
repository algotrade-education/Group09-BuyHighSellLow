"""
Data loader module for loading market data from database or CSV files.
Supports single-file and chunked CSV saving (split by month or year).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config.config import (
    CACHE_DIR,
    DATA_DIR,
    IS_SAMPLE_END,
    IS_SAMPLE_START,
    OUT_SAMPLE_END,
    OUT_SAMPLE_START,
)
from src.database.data_service import fetch_and_merge_data

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and validates market data from various sources.

    Directory layout
    ----------------
    data/
        <contract>_YYYYMM.csv        raw monthly chunk files (source data)
        is/
            <contract>_data.parquet  filtered IS parquet (auto-generated)
        os/
            <contract>_data.parquet  filtered OS parquet (auto-generated)
    cache/
        <contract>_<start>_<end>.parquet  full-range fetch cache
    """

    def __init__(self, data_dir: str = DATA_DIR, cache_dir: str = CACHE_DIR):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Directory containing data files
            cache_dir: Directory for caching files
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary data directories if they don't exist."""
        directories = [
            self.data_dir,
            self.data_dir / "is",  # In-sample
            self.data_dir / "os",  # Out-of-sample
            self.cache_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(
        self, contract_name: str, start_date: str, end_date: str
    ) -> Path:
        """Generate a full-range fetch cache filename in cache/."""
        safe_start = pd.to_datetime(start_date).strftime("%Y%m%d_%H%M%S")
        safe_end = pd.to_datetime(end_date).strftime("%Y%m%d_%H%M%S")
        filename = f"{contract_name}_{safe_start}_{safe_end}.parquet"
        return self.cache_dir / filename

    def _get_sample_parquet_path(self, contract_name: str, subdir: str) -> Path:
        """Return the parquet path inside data/is/ or data/os/."""
        return self.data_dir / subdir / f"{contract_name}_data.parquet"

    @staticmethod
    def _normalize_datetime(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
        """
        Parse, coerce, and sort a DataFrame's datetime column in-place.

        - Converts *col* to ``datetime64`` (coerce invalid entries to NaT).
        - Drops rows where *col* is NaT.
        - Sorts by *col* and resets the index.

        Args:
            df:  DataFrame to normalise.
            col: Name of the datetime column (default ``'datetime'``).

        Returns:
            Cleaned, sorted DataFrame.
        """
        if col not in df.columns:
            return df
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.dropna(subset=[col])
        return df.sort_values(col).reset_index(drop=True)

    def _iter_periods(
        self,
        start_date: str,
        end_date: str,
        chunk_by: str,
    ):
        """
        Yield ``(suffix, period_start, period_end)`` for each chunk period.

        Args:
            start_date: Overall start date.
            end_date: Overall end date.
            chunk_by: ``'month'`` or ``'year'``.

        Yields:
            tuple[str, pd.Timestamp, pd.Timestamp]
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        if chunk_by == "month":
            cur = start.to_period("M").to_timestamp()
            while cur <= end:
                p_end = (
                    (cur + pd.offsets.MonthEnd(0)).normalize()
                    + pd.Timedelta(hours=23, minutes=59, seconds=59)
                )
                yield cur.strftime("%Y_%m"), cur, min(p_end, end)
                cur = (cur + pd.offsets.MonthBegin(1)).normalize()
        else:  # year
            cur = start.to_period("Y").to_timestamp()
            while cur <= end:
                p_end = pd.Timestamp(cur.year, 12, 31, 23, 59, 59)
                yield cur.strftime("%Y"), cur, min(p_end, end)
                cur = pd.Timestamp(cur.year + 1, 1, 1)

    def _fetch_and_save_period(
        self,
        contract_name: str,
        s_str: str,
        e_str: str,
        csv_path: Path,
        idx: int,
        total: int,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch one time period from the DB and immediately save to *csv_path*.

        Returns the DataFrame on success, ``None`` on empty result or error.
        """
        logger.info(
            "[%d/%d] Fetching %s to %s ...", idx, total, s_str, e_str
        )
        try:
            df = fetch_and_merge_data(contract_name, s_str, e_str)
        except Exception as exc:
            logger.error("[%d/%d] Fetch failed: %s", idx, total, exc)
            return None

        if df.empty:
            logger.warning("[%d/%d] No data for %s to %s", idx, total, s_str, e_str)
            return None

        df = self._normalize_datetime(df)

        df.to_csv(csv_path, index=False)
        logger.info("[%d/%d] Saved %s  (%d rows)", idx, total, csv_path.name, len(df))
        return df

    def _load_sample(
        self,
        contract_name: str,
        start_date: str,
        end_date: str,
        subdir: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Internal helper to load sample data.

        Priority:
        1. Parquet in data/is/ or data/os/  (fast)
        2. Monthly CSV chunks in data/ root  (VN30F1M_YYYYMM.csv)
        3. Legacy single CSV in data/is/ or data/os/  (backward compat)

        After reading from CSV the filtered result is saved as a parquet
        in data/is/ or data/os/ for fast subsequent loads.

        Args:
            contract_name (str): Type of contract to load
            start_date (str): Start date for filtering
            end_date (str): End date for filtering
            subdir (str): Subdirectory ('is' or 'os')
            use_cache (bool): Whether to use cached parquet if available

        Returns:
            pd.DataFrame: Loaded sample data
        """
        parquet_path = self._get_sample_parquet_path(contract_name, subdir)

        if use_cache and parquet_path.exists():
            logger.info("Loading %s parquet: %s", subdir, parquet_path)
            try:
                return pd.read_parquet(parquet_path)
            except Exception as e:
                logger.warning("Parquet load failed: %s", e)

        # Try CSV chunks flat in data/ root  (e.g. VN30F1M_202301.csv)
        chunk_files = sorted(self.data_dir.glob(f"{contract_name}_*.csv"))
        if chunk_files:
            logger.info(
                "Loading %d chunk file(s) from %s", len(chunk_files), self.data_dir
            )
            df = pd.concat(
                [pd.read_csv(p) for p in chunk_files], ignore_index=True
            )
        else:
            raise FileNotFoundError(f"No data found for {contract_name}")

        # Normalize datetime, filter to date range, save parquet
        df = self._normalize_datetime(df)

        # Filter to correct date range
        mask = (df["datetime"] >= pd.to_datetime(start_date)) & (
            df["datetime"] <= pd.to_datetime(end_date)
        )
        df = df.loc[mask].reset_index(drop=True)

        # Save filtered result as parquet in is/os
        if use_cache and not df.empty:
            try:
                df.to_parquet(parquet_path)
                logger.info("Saved %s parquet: %s", subdir, parquet_path)
            except Exception as e:
                logger.error("Failed to save %s parquet: %s", subdir, e)

        return df

    def load_csv(
        self,
        filepath: Union[str, Path],
        parse_dates: bool = True,
        date_column: str = "datetime",
    ) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath (Union[str, Path]): Path to the CSV file
            parse_dates (bool): Whether to parse date columns
            date_column (str): Name of the date column

        Returns:
            pd.DataFrame: DataFrame with loaded data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            df = pd.read_csv(filepath)

            if parse_dates and date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.sort_values(date_column).reset_index(drop=True)

            return df
        except Exception as e:
            logger.error("Error loading CSV %s: %s", filepath, e)
            raise

    def load_in_sample(
        self,
        contract_name: str = "VN30F1M",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load in-sample data for backtesting.

        Args:
            contract_name (str): Type of contract to load
            use_cache (bool): Whether to use cached data if available

        Returns:
            pd.DataFrame: In-sample data
        """
        return self._load_sample(
            contract_name, IS_SAMPLE_START, IS_SAMPLE_END, "is", use_cache
        )

    def load_out_of_sample(
        self,
        contract_name: str = "VN30F1M",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load out-of-sample data for validation.

        Args:
            contract_name (str): Type of contract to load
            use_cache (bool): Whether to use cached data if available

        Returns:
            pd.DataFrame: Out-of-sample data
        """
        return self._load_sample(
            contract_name, OUT_SAMPLE_START, OUT_SAMPLE_END, "os", use_cache
        )

    def fetch_from_database(
        self,
        contract_name: str = "VN30F1M",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        chunk_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch data from the database and optionally persist to disk.

        When *chunk_by* is set the fetch is **incremental**: each period is
        queried, its CSV written to *data/* immediately, and already-saved
        periods are skipped automatically (resumable).
        When *chunk_by* is ``None`` the full range is fetched at once and
        optionally written to *save_path*.

        Args:
            contract_name: Contract symbol.
            start_date: Start of the date range.
            end_date: End of the date range (defaults to today).
            save_path: Single-file CSV path (only used when *chunk_by* is None).
            use_cache: Load from / save to full-range Parquet cache.
            chunk_by: ``'month'`` or ``'year'`` for incremental fetching.

        Returns:
            The full concatenated DataFrame.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if chunk_by:
            if chunk_by not in ("month", "year"):
                raise ValueError(f"chunk_by must be 'month' or 'year', got {chunk_by!r}")
            return self._fetch_chunked(contract_name, start_date, end_date, chunk_by)

        return self._fetch_single(contract_name, start_date, end_date, save_path, use_cache)

    # ------------------------------------------------------------------
    # Internal fetch implementations
    # ------------------------------------------------------------------

    def _fetch_chunked(
        self,
        contract_name: str,
        start_date: str,
        end_date: str,
        chunk_by: str,
    ) -> pd.DataFrame:
        """Incremental fetch: one DB call per period, CSV saved immediately."""
        periods = list(self._iter_periods(start_date, end_date, chunk_by))
        total = len(periods)
        all_parts: list[pd.DataFrame] = []

        for idx, (suffix, p_start, p_end) in enumerate(periods, 1):
            csv_path = self.data_dir / f"{contract_name}_{suffix}.csv"
            if csv_path.exists():
                logger.info("[%d/%d] Skipping %s (already exists)", idx, total, csv_path.name)
                all_parts.append(pd.read_csv(csv_path))
                continue

            s_str = p_start.strftime("%Y-%m-%d %H:%M:%S")
            e_str = p_end.strftime("%Y-%m-%d %H:%M:%S")
            df_period = self._fetch_and_save_period(
                contract_name, s_str, e_str, csv_path, idx, total
            )
            if df_period is not None:
                all_parts.append(df_period)

        if not all_parts:
            return pd.DataFrame()

        df = pd.concat(all_parts, ignore_index=True)
        logger.info("Fetch complete: %d rows across %d period(s).", len(df), total)
        return df

    def _fetch_single(
        self,
        contract_name: str,
        start_date: str,
        end_date: str,
        save_path: Optional[Union[str, Path]],
        use_cache: bool,
    ) -> pd.DataFrame:
        """Single-range fetch with optional Parquet cache and CSV export."""
        cache_path = self._get_cache_path(contract_name, start_date, end_date)

        if use_cache and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning("Cache load failed: %s. Fetching from database...", e)

        logger.info(
            "Fetching %s (%s to %s)...", contract_name, start_date, end_date
        )
        try:
            df = fetch_and_merge_data(contract_name, start_date, end_date)
        except Exception as e:
            logger.error("Fetch failed: %s", e, exc_info=True)
            return pd.DataFrame()

        if df.empty:
            logger.warning("No data for %s (%s to %s)", contract_name, start_date, end_date)
            return df

        df = self._normalize_datetime(df)

        if use_cache:
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                logger.error("Failed to cache data: %s", e)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info("Saved CSV to %s", save_path)

        return df
