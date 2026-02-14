"""
Data loader module for loading market data from database or CSV files.
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

    Supports loading from:
    - PostgreSQL database via DataService
    - CSV files
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
        """
        Generate a consistent cache filename.

        Args:
            contract_name (str): Type of contract
            start_date (str): Start date string
            end_date (str): End date string

        Returns:
            Path: Path to the cache file
        """
        safe_start = pd.to_datetime(start_date).strftime("%Y%m%d_%H%M%S")
        safe_end = pd.to_datetime(end_date).strftime("%Y%m%d_%H%M%S")
        filename = f"{contract_name}_{safe_start}_{safe_end}.parquet"
        return self.cache_dir / filename

    def _load_sample(
        self,
        contract_name: str,
        start_date: str,
        end_date: str,
        subdir: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Internal helper to load sample data (cache or CSV).

        Args:
            contract_name (str): Type of contract to load
            start_date (str): Start date for filtering
            end_date (str): End date for filtering
            subdir (str): Subdirectory ('is' or 'os')
            use_cache (bool): Whether to use cached data if available

        Returns:
            pd.DataFrame: Loaded sample data
        """
        cache_path = self._get_cache_path(contract_name, start_date, end_date)

        if use_cache and cache_path.exists():
            logger.info("Loading %s cache: %s", subdir, cache_path)
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning("Cache load failed: %s", e)

        # Fallback to CSV
        filepath = self.data_dir / subdir / f"{contract_name}_data.csv"
        df = self.load_csv(filepath)

        # Enforce datetime format
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        # Filter to correct date range
        mask = (df["datetime"] >= pd.to_datetime(start_date)) & (
            df["datetime"] <= pd.to_datetime(end_date)
        )
        df = df.loc[mask].reset_index(drop=True)

        # Save to cache
        if use_cache and not df.empty:
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                logger.error("Failed to create cache: %s", e)

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
    ) -> pd.DataFrame:
        """
        Fetch data directly from the database and optionally save to CSV/Parquet.

        Args:
            contract_name (str): Type of contract to fetch
            start_date (str): Start date for data retrieval
            end_date (Optional[str]): End date for data retrieval
            save_path (Optional[Union[str, Path]]): Path to save CSV file
            use_cache (bool): Whether to use cached data if available

        Returns:
            pd.DataFrame: DataFrame with fetched data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Define cache path
        cache_path = self._get_cache_path(contract_name, start_date, end_date)

        # Check cache
        if use_cache and cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning("Failed to load cache: %s. Fetching from database...", e)

        logger.info(
            "Fetching data for %s (%s to %s)...", contract_name, start_date, end_date
        )

        try:
            df = fetch_and_merge_data(contract_name, start_date, end_date)
        except Exception as e:
            logger.error("Error fetching data from database: %s", e, exc_info=True)
            return pd.DataFrame()

        if df.empty:
            logger.warning(
                "No data found for %s (%s-%s)", contract_name, start_date, end_date
            )
            return df

        # Parse datetime if not already
        if "datetime" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
                df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

        # Save to Cache
        if use_cache:
            try:
                df.to_parquet(cache_path)
            except Exception as e:
                logger.error("Failed to cache data: %s", e)

        # Save to CSV if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)

        return df
