"""
Abstract interface for Data Service.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from types import TracebackType

import pandas as pd

logger = logging.getLogger(__name__)


class DataServiceBase(ABC):
    """
    Interface for market data service.

    All methods that return DataFrame with concrete schema.
    Returns empty DataFrame if no data, never None.
    Caller will validate schema and content after loading, so implementations can be lenient in what they return.

    Tick data schema (matched):
        datetime    : datetime64[ns]
        tickersymbol: str
        price       : float64
        quantity    : float64

    OHLCV schema (from fetch_ohlcv):
        datetime: datetime64[ns]
        open    : float64
        high    : float64
        low     : float64
        close   : float64
        volume  : float64
    """

    # --- Tick Data ---
    @abstractmethod
    def get_matched_data(
        self,
        contract_name: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Returns tick data for given contract and date range.

        Args:
            contract_name: e.g. "VN30F1M", "VN30F1Q", "VN30F1Y"
            from_date: Inclusive lower bound in "YYYY-MM-DD" format
            to_date: Exclusive upper bound in "YYYY-MM-DD" format

        Returns:
            DataFrame with tick data schema, sorted by datetime ascending.
            Empty DataFrame if no data.
        """
        ...

    @abstractmethod
    def get_matched_data_in_range(
        self,
        contract_name: str,
        from_datetime: datetime,
        to_datetime: datetime,
    ) -> pd.DataFrame:
        """
        Same as get_matched_data but with datetime arguments instead of string.
        Used for fetching data in bucket bars.

        Args:
            contract_name: e.g. "VN30F1M", "VN30F1Q", "VN30F1Y"
            from_datetime: Inclusive lower bound
            to_datetime: Exclusive upper bound

        Returns:
            DataFrame with tick data schema, sorted by datetime ascending.
            Empty DataFrame if no data.
        """
        ...

    @abstractmethod
    def get_last_matched_before(
        self,
        contract_name: str,
        before_datetime: datetime,
    ) -> pd.DataFrame:
        """
        Return the last tick data before given datetime.
        Used for backfilling missing data at the start of a date range.

        Args:
            contract_name: e.g. "VN30F1M", "VN30F1Q", "VN30F1Y"
            before_datetime: Exclusive upper bound in "YYYY-MM-DD HH:MM:SS" format

        Returns:
            DataFrame with tick data schema, sorted by datetime ascending.
            Empty DataFrame if no data.
        """
        ...

    # --- Reference Data ---

    @abstractmethod
    def get_close_data(
        self,
        contract_name: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Returns daily close price for given contract and date range.

        Args:
            contract_name: e.g. "VN30F1M", "VN30F1Q", "VN30F1Y"
            from_date: Inclusive lower bound in "YYYY-MM-DD" format
            to_date: Exclusive upper bound in "YYYY-MM-DD" format

        Returns:
            DataFrame with columns ["datetime", "close"], sorted by datetime ascending.
            Empty DataFrame if no data.
        """
        ...

    @abstractmethod
    def get_bid_ask_data(
        self,
        contract_name: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Returns daily bid and ask price for given contract and date range.

        Args:
            contract_name: e.g. "VN30F1M", "VN30F1Q", "VN30F1Y"
            from_date: Inclusive lower bound in "YYYY-MM-DD" format
            to_date: Exclusive upper bound in "YYYY-MM-DD" format

        Returns:
            DataFrame with columns ["datetime", "bid", "ask"], sorted by datetime ascending.
            Empty DataFrame if no data.
        """
        ...

    # --- OHLCV Data ---
    def fetch_ohlcv(
        self,
        contract_name: str,
        from_date: str,
        to_date: str,
        chunk_size_days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch tick data into chunks and aggregate into OHLCV.

        Args:
            contract_name: e.g. "VN30F1M", "VN30F1Q", "VN30F1Y"
            from_date: Inclusive lower bound in "YYYY-MM-DD" format
            to_date: Exclusive upper bound in "YYYY-MM-DD" format
            chunk_size_days: Number of days to fetch per chunk. Default 30.

        Returns:
            DataFrame with OHLCV schema, sorted by datetime ascending.
            Empty DataFrame if no data.
        """
        chunks = list(self._fetch_tick_chunks(contract_name, from_date, to_date, chunk_size_days))
        if not chunks:
            return pd.DataFrame()

        ticks = pd.concat(chunks, ignore_index=True)
        # Drop only exact duplicates to avoid losing legitimate trades on the same timestamp
        ticks = ticks.drop_duplicates(keep="last")
        ticks = ticks.sort_values("datetime").reset_index(drop=True)

        return self._aggregate_to_ohlcv(ticks)

    def fetch_bucket_bar(
        self,
        contract_name: str,
        bucket_start: datetime,
        bucket_end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch one bar of tick data between bucket_start (inclusive) and bucket_end (exclusive).

        Returns:
            DataFrame with OHLCV schema, sorted by datetime ascending.
            Empty DataFrame if no data.
        """
        ticks = self.get_matched_data_in_range(
            contract_name=contract_name,
            from_datetime=bucket_start,
            to_datetime=bucket_end,
        )
        if ticks.empty:
            return pd.DataFrame()

        ticks["datetime"] = pd.to_datetime(ticks["datetime"])
        ticks = ticks.sort_values("datetime").reset_index(drop=True)
        ticks["price"] = ticks["price"].astype(float)

        volume = self._calculate_volume(ticks, contract_name, bucket_start)

        return pd.DataFrame(
            [
                {
                    "datetime": bucket_start,
                    "open": float(ticks["price"].iloc[0]),
                    "high": float(ticks["price"].max()),
                    "low": float(ticks["price"].min()),
                    "close": float(ticks["price"].iloc[-1]),
                    "volume": volume,
                    "rows": len(ticks),
                }
            ]
        )

    # --- Lifecycle ---

    def close(self) -> None:
        """Close any open connections or resources."""
        return None

    def __enter__(self) -> DataServiceBase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- Private Helpers ---

    def _fetch_tick_chunks(
        self,
        contract_name: str,
        from_date: str,
        to_date: str,
        chunk_size_days: int = 30,
    ) -> Iterator[pd.DataFrame]:
        """
        Generator - yield tick data in chunks of chunk_size_days until the whole date range is covered.
        """

        start = pd.Timestamp(from_date)
        end = pd.Timestamp(to_date)
        current = start

        while current <= end:
            chunk_end = min(current + pd.Timedelta(days=chunk_size_days), end)
            s = current.strftime("%Y-%m-%d")
            e = chunk_end.strftime("%Y-%m-%d")

            logger.debug("Fetching tick data for %s from %s to %s", contract_name, s, e)
            try:
                chunk = self.get_matched_data(
                    contract_name=contract_name,
                    from_date=s,
                    to_date=e,
                )
                if not chunk.empty:
                    yield chunk
            except Exception as ex:
                logger.error(
                    "Error fetching tick data for %s from %s to %s: %s", contract_name, s, e, ex
                )
                raise

            current = chunk_end + pd.Timedelta(days=1)

    def _aggregate_to_ohlcv(self, ticks: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tick data to OHLCV on 1-minute frequency."""
        if ticks.empty:
            return pd.DataFrame()

        ticks = ticks.copy()
        ticks["datetime"] = pd.to_datetime(ticks["datetime"])
        ticks["price"] = ticks["price"].astype(float)
        ticks = ticks.set_index("datetime").sort_index()

        ohlcv: pd.DataFrame = ticks.resample("1min").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
        )

        if "quantity" in ticks.columns:
            qty = pd.to_numeric(ticks["quantity"], errors="coerce").fillna(0)

            dt_index = pd.DatetimeIndex(ticks.index)

            # quantity is cumulative within each trading day; it resets when a new day starts.
            # Therefore diff must also reset at day boundaries.
            day_keys = dt_index.normalize()  # Group by date, ignore time
            tick_vols = qty.groupby(day_keys).diff()
            first_tick_of_day = tick_vols.isna()

            tick_vols.loc[first_tick_of_day] = qty.loc[first_tick_of_day]
            tick_vols = tick_vols.clip(lower=0)

            vol = tick_vols.resample("1min").sum()
            ohlcv["volume"] = vol
        else:
            ohlcv["volume"] = 0.0

        return ohlcv.dropna(subset=["close"]).reset_index()

    def _calculate_volume(
        self,
        ticks: pd.DataFrame,
        contract_name: str,
        bucket_start: datetime,
    ) -> float:
        """
        Calculate volume for one bucket from tick quantity (cumulated)
        """
        if "quantity" not in ticks.columns:
            return 0.0

        qty_series = pd.to_numeric(ticks["quantity"], errors="coerce").fillna(0)

        # Get the quantity of the last tick before bucket_start to calculate volume in the bucket
        prev_df = self.get_last_matched_before(
            contract_name=contract_name, before_datetime=bucket_start
        )
        prev_qty: float | None = None
        if not prev_df.empty and "quantity" in prev_df.columns:
            prev_qty = pd.to_numeric(prev_df["quantity"], errors="coerce").iloc[0]
            if pd.isna(prev_qty):
                prev_qty = None

        if prev_qty is not None:
            full_qty = pd.concat([pd.Series([prev_qty]), qty_series], ignore_index=True)
            diffs = full_qty.diff().iloc[1:]  # First diff is NaN, ignore
        else:
            diffs = qty_series.diff()
            if not diffs.empty:
                diffs.iloc[0] = qty_series.iloc[0]  # First diff is just the first quantity

        return float(diffs.clip(lower=0).sum())  # Only count positive diffs as volume
