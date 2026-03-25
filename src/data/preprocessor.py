"""
data/preprocessor.py

Clean, resample, filter trading hours.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.schemas.base import ResampleFreq
from config.schemas.session import Session, SessionConfig, VN30SessionConfig

logger = logging.getLogger(__name__)

DATETIME_COL = "datetime"


class DataPreprocessor:
    """
    Process raw OHLCV DataFrame
        1. clean()               - dedup, sort, validate
        2. resample()            - 1min -> 5min / 15min / 30min
        3. filter_trading_hours() - only keep bars within trading hours

    Usage:
        preprocessor = DataPreprocessor()
        df = preprocessor.prepare(raw_df, freq="5min")
    """

    def __init__(
        self,
        session: SessionConfig | None = None,
        datetime_col: str = DATETIME_COL,
    ) -> None:
        self._session = session if session is not None else VN30SessionConfig()
        self._dt_col = datetime_col

    # --- Public API ---

    def prepare(self, df: pd.DataFrame, freq: ResampleFreq = "5min") -> pd.DataFrame:
        """
        Full pipeline: clean -> resample -> filter.
        Shortcut method for normal use case.
        """
        df = self.clean(df)

        if freq != "1min":
            df = self.resample(df, freq)

        df = self.filter_trading_hours(df)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Ensure datetime column is datetime64
        - Sort by datetime
        - Remove duplicate timestamps (keep last - assume last is most correct)
        - Reset index
        """
        df = df.copy()

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[self._dt_col]):
            df[self._dt_col] = pd.to_datetime(df[self._dt_col])

        # Sort
        df = df.sort_values(self._dt_col, ignore_index=True)

        # Dedup - keep last occurrence
        n_before = len(df)
        df = df.drop_duplicates(subset=[self._dt_col], keep="last")
        df = df.reset_index(drop=True)
        n_dupes = n_before - len(df)
        if n_dupes > 0:
            logger.warning("Removed %d duplicate timestamps.", n_dupes)

        return df

    def resample(self, df: pd.DataFrame, freq: ResampleFreq) -> pd.DataFrame:
        """
        Resample from tick-by-tick to 1min / 5min / 15min / 30min using standard OHLCV aggregation rules.

        Rules:
            open   = first
            high   = max
            low    = min
            close  = last
            volume = sum

        Do not create bars for times outside trading hours -
        dropna() after resample removes synthetic empty bars.
        """
        self._validate_input(
            df,
            require_columns=[self._dt_col, "open", "high", "low", "close"],
        )

        df = df.copy()
        df = df.set_index(self._dt_col)

        agg_rules: dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # Only keep agg rules for columns that exist in df
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

        resampled = (
            df.resample(self._to_pandas_freq(freq))
            .agg(agg_rules)
            .dropna(subset=["close"])  # Drop bars with no trades (close is NaN)
            .reset_index()
        )

        logger.debug(
            "Resampled %d -> %d bars (%s).",
            len(df),
            len(resampled),
            freq,
        )
        return resampled

    def filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Only keep bars within trading hours.
        Exclude CLOSED session (before morning, lunch break, after close).

        Implementation uses minutes-since-midnight integer
        Does not use dt.time comparison (not vectorized, slow with large DataFrames).
        """
        self._validate_input(df, require_columns=[self._dt_col])

        df = df.copy()

        dt = pd.to_datetime(df[self._dt_col])

        # Minutes since midnight - integer, vectorized
        minutes = dt.dt.hour * 60 + dt.dt.minute

        sess = self._session
        morning_start = sess.MORNING_START.hour * 60 + sess.MORNING_START.minute
        morning_end = sess.MORNING_END.hour * 60 + sess.MORNING_END.minute
        afternoon_start = sess.AFTERNOON_START.hour * 60 + sess.AFTERNOON_START.minute

        # Handle markets with or without ATC
        if sess.has_atc():
            atc_end = sess.ATC_END.hour * 60 + sess.ATC_END.minute  # type: ignore
            mask = ((minutes >= morning_start) & (minutes < morning_end)) | (
                (minutes >= afternoon_start) & (minutes < atc_end)
            )
        else:
            afternoon_end = sess.AFTERNOON_END.hour * 60 + sess.AFTERNOON_END.minute
            mask = ((minutes >= morning_start) & (minutes < morning_end)) | (
                (minutes >= afternoon_start) & (minutes < afternoon_end)
            )

        filtered = df[mask].reset_index(drop=True)
        n_removed = len(df) - len(filtered)
        if n_removed > 0:
            logger.debug("Filtered out %d bars outside trading hours.", n_removed)

        return filtered

    def add_session_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'session' column with values: 'morning', 'afternoon', 'atc', 'closed'.
        Use integer encoding for faster groupby operations.

        New columns added:
            session       : string label ('morning', 'afternoon', 'atc', 'closed')
            session_id    : integer (date * 10 + session_num) for groupby (morning=1, afternoon=2, atc=3, closed=0)
        """
        df = df.copy()
        dt = pd.to_datetime(df[self._dt_col])
        minutes = dt.dt.hour * 60 + dt.dt.minute

        sess = self._session
        morning_start = sess.MORNING_START.hour * 60 + sess.MORNING_START.minute
        morning_end = sess.MORNING_END.hour * 60 + sess.MORNING_END.minute
        afternoon_start = sess.AFTERNOON_START.hour * 60 + sess.AFTERNOON_START.minute
        afternoon_end = sess.AFTERNOON_END.hour * 60 + sess.AFTERNOON_END.minute

        conditions = [
            (minutes >= morning_start) & (minutes < morning_end),
            (minutes >= afternoon_start) & (minutes < afternoon_end),
        ]
        choices = [
            Session.MORNING.value,
            Session.AFTERNOON.value,
        ]

        # Add ATC condition if market has ATC session
        if sess.has_atc():
            atc_start = sess.ATC_START.hour * 60 + sess.ATC_START.minute  # type: ignore
            atc_end = sess.ATC_END.hour * 60 + sess.ATC_END.minute  # type: ignore
            conditions.append((minutes >= atc_start) & (minutes < atc_end))
            choices.append(Session.ATC.value)

        df["session"] = np.select(conditions, choices, default=Session.CLOSED.value)

        # Integer session_id for groupby: yyyymmdd * 10 + session_num
        # morning=1, afternoon=2, atc=3, closed=0
        session_num_map = {
            Session.MORNING.value: 1,
            Session.AFTERNOON.value: 2,
            Session.ATC.value: 3,
            Session.CLOSED.value: 0,
        }
        date_int = dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day
        session_num = df["session"].map(session_num_map).fillna(0).astype(int)
        df["session_id"] = date_int * 10 + session_num

        return df

    # --- Helpers ---

    @staticmethod
    def _to_pandas_freq(freq: str) -> str:
        """Convert our freq format -> pandas offset alias."""
        mapping = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1H": "1H",
            "1D": "1D",
            "1W": "1W",
            "1M": "1M",
        }

        if freq not in mapping:
            raise ValueError(f"Unsupported freq '{freq}'. Supported: {list(mapping.keys())}")

        return mapping[freq]

    def _validate_input(self, df: pd.DataFrame, require_columns: list[str] | None = None) -> None:
        """
        Validate input DataFrame for expected columns and data quality.
        """
        if df is None:
            raise ValueError("Input DataFrame is None.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if require_columns:
            missing = set(require_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        # Additional checks can be added here (e.g. check for negative volumes, non-positive prices, etc.)
        # For now we keep it simple and just check for required columns and non-empty DataFrame.
