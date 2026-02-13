"""
Data preprocessor for cleaning data and calculating technical indicators.
"""

import logging
from datetime import time
from typing import Optional

import pandas as pd

from config.config import (
    ATC_END,
    TRADING_END,
    TRADING_START,
)
from src.data.validators import DataValidator

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocesses market data and calculates indicators.

    Handles:
    - Data cleaning and validation
    - OHLC resampling from tick data
    - Add technical indicator calculation (SMA, Bollinger Bands)
    """

    def __init__(
        self,
        sma_period: int = 20,
        bb_std: float = 2.0,
        slope_lookback: int = 1,
    ):
        """
        Initialize the preprocessor.

        Args:
            sma_period: Period for SMA calculation
            bb_std: Standard deviation multiplier for Bollinger Bands
            slope_lookback: Number of periods for slope calculation
            validate_data: Whether to validate data quality
            strict_validation: If True, warnings become errors
        """
        self.sma_period = sma_period
        self.bb_std = bb_std
        self.slope_lookback = slope_lookback
        self.validator = DataValidator()

    def clean_data(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Clean raw data by handling missing values and duplicates.

        Args:
            df: Raw DataFrame
            copy: If True, create a copy; if False, modify in-place (more memory efficient)

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df

        if copy:
            df = df.copy()

        # Remove duplicates (in-place)
        df.drop_duplicates(inplace=True)

        # Forward fill missing values for price columns
        price_cols = ["price", "best-bid", "best-ask", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Drop remaining rows with missing critical values (in-place)
        df.dropna(subset=["datetime"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def resample_to_ohlc(
        self,
        df: pd.DataFrame,
        freq: str = "5min",
        price_column: str = "price",
        datetime_column: str = "datetime",
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Resample tick data to OHLC bars.

        Args:
            df: Tick data DataFrame
            freq: Resampling frequency (e.g., '1min', '5min', '1h')
            price_column: Column containing price data
            datetime_column: Column containing datetime data
            copy: If True, create a copy; if False, modify original (more memory efficient)

        Returns:
            OHLC DataFrame
        """
        if df.empty:
            return pd.DataFrame()

        if datetime_column not in df.columns:
            raise ValueError(
                f"Datetime column '{datetime_column}' not found in DataFrame."
            )

        if copy:
            df = df.copy()
        
        df.set_index(datetime_column, inplace=True)

        logger.info("Resampling data to %s...", freq)
        
        # Define aggregation dictionary
        agg_dict = {
            price_column: "ohlc"
        }
        
        # Add volume aggregation if present
        if "volume" in df.columns:
            agg_dict["volume"] = "sum"

        # Resample
        resampled = df.resample(freq).agg(agg_dict) # type: ignore
        
        # Handle price OHLC columns
        price_ohlc = resampled[price_column].copy()
        
        # Handle volume if exists
        if "volume" in df.columns:
             price_ohlc["volume"] = resampled["volume"].values

        result = price_ohlc.dropna().reset_index()

        # Validate OHLC relationships if validation is enabled
        validation_result = self.validator.validate_ohlc(result)
        if not validation_result:
            raise ValueError("OHLC validation failed")

        return result

    def _derive_volume(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Derive per-tick volume from cumulative quantity column.
        Groups by date to handle daily resets.

        Args:
            df: DataFrame with quantity column
            copy: If True, create a copy; if False, modify in-place
        """
        if "quantity" not in df.columns:
            return df
            
        if "volume" in df.columns:
            return df

        if copy:
            df = df.copy()
        
        if "datetime" in df.columns:
            # Sort just in case
            df.sort_values("datetime", inplace=True)
            
            # Calculate diff grouped by date (handles daily resets)
            # For the first tick of each day (NaN diff), fill with quantity value itself
            df["volume"] = (
                df.groupby(df["datetime"].dt.normalize())["quantity"] # type: ignore
                .diff()
                .fillna(df["quantity"])
            )
            
            # Ensure no negative volumes (data errors)
            df.loc[df["volume"] < 0, "volume"] = 0
            
        return df

    def add_sma(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
        column: str = "close",
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Add Simple Moving Average to DataFrame.
        
        Args:
            df: Input DataFrame
            period: SMA period
            column: Column to calculate SMA on
            copy: If True, create a copy; if False, modify in-place
        """
        period = period or self.sma_period
        if copy:
            df = df.copy()
        df[f"sma_{period}"] = df[column].rolling(window=period).mean()
        return df

    def add_sma_slope(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
        lookback: int = 1,
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Add SMA slope (difference from previous value).
        
        Args:
            df: Input DataFrame
            period: SMA period
            lookback: Number of periods to look back for slope
            copy: If True, create a copy; if False, modify in-place
        """
        period = period or self.sma_period
        sma_col = f"sma_{period}"

        if sma_col not in df.columns:
            # Don't copy here if we're not copying overall
            df = self.add_sma(df, period, copy=copy)
            copy = False  # Already have the data we need

        if copy:
            df = df.copy()
        df[f"sma_{period}_slope"] = df[sma_col] - df[sma_col].shift(lookback)

        return df

    def add_volume_ma(
        self,
        df: pd.DataFrame,
        period: int = 20,
        column: str = "volume",
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Add Volume Moving Average.
        
        Args:
            df: Input DataFrame
            period: MA period for volume
            column: Volume column name
            copy: If True, create a copy; if False, modify in-place
        """
        if column not in df.columns:
            return df
        
        if copy:
            df = df.copy()
        df[f"volume_ma_{period}"] = df[column].rolling(window=period).mean()
        return df

    def add_rsi(
        self,
        df: pd.DataFrame,
        period: int = 14,
        column: str = "close",
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).
        
        Args:
            df: Input DataFrame
            period: RSI period
            column: Price column
            copy: If True, create a copy; if False, modify in-place
        """
        if copy:
            df = df.copy()

        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        
        # Fill NaN with 50 (neutral) for initial periods to avoid validation errors
        df[f"rsi_{period}"] = df[f"rsi_{period}"].fillna(50)
        
        return df

    def add_adx(
        self,
        df: pd.DataFrame,
        period: int = 14,
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX).
        
        Args:
            df: Input DataFrame
            period: ADX period
            copy: If True, create a copy; if False, modify in-place
        """
        if copy:
            df = df.copy()

        # Calculate True Range
        df["h-l"] = df["high"] - df["low"]
        df["h-pc"] = abs(df["high"] - df["close"].shift(1))
        df["l-pc"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["h-l", "h-pc", "l-pc"]].max(axis=1)

        # Calculate DM
        df["up_move"] = df["high"] - df["high"].shift(1)
        df["down_move"] = df["low"].shift(1) - df["low"]

        df["plus_dm"] = 0.0
        df.loc[(df["up_move"] > df["down_move"]) & (df["up_move"] > 0), "plus_dm"] = df["up_move"]

        df["minus_dm"] = 0.0
        df.loc[(df["down_move"] > df["up_move"]) & (df["down_move"] > 0), "minus_dm"] = df["down_move"]

        # Calculate Smoothed components (Wilder's Smoothing)
        # Using EMA as approximation for efficiency, or rolling mean
        df["tr_smooth"] = df["tr"].rolling(window=period).mean()
        df["plus_dm_smooth"] = df["plus_dm"].rolling(window=period).mean()
        df["minus_dm_smooth"] = df["minus_dm"].rolling(window=period).mean()

        # Calculate DI
        df["plus_di"] = 100 * (df["plus_dm_smooth"] / df["tr_smooth"])
        df["minus_di"] = 100 * (df["minus_dm_smooth"] / df["tr_smooth"])

        # Calculate DX
        df["dx"] = 100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])

        # Calculate ADX
        df[f"adx_{period}"] = df["dx"].rolling(window=period).mean()
        
        # Fill NaN
        df[f"adx_{period}"] = df[f"adx_{period}"].fillna(0)

        # Cleanup temporary columns
        cols_to_drop = ["h-l", "h-pc", "l-pc", "tr", "up_move", "down_move", 
                       "plus_dm", "minus_dm", "tr_smooth", "plus_dm_smooth", 
                       "minus_dm_smooth", "plus_di", "minus_di", "dx"]
        df.drop(columns=cols_to_drop, inplace=True)

        return df

    def add_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
        std_dev: Optional[float] = None,
        column: str = "close",
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands to DataFrame.
        
        Args:
            df: Input DataFrame
            period: SMA period for middle band
            std_dev: Standard deviation multiplier
            column: Column to calculate bands on
            copy: If True, create a copy; if False, modify in-place
        """
        period = period or self.sma_period
        std_dev = std_dev or self.bb_std

        if copy:
            df = df.copy()

        # Calculate middle band (SMA)
        sma_col = f"sma_{period}"
        if sma_col not in df.columns:
            df[sma_col] = df[column].rolling(window=period).mean()

        # Calculate standard deviation
        rolling_std = df[column].rolling(window=period).std()

        # Calculate upper and lower bands
        df["bb_upper"] = df[sma_col] + (std_dev * rolling_std)
        df["bb_middle"] = df[sma_col]
        df["bb_lower"] = df[sma_col] - (std_dev * rolling_std)

        return df

    def add_all_indicators(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Add all required indicators for the strategy.
        
        Args:
            df: Input DataFrame
            copy: If True, create initial copy; if False, modify in-place (memory efficient)
        
        Returns:
            DataFrame with all indicators
        """
        logger.debug("Calculating indicators...")
        # Only copy once at the start, then modify in-place
        df = self.add_sma(df, copy=copy)
        df = self.add_sma_slope(df, lookback=self.slope_lookback, copy=False)
        df = self.add_bollinger_bands(df, copy=False)
        df = self.add_volume_ma(df, period=20, copy=False)
        df = self.add_rsi(df, period=14, copy=False)
        df = self.add_adx(df, period=14, copy=False)

        return df

    def filter_trading_hours(
        self,
        df: pd.DataFrame,
        datetime_column: str = "datetime",
        include_atc: bool = False,
        copy: bool = True,
    ) -> pd.DataFrame:
        """
        Filter data to trading hours only.
        
        Args:
            df: Input DataFrame
            datetime_column: Name of datetime column
            include_atc: Whether to include ATC session
            copy: If True, create a copy; if False, modify in-place
        """
        if df.empty:
            return df

        if copy:
            df = df.copy()

        # Parse trading hours
        trading_start = time.fromisoformat(TRADING_START)
        trading_end = time.fromisoformat(TRADING_END)
        atc_end = time.fromisoformat(ATC_END)

        end_time = atc_end if include_atc else trading_end

        # Extract time from datetime
        df["_time"] = pd.to_datetime(df[datetime_column]).dt.time

        # Filter to trading hours
        mask = (df["_time"] >= trading_start) & (df["_time"] <= end_time)
        df = df[mask].drop(columns=["_time"])
        df.reset_index(drop=True, inplace=True)

        return df

    def prepare_for_backtest(
        self,
        df: pd.DataFrame,
        resample_freq: Optional[
            str
        ] = None,  # Can be '1min', '5min', '15min', '1h', '1d'
    ) -> pd.DataFrame:
        """Full preprocessing pipeline for backtesting."""
        logger.info("Starting data preprocessing...")

        # Clean data
        df = self.clean_data(df)

        # Derive volume from quantity if needed
        df = self._derive_volume(df)

        # Resample if needed
        if resample_freq:
            df = self.resample_to_ohlc(df, freq=resample_freq)

        # Filter trading hours
        df = self.filter_trading_hours(df, include_atc=True)

        # Add all indicators
        df = self.add_all_indicators(df)

        # Drop rows with NaN indicators (warmup period)
        df = df.dropna().reset_index(drop=True)

        # Final validation
        if not df.empty:
            datetime_result = self.validator.validate_datetime(df)
            if not datetime_result:
                raise ValueError("Datetime validation failed in final dataset.")

        logger.info("Preprocessing complete. Final dataset size: %s rows.", len(df))
        return df
