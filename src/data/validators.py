"""
Data validation utilities for ensuring data quality and integrity.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data validator for market data.

    Validates:
    - OHLC relationships
    - NaN/inf values
    - Data gaps
    """

    def __init__(
        self,
        max_gap_minutes: int = 60,
    ):
        """
        Initialize validator.

        Args:
            max_gap_minutes: Maximum allowed time gap between bars (minutes)
        """
        self.max_gap_minutes = max_gap_minutes

    def validate_ohlc(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLC data integrity.

        Checks:
        - Required columns exist
        - OHLC relationships (high >= low, etc.)
        - No NaN/inf values
        - Positive prices
        - Reasonable price changes
        """
        errors = []

        # Check required columns
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False

        # Check for empty dataframe
        if df.empty:
            errors.append("DataFrame is empty")
            return False

        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.any():
            msg = f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}"
            errors.append(msg)

        # Check for inf values
        inf_mask = np.isinf(df[required_cols]).any(axis=1)
        if inf_mask.any():
            msg = f"Infinite values found in {inf_mask.sum()} rows"
            errors.append(msg)

        # Check OHLC relationships
        invalid_high = df["high"] < df[["low", "open", "close"]].max(axis=1)
        if invalid_high.any():
            errors.append(f"High price invalid in {invalid_high.sum()} rows")

        invalid_low = df["low"] > df[["high", "open", "close"]].min(axis=1)
        if invalid_low.any():
            errors.append(f"Low price invalid in {invalid_low.sum()} rows")

        # Check for non-positive prices
        non_positive = (df[required_cols] <= 0).any(axis=1)
        if non_positive.any():
            errors.append(f"Non-positive prices in {non_positive.sum()} rows")

        is_valid = len(errors) == 0
        return is_valid

    def validate_datetime(
        self, df: pd.DataFrame, datetime_col: str = "datetime"
    ) -> bool:
        """
        Validate datetime column and check for gaps.

        Args:
            df: DataFrame to validate.
            datetime_col: Name of the datetime column.

        Returns:
            True if valid (no parse errors or missing values), False otherwise.
        """
        errors = []
        warnings = []

        if datetime_col not in df.columns:
            errors.append(f"Datetime column '{datetime_col}' not found")
            return False

        # Check if datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            try:
                df[datetime_col] = pd.to_datetime(df[datetime_col])
            except Exception as e:
                errors.append(f"Cannot parse datetime column: {e}")
                return False

        # Check for NaN datetimes
        nan_count = df[datetime_col].isna().sum()
        if nan_count > 0:
            errors.append(f"NaN values in datetime column: {nan_count} rows")

        # Check chronological order
        if not df[datetime_col].is_monotonic_increasing:
            warnings.append("Datetime is not in chronological order")

        # Check for duplicates
        dup_count = df[datetime_col].duplicated().sum()
        if dup_count > 0:
            warnings.append(f"Duplicate timestamps: {dup_count} rows")

        # Check for gaps
        if len(df) > 1:
            time_diffs = df[datetime_col].diff()
            max_gap = time_diffs.max()

            # Convert to minutes
            max_gap_minutes = max_gap.total_seconds() / 60 if pd.notna(max_gap) else 0

            if max_gap_minutes > self.max_gap_minutes:
                warnings.append(
                    f"Large time gap detected: {max_gap_minutes:.0f} minutes "
                    f"(threshold: {self.max_gap_minutes} minutes)"
                )

        is_valid = len(errors) == 0
        return is_valid

    def validate_all(
        self,
        df: pd.DataFrame,
        datetime_col: str = "datetime",
    ) -> bool:
        """
        Run all validations and combine results.

        Args:
            df: DataFrame to validate.
            datetime_col: Name of datetime column.

        Returns:
            bool: True if all validations pass, False otherwise.
        """

        # Validate datetime
        dt_result = self.validate_datetime(df, datetime_col)

        # Validate OHLC
        ohlc_result = self.validate_ohlc(df)

        is_valid = dt_result and ohlc_result
        if not is_valid:
            logger.error("Data validation failed")

        return is_valid
