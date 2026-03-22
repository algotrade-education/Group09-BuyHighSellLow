"""
Validate DataFrames before feeding into pipeline.
Return (bool, List[str]) instead of just bool - caller gets specific reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Required columns for OHLCV data
REQUIRED_OHLCV_COLS = {"open", "high", "low", "close", "volume"}
REQUIRED_DATETIME_COL = "datetime"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid

    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"{len(self.errors)} error(s):")
            lines.extend(f"  ✗ {e}" for e in self.errors)

        if self.warnings:
            lines.append(f"{len(self.warnings)} warning(s):")
            lines.extend(f"  ⚠ {w}" for w in self.warnings)

        return "\n".join(lines) if lines else "OK"


class DataValidator:
    """
    Validate OHLCV DataFrames.

    Usage:
        validator = DataValidator()
        result = validator.validate_ohlcv(df)
        if not result:
            raise ValueError(result.summary())
    """

    def validate_ohlcv(self, df: pd.DataFrame) -> ValidationResult:
        """
        Full OHLCV validation.
        Runs all checks and collects all errors - does not stop at first error.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if df.empty:
            return ValidationResult(False, ["DataFrame rỗng."])

        # ── Schema ────────────────────────────────────────────────
        errors.extend(self._check_columns(df))

        # If required base columns are missing, further checks cannot run
        if errors:
            return ValidationResult(False, errors)

        # ── Datetime ──────────────────────────────────────────────
        errors.extend(self._check_datetime(df))

        # ── Price relationships ───────────────────────────────────
        errors.extend(self._check_price_relationships(df))

        # ── Values ───────────────────────────────────────────────
        errors.extend(self._check_price_values(df))
        errors.extend(self._check_volume(df))

        # ── NaN ───────────────────────────────────────────────────
        errors.extend(self._check_nan(df))

        # ── Warnings (do not fail validation) ─────────────────────
        warnings.extend(self._check_gaps(df))
        warnings.extend(self._check_volume_anomalies(df))

        is_valid = len(errors) == 0
        result = ValidationResult(is_valid, errors, warnings)

        if not is_valid:
            logger.warning("OHLCV validation failed:\n%s", result.summary())
        elif warnings:
            logger.debug("OHLCV validation passed with warnings:\n%s", result.summary())

        return result

    def validate_schema(
        self,
        df: pd.DataFrame,
        required_cols: list[str],
    ) -> ValidationResult:
        """
        Only check columns and dtypes - lighter than validate_ohlcv.
        Use after loading from cache to verify schema is not corrupt.
        """
        errors = self._check_columns(df, required_cols)
        return ValidationResult(len(errors) == 0, errors)

    # ── Private checks ────────────────────────────────────────────

    def _check_columns(
        self,
        df: pd.DataFrame,
        required: set[str] | list[str] | None = None,
    ) -> list[str]:
        errors: list[str] = []

        cols_needed = (
            required if required is not None else (REQUIRED_OHLCV_COLS | {REQUIRED_DATETIME_COL})
        )
        missing = set(cols_needed) - set(df.columns)

        if missing:
            errors.append(f"Thiếu columns: {sorted(missing)}")

        return errors

    def _check_datetime(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        dt_col = df[REQUIRED_DATETIME_COL]

        # Check dtype
        if not pd.api.types.is_datetime64_any_dtype(dt_col):
            errors.append(
                f"Column '{REQUIRED_DATETIME_COL}' must be datetime64, current type: {dt_col.dtype}."
            )
            return errors  # Do not check further if datetime column is not correct type

        # Check duplicates
        n_dupes = dt_col.duplicated().sum()
        if n_dupes > 0:
            errors.append(f"{n_dupes} duplicate timestamps.")

        # Check sorted
        if not dt_col.is_monotonic_increasing:
            errors.append("Timestamps must be sorted in increasing order.")

        return errors

    def _check_price_relationships(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []

        # high >= low
        bad = (df["high"] < df["low"]).sum()
        if bad > 0:
            errors.append(f"{bad} bars has high < low.")

        # high >= open
        bad = (df["high"] < df["open"]).sum()
        if bad > 0:
            errors.append(f"{bad} bars has high < open.")

        # high >= close
        bad = (df["high"] < df["close"]).sum()
        if bad > 0:
            errors.append(f"{bad} bars has high < close.")

        # low <= open
        bad = (df["low"] > df["open"]).sum()
        if bad > 0:
            errors.append(f"{bad} bars has low > open.")

        # low <= close
        bad = (df["low"] > df["close"]).sum()
        if bad > 0:
            errors.append(f"{bad} bars has low > close.")

        return errors

    def _check_price_values(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []

        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                continue

            n_zero_neg = (df[col] <= 0).sum()
            if n_zero_neg > 0:
                errors.append(f"{n_zero_neg} bars has {col} <= 0.")

        return errors

    def _check_volume(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []

        if "volume" not in df.columns:
            return errors

        n_neg = (df["volume"] < 0).sum()
        if n_neg > 0:
            errors.append(f"{n_neg} bars has negative volume.")

        return errors

    def _check_nan(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []

        critical = [
            c for c in ("open", "high", "low", "close", REQUIRED_DATETIME_COL) if c in df.columns
        ]
        for col in critical:
            n_nan = df[col].isna().sum()
            if n_nan > 0:
                errors.append(f"{n_nan} NaN in column '{col}'.")

        return errors

    def _check_gaps(self, df: pd.DataFrame) -> list[str]:
        """Warn if unusually large time gaps are detected (potential missing data)."""
        warnings: list[str] = []
        if REQUIRED_DATETIME_COL not in df.columns:
            return warnings
        if not pd.api.types.is_datetime64_any_dtype(df[REQUIRED_DATETIME_COL]):
            return warnings

        dt = df[REQUIRED_DATETIME_COL]
        if len(dt) < 2:
            return warnings

        diffs = dt.diff().dropna()
        positive_diffs = diffs[diffs > pd.Timedelta(0)]
        if positive_diffs.empty:
            return warnings
        min_diff = positive_diffs.min()

        # Gap > 10x minimum observed interval is abnormal
        large_gaps = (diffs > min_diff * 10).sum()
        if large_gaps > 0:
            warnings.append(
                f"{large_gaps} abnormal time gap (> 10x minimum interval). May indicate missing data or holidays."
            )
        return warnings

    def _check_volume_anomalies(self, df: pd.DataFrame) -> list[str]:
        """Warn if volume = 0 appears during trading hours."""
        warnings: list[str] = []
        if "volume" not in df.columns:
            return warnings

        n_zero = (df["volume"] == 0).sum()
        if n_zero > 0:
            warnings.append(f"{n_zero} bars has volume = 0.")

        return warnings
