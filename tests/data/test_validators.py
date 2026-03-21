"""Tests for DataValidator."""

import pandas as pd
import pytest

from src.data.validators import DataValidator, ValidationResult


@pytest.fixture
def valid_ohlcv():
    """Valid OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 09:00", periods=10, freq="5min"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
    )


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_bool_valid(self):
        result = ValidationResult(is_valid=True)
        assert bool(result) is True

    def test_bool_invalid(self):
        result = ValidationResult(is_valid=False, errors=["Error 1"])
        assert bool(result) is False

    def test_summary_no_issues(self):
        result = ValidationResult(is_valid=True)
        assert result.summary() == "OK"

    def test_summary_with_errors(self):
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        summary = result.summary()
        assert "2 error(s)" in summary
        assert "Error 1" in summary
        assert "Error 2" in summary

    def test_summary_with_warnings(self):
        result = ValidationResult(is_valid=True, warnings=["Warning 1"])
        summary = result.summary()
        assert "1 warning(s)" in summary
        assert "Warning 1" in summary

    def test_summary_with_both(self):
        result = ValidationResult(is_valid=False, errors=["Error 1"], warnings=["Warning 1"])
        summary = result.summary()
        assert "error(s)" in summary
        assert "warning(s)" in summary


class TestDataValidatorOHLCV:
    """Test DataValidator.validate_ohlcv method."""

    def test_validate_valid_data(self, valid_ohlcv):
        validator = DataValidator()

        result = validator.validate_ohlcv(valid_ohlcv)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_empty_dataframe(self):
        validator = DataValidator()

        result = validator.validate_ohlcv(pd.DataFrame())

        assert not result.is_valid
        assert "rỗng" in result.errors[0]

    def test_validate_missing_columns(self):
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=5),
                "open": [100, 101, 102, 103, 104],
            }
        )
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("Thiếu columns" in e for e in result.errors)

    def test_validate_duplicate_timestamps(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("duplicate" in e for e in result.errors)

    def test_validate_unsorted_timestamps(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df = df.iloc[[1, 0, 2, 3, 4, 5, 6, 7, 8, 9]]
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("sorted" in e for e in result.errors)


class TestDataValidatorPriceRelationships:
    """Test price relationship validations."""

    def test_high_less_than_low(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "high"] = 90  # Less than low
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("high < low" in e for e in result.errors)

    def test_high_less_than_open(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "high"] = 95  # Less than open
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("high < open" in e for e in result.errors)

    def test_high_less_than_close(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "high"] = 95  # Less than close
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("high < close" in e for e in result.errors)

    def test_low_greater_than_open(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "low"] = 105  # Greater than open
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("low > open" in e for e in result.errors)

    def test_low_greater_than_close(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "low"] = 105  # Greater than close
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("low > close" in e for e in result.errors)


class TestDataValidatorPriceValues:
    """Test price value validations."""

    def test_negative_open(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "open"] = -100
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("open <= 0" in e for e in result.errors)

    def test_zero_close(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "close"] = 0
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("close <= 0" in e for e in result.errors)

    def test_negative_volume(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "volume"] = -1000
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("negative volume" in e for e in result.errors)


class TestDataValidatorNaN:
    """Test NaN validations."""

    def test_nan_in_close(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "close"] = None
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("NaN" in e and "close" in e for e in result.errors)

    def test_nan_in_datetime(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "datetime"] = pd.NaT
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("NaN" in e and "datetime" in e for e in result.errors)


class TestDataValidatorWarnings:
    """Test warning validations."""

    def test_zero_volume_warning(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "volume"] = 0
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert result.is_valid
        assert any("volume = 0" in w for w in result.warnings)

    def test_time_gap_warning(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2024-01-01 09:00",
                        "2024-01-01 09:05",
                        "2024-01-01 12:00",  # Large gap
                    ]
                ),
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
            }
        )
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert result.is_valid
        assert any("gap" in w for w in result.warnings)


class TestDataValidatorSchema:
    """Test DataValidator.validate_schema method."""

    def test_validate_schema_success(self, valid_ohlcv):
        validator = DataValidator()

        result = validator.validate_schema(valid_ohlcv, ["datetime", "open", "close"])

        assert result.is_valid

    def test_validate_schema_missing_columns(self, valid_ohlcv):
        validator = DataValidator()

        result = validator.validate_schema(valid_ohlcv, ["datetime", "missing_col"])

        assert not result.is_valid
        assert any("Thiếu columns" in e for e in result.errors)


class TestDataValidatorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataframe(self):
        df = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2024-01-01 09:00")],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert result.is_valid

    def test_wrong_datetime_dtype(self):
        df = pd.DataFrame(
            {
                "datetime": ["2024-01-01", "2024-01-02"],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 1100],
            }
        )
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert any("datetime64" in e for e in result.errors)

    def test_multiple_errors_collected(self, valid_ohlcv):
        df = valid_ohlcv.copy()
        df.loc[0, "high"] = 90  # high < low
        df.loc[1, "close"] = -100  # negative price
        df.loc[2, "volume"] = -1000  # negative volume
        validator = DataValidator()

        result = validator.validate_ohlcv(df)

        assert not result.is_valid
        assert len(result.errors) >= 3
