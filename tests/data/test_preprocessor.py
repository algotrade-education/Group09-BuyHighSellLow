"""Tests for DataPreprocessor."""

import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def sample_df():
    """Sample OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 09:00", periods=20, freq="1min"),
            "open": range(100, 120),
            "high": range(101, 121),
            "low": range(99, 119),
            "close": range(100, 120),
            "volume": [1000 + i * 100 for i in range(20)],
        }
    )


@pytest.fixture
def unsorted_df():
    """Unsorted DataFrame with duplicates."""
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 09:05",
                    "2024-01-01 09:00",
                    "2024-01-01 09:10",
                    "2024-01-01 09:05",
                ]
            ),
            "open": [101.0, 100.0, 102.0, 101.5],
            "high": [102.0, 101.0, 103.0, 102.5],
            "low": [100.0, 99.0, 101.0, 100.5],
            "close": [101.5, 100.5, 102.5, 101.8],
            "volume": [1100, 1000, 1200, 1150],
        }
    )


class TestDataPreprocessorInit:
    """Test DataPreprocessor initialization."""

    def test_init_defaults(self):
        preprocessor = DataPreprocessor()
        assert preprocessor._dt_col == "datetime"
        assert preprocessor._session is not None

    def test_init_custom_session(self):
        from config.schemas.session import VN30SessionConfig

        session = VN30SessionConfig()
        preprocessor = DataPreprocessor(session=session)
        assert preprocessor._session == session


class TestDataPreprocessorClean:
    """Test DataPreprocessor.clean method."""

    def test_clean_sorts_data(self, unsorted_df):
        preprocessor = DataPreprocessor()

        result = preprocessor.clean(unsorted_df)

        assert result["datetime"].is_monotonic_increasing

    def test_clean_removes_duplicates(self, unsorted_df):
        preprocessor = DataPreprocessor()

        result = preprocessor.clean(unsorted_df)

        # Should keep last duplicate (09:05 with open=101.5)
        assert len(result) == 3
        assert result[result["datetime"] == "2024-01-01 09:05"]["open"].iloc[0] == 101.5

    def test_clean_converts_datetime(self):
        df = pd.DataFrame(
            {
                "datetime": ["2024-01-01 09:00", "2024-01-01 09:05"],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000, 1100],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.clean(df)

        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_clean_resets_index(self, unsorted_df):
        preprocessor = DataPreprocessor()

        result = preprocessor.clean(unsorted_df)

        assert result.index.tolist() == [0, 1, 2]


class TestDataPreprocessorResample:
    """Test DataPreprocessor.resample method."""

    def test_resample_1min_to_5min(self, sample_df):
        preprocessor = DataPreprocessor()

        result = preprocessor.resample(sample_df, "5min")

        assert len(result) == 4
        assert result["open"].iloc[0] == 100
        assert result["close"].iloc[0] == 104

    def test_resample_aggregation_rules(self):
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=5, freq="1min"),
                "open": [100, 101, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.resample(df, "5min")

        assert result["open"].iloc[0] == 100  # first
        assert result["high"].iloc[0] == 109  # max
        assert result["low"].iloc[0] == 95  # min
        assert result["close"].iloc[0] == 104  # last
        assert result["volume"].iloc[0] == 6000  # sum

    def test_resample_drops_empty_bars(self):
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=3, freq="1min"),
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [1000, 1100, 1200],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.resample(df, "5min")

        # Should only have 1 bar (3 minutes < 5 minutes)
        assert len(result) == 1

    def test_resample_unsupported_freq(self, sample_df):
        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError):
            preprocessor.resample(sample_df, "invalid_freq")


class TestDataPreprocessorFilterTradingHours:
    """Test DataPreprocessor.filter_trading_hours method."""

    def test_filter_keeps_morning_session(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2024-01-01 09:00",  # Morning start - kept
                        "2024-01-01 10:00",  # Morning - kept
                        "2024-01-01 11:29",  # Last morning bar - kept
                        "2024-01-01 11:30",  # MORNING_END is exclusive -> lunch break, filtered out
                    ]
                ),
                "open": [100, 101, 102, 103],
                "high": [101, 102, 103, 104],
                "low": [99, 100, 101, 102],
                "close": [100, 101, 102, 103],
                "volume": [1000, 1100, 1200, 1300],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.filter_trading_hours(df)

        assert len(result) == 3  # 11:30 filtered out (exclusive boundary)
        assert "2024-01-01 11:30" not in result["datetime"].astype(str).values

    def test_filter_removes_lunch_break(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2024-01-01 11:25",  # Last morning bar - kept
                        "2024-01-01 11:30",  # Lunch start - filtered out (exclusive boundary)
                        "2024-01-01 12:00",  # Lunch
                        "2024-01-01 13:00",  # Afternoon start
                    ]
                ),
                "open": [99, 100, 101, 102],
                "high": [105, 101, 102, 103],
                "low": [99, 100, 101, 102],
                "close": [100, 101, 102, 103],
                "volume": [1000, 1100, 1200, 1300],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.filter_trading_hours(df)

        # 11:30 (exclusive boundary) and 12:00 (lunch) are both filtered out -> 2 bars remain
        assert len(result) == 2
        assert "2024-01-01 11:30" not in result["datetime"].astype(str).values

    def test_filter_keeps_afternoon_session(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2024-01-01 13:00",  # Afternoon start - kept
                        "2024-01-01 14:00",  # Afternoon - kept
                        "2024-01-01 14:30",  # ATC start - kept (within ATC session)
                        "2024-01-01 14:44",  # Last ATC bar - kept
                        "2024-01-01 14:45",  # ATC_END is exclusive -> after close, filtered out
                    ]
                ),
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.filter_trading_hours(df)

        assert len(result) == 4  # 14:45 filtered out (exclusive boundary)
        assert "2024-01-01 14:45" not in result["datetime"].astype(str).values

    def test_filter_removes_before_market_open(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2024-01-01 08:00",  # Before open
                        "2024-01-01 09:00",  # Market open
                    ]
                ),
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.filter_trading_hours(df)

        assert len(result) == 1
        assert result["datetime"].iloc[0].hour == 9


class TestDataPreprocessorPrepare:
    """Test DataPreprocessor.prepare method."""

    def test_prepare_full_pipeline(self, unsorted_df):
        preprocessor = DataPreprocessor()

        result = preprocessor.prepare(unsorted_df, freq="1min")

        # Should be cleaned, sorted, and filtered
        assert result["datetime"].is_monotonic_increasing
        assert len(result) <= len(unsorted_df)

    def test_prepare_with_resample(self):
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=10, freq="1min"),
                "open": range(100, 110),
                "high": range(101, 111),
                "low": range(99, 109),
                "close": range(100, 110),
                "volume": [1000] * 10,
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.prepare(df, freq="5min")

        assert len(result) == 2


class TestDataPreprocessorSessionLabel:
    """Test DataPreprocessor.add_session_label method."""

    def test_add_session_label_morning(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-01 09:00", "2024-01-01 10:00"]),
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.add_session_label(df)

        assert "session" in result.columns
        assert "session_id" in result.columns
        assert result["session"].iloc[0] == "morning"

    def test_add_session_label_afternoon(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-01 13:00", "2024-01-01 14:00"]),
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.add_session_label(df)

        assert result["session"].iloc[0] == "afternoon"

    def test_add_session_label_closed(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 12:00"]),
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.add_session_label(df)

        assert result["session"].iloc[0] == "closed"
        assert result["session"].iloc[1] == "closed"

    def test_session_id_encoding(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-01 09:00", "2024-01-01 13:00"]),
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [1000, 1100],
            }
        )
        preprocessor = DataPreprocessor()

        result = preprocessor.add_session_label(df)

        # Morning = 1, Afternoon = 2
        morning_id = result["session_id"].iloc[0]
        afternoon_id = result["session_id"].iloc[1]

        assert morning_id % 10 == 1
        assert afternoon_id % 10 == 2


class TestDataPreprocessorValidation:
    """Test DataPreprocessor input validation."""

    def test_validate_input_none(self):
        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError, match="is None"):
            preprocessor._validate_input(None)

    def test_validate_input_empty(self):
        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError, match="is empty"):
            preprocessor._validate_input(pd.DataFrame())

    def test_validate_input_missing_columns(self, sample_df):
        preprocessor = DataPreprocessor()

        with pytest.raises(ValueError, match="missing required columns"):
            preprocessor._validate_input(sample_df, require_columns=["datetime", "missing_col"])
