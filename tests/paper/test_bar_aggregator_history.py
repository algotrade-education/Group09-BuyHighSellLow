"""Unit tests for BarAggregator history preloading and bar seeding.

Tests verify warmup and initialization functionality:
- preload_history() loads historical bars for indicator warmup
- Invalid bar detection and filtering (high < low)
- seed_current_live_bar() initializes current bar from incomplete data
- Timestamp handling from bar dict (not datetime.now())
- Bucket validation for seeded bars

Test organization:
- preload_history: Historical data loading
- seed_current_live_bar: Current bar initialization
- Timestamp handling: Deterministic time sources
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.paper.bar_aggregator import BarAggregator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_aggregator(freq_minutes: int = 5) -> BarAggregator:
    from src.engine.session.base import AlwaysOpenSession

    return BarAggregator(
        freq_minutes=freq_minutes,
        atr_period=14,
        fallback_bar_provider=None,
        runtime_config={},
        session_manager=AlwaysOpenSession(),
    )


def dt(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2024, 1, 15, hour, minute, second)


def make_bar_row(
    bucket: datetime,
    open_: float = 1300.0,
    high: float = 1320.0,
    low: float = 1290.0,
    close: float = 1310.0,
    volume: float = 100.0,
) -> dict:
    return {
        "datetime": bucket,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


# ---------------------------------------------------------------------------
# preload_history - valid rows
# ---------------------------------------------------------------------------


class TestPreloadHistoryValidRows:
    def test_single_valid_row_appended(self):
        agg = make_aggregator()
        df = pd.DataFrame([make_bar_row(dt(9, 0))])
        agg.preload_history(df)
        assert len(agg._history) == 1

    def test_multiple_valid_rows_all_appended(self):
        agg = make_aggregator()
        rows = [make_bar_row(dt(9, i * 5)) for i in range(5)]
        df = pd.DataFrame(rows)
        agg.preload_history(df)
        assert len(agg._history) == 5

    def test_history_preserves_ohlcv_values(self):
        agg = make_aggregator()
        row = make_bar_row(
            dt(9, 0), open_=1300.0, high=1320.0, low=1290.0, close=1310.0, volume=50.0
        )
        df = pd.DataFrame([row])
        agg.preload_history(df)
        stored = agg._history[0]
        assert stored["open"] == 1300.0
        assert stored["high"] == 1320.0
        assert stored["low"] == 1290.0
        assert stored["close"] == 1310.0
        assert stored["volume"] == 50.0

    def test_history_preserves_datetime(self):
        agg = make_aggregator()
        bucket = dt(9, 5)
        df = pd.DataFrame([make_bar_row(bucket)])
        agg.preload_history(df)
        assert agg._history[0]["datetime"] == bucket

    def test_preload_does_not_call_on_bar(self):
        """preload_history must NOT invoke the on_bar callback."""
        agg = make_aggregator()
        called = []
        agg.set_on_bar(called.append)
        df = pd.DataFrame([make_bar_row(dt(9, 0)), make_bar_row(dt(9, 5))])
        agg.preload_history(df)
        assert called == []

    def test_empty_dataframe_leaves_history_empty(self):
        agg = make_aggregator()
        agg.preload_history(
            pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        )
        assert agg._history == []

    def test_equal_high_low_is_valid(self):
        """high == low is valid (doji candle); must NOT be skipped."""
        agg = make_aggregator()
        row = make_bar_row(dt(9, 0), high=1300.0, low=1300.0)
        df = pd.DataFrame([row])
        agg.preload_history(df)
        assert len(agg._history) == 1


# ---------------------------------------------------------------------------
# preload_history - invalid rows (high < low)
# ---------------------------------------------------------------------------


class TestPreloadHistoryInvalidRows:
    def test_row_with_high_less_than_low_is_skipped(self):
        agg = make_aggregator()
        bad_row = make_bar_row(dt(9, 0), high=1290.0, low=1310.0)  # high < low
        df = pd.DataFrame([bad_row])
        agg.preload_history(df)
        assert len(agg._history) == 0

    def test_warning_logged_for_invalid_row(self, caplog):
        agg = make_aggregator()
        bad_row = make_bar_row(dt(9, 0), high=1290.0, low=1310.0)
        df = pd.DataFrame([bad_row])
        import logging

        with caplog.at_level(logging.WARNING, logger="src.paper.bar_aggregator"):
            agg.preload_history(df)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"

    def test_mixed_rows_only_valid_appended(self):
        agg = make_aggregator()
        rows = [
            make_bar_row(dt(9, 0), high=1320.0, low=1290.0),  # valid
            make_bar_row(dt(9, 5), high=1280.0, low=1310.0),  # invalid: high < low
            make_bar_row(dt(9, 10), high=1330.0, low=1300.0),  # valid
        ]
        df = pd.DataFrame(rows)
        agg.preload_history(df)
        assert len(agg._history) == 2

    def test_warning_count_matches_invalid_rows(self, caplog):
        agg = make_aggregator()
        rows = [
            make_bar_row(dt(9, 0), high=1280.0, low=1310.0),  # invalid
            make_bar_row(dt(9, 5), high=1320.0, low=1290.0),  # valid
            make_bar_row(dt(9, 10), high=1270.0, low=1300.0),  # invalid
        ]
        df = pd.DataFrame(rows)
        import logging

        with caplog.at_level(logging.WARNING, logger="src.paper.bar_aggregator"):
            agg.preload_history(df)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 2

    def test_all_invalid_rows_leaves_history_empty(self):
        agg = make_aggregator()
        rows = [make_bar_row(dt(9, i * 5), high=1280.0, low=1310.0) for i in range(3)]
        df = pd.DataFrame(rows)
        agg.preload_history(df)
        assert agg._history == []


# ---------------------------------------------------------------------------
# seed_current_live_bar - timestamps from bar dict (Requirement 3.3)
# ---------------------------------------------------------------------------


class TestSeedCurrentLiveBar:
    def test_first_trade_ts_set_from_bar_datetime(self):
        """_bar_first_trade_ts must equal bar_dict['datetime'], not datetime.now()."""
        agg = make_aggregator()
        bucket = dt(9, 5)
        bar = make_bar_row(bucket)
        agg.seed_current_live_bar(bar, validate_bucket=False)
        assert agg._bar_first_trade_ts == bucket

    def test_last_trade_ts_set_from_bar_datetime(self):
        """_bar_last_trade_ts must equal bar_dict['datetime'], not datetime.now()."""
        agg = make_aggregator()
        bucket = dt(9, 5)
        bar = make_bar_row(bucket)
        agg.seed_current_live_bar(bar, validate_bucket=False)
        assert agg._bar_last_trade_ts == bucket

    def test_timestamps_not_current_time(self):
        """Timestamps must NOT be close to datetime.now() - they come from the bar."""
        agg = make_aggregator()
        # Use a clearly historical bucket
        bucket = datetime(2023, 6, 1, 9, 5, 0)
        bar = make_bar_row(bucket)
        agg.seed_current_live_bar(bar, validate_bucket=False)
        now = datetime.now()
        # The seeded timestamps should be far from now (more than 1 day apart)
        diff = abs((now - agg._bar_first_trade_ts).total_seconds())
        assert diff > 86400, f"Expected timestamp far from now, got diff={diff}s"

    def test_current_bucket_set_from_bar_datetime(self):
        agg = make_aggregator()
        bucket = dt(9, 5)
        bar = make_bar_row(bucket)
        agg.seed_current_live_bar(bar, validate_bucket=False)
        assert agg._current_bucket == bucket

    def test_ohlcv_seeded_correctly(self):
        agg = make_aggregator()
        bar = make_bar_row(
            dt(9, 5), open_=1300.0, high=1320.0, low=1290.0, close=1310.0, volume=75.0
        )
        agg.seed_current_live_bar(bar, validate_bucket=False)
        assert agg._open == 1300.0
        assert agg._high == 1320.0
        assert agg._low == 1290.0
        assert agg._close == 1310.0
        assert agg._volume == 75.0

    def test_has_live_trade_is_true_after_seed(self):
        """Seeded bar is marked as having live trade data."""
        agg = make_aggregator()
        agg.seed_current_live_bar(make_bar_row(dt(9, 5)), validate_bucket=False)
        assert agg._has_live_trade is True

    def test_trade_count_set_from_bar_dict(self):
        """Trade count should be set from bar_dict, defaulting to 1."""
        agg = make_aggregator()
        bar = make_bar_row(dt(9, 5))
        bar["trade_count"] = 5
        agg.seed_current_live_bar(bar, validate_bucket=False)
        assert agg._trade_count == 5

    def test_subsequent_tick_updates_accumulators(self):
        """After seeding, a tick in the same bucket should update accumulators normally."""
        agg = make_aggregator(freq_minutes=5)
        bucket = dt(9, 5)
        agg.seed_current_live_bar(make_bar_row(bucket, close=1310.0), validate_bucket=False)

        # Tick within the same bucket
        agg.on_tick(dt(9, 6), price=1315.0, volume=5)

        assert agg._has_live_trade is True
        assert agg._close == 1315.0
        assert agg._trade_count == 2  # 1 from seed + 1 from tick
