"""Unit tests for BarAggregator core functionality.

Tests verify fundamental bar aggregation operations:
- OHLC accumulation within a single bucket
- Bucket rollover when tick arrives in new bucket
- Clock-based rollover via check_time()
- Callback registration via set_on_bar()
- Accumulator reset after bar emission

Test organization:
- _floor_to_bucket: Timestamp bucketing helper
- set_on_bar: Callback registration
- on_tick (single bucket): OHLC accumulation
- on_tick (bucket rollover): Bar emission and reset
- check_time: Clock-based rollover
- _reset_accumulators: State cleanup
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.paper.bar_aggregator import BarAggregator, _floor_to_bucket

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_aggregator(freq_minutes: int = 5) -> BarAggregator:
    return BarAggregator(
        freq_minutes=freq_minutes,
        atr_period=14,
        fallback_bar_provider=None,
        runtime_config={},
    )


def dt(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2024, 1, 15, hour, minute, second)


# ---------------------------------------------------------------------------
# _floor_to_bucket helper
# ---------------------------------------------------------------------------


class TestFloorToBucket:
    def test_already_on_boundary(self):
        d = datetime(2024, 1, 15, 9, 5, 0)
        assert _floor_to_bucket(d, 5) == datetime(2024, 1, 15, 9, 5, 0)

    def test_floors_seconds_and_sub_minutes(self):
        d = datetime(2024, 1, 15, 9, 7, 45)
        assert _floor_to_bucket(d, 5) == datetime(2024, 1, 15, 9, 5, 0)

    def test_floors_to_hour_boundary(self):
        d = datetime(2024, 1, 15, 10, 0, 30)
        assert _floor_to_bucket(d, 5) == datetime(2024, 1, 15, 10, 0, 0)

    def test_1min_freq(self):
        d = datetime(2024, 1, 15, 9, 3, 59)
        assert _floor_to_bucket(d, 1) == datetime(2024, 1, 15, 9, 3, 0)

    def test_15min_freq(self):
        d = datetime(2024, 1, 15, 9, 22, 0)
        assert _floor_to_bucket(d, 15) == datetime(2024, 1, 15, 9, 15, 0)


# ---------------------------------------------------------------------------
# set_on_bar
# ---------------------------------------------------------------------------


class TestSetOnBar:
    def test_registers_callback(self):
        agg = make_aggregator()
        cb = MagicMock()
        agg.set_on_bar(cb)
        assert agg._on_bar is cb

    def test_replaces_existing_callback(self):
        agg = make_aggregator()
        cb1 = MagicMock()
        cb2 = MagicMock()
        agg.set_on_bar(cb1)
        agg.set_on_bar(cb2)
        assert agg._on_bar is cb2


# ---------------------------------------------------------------------------
# on_tick - single bucket accumulation
# ---------------------------------------------------------------------------


class TestOnTickSingleBucket:
    def test_first_tick_sets_ohlc(self):
        agg = make_aggregator()
        agg.on_tick(dt(9, 1), price=1300.0, volume=10)

        assert agg._open == 1300.0
        assert agg._high == 1300.0
        assert agg._low == 1300.0
        assert agg._close == 1300.0
        assert agg._volume == 10

    def test_second_tick_updates_close_and_volume(self):
        agg = make_aggregator()
        agg.on_tick(dt(9, 1), price=1300.0, volume=10)
        agg.on_tick(dt(9, 2), price=1310.0, volume=5)

        assert agg._open == 1300.0
        assert agg._close == 1310.0
        assert agg._volume == 15

    def test_high_tracks_maximum(self):
        agg = make_aggregator()
        for price in [1300, 1320, 1310, 1315]:
            agg.on_tick(dt(9, 1), price=float(price), volume=1)
        assert agg._high == 1320.0

    def test_low_tracks_minimum(self):
        agg = make_aggregator()
        for price in [1300, 1280, 1290, 1295]:
            agg.on_tick(dt(9, 1), price=float(price), volume=1)
        assert agg._low == 1280.0

    def test_open_is_first_tick(self):
        agg = make_aggregator()
        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 2), price=1350.0, volume=1)
        agg.on_tick(dt(9, 3), price=1250.0, volume=1)
        assert agg._open == 1300.0

    def test_close_is_last_tick(self):
        agg = make_aggregator()
        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 2), price=1350.0, volume=1)
        agg.on_tick(dt(9, 3), price=1275.0, volume=1)
        assert agg._close == 1275.0

    def test_trade_count_increments(self):
        agg = make_aggregator()
        for i in range(5):
            agg.on_tick(dt(9, i), price=1300.0, volume=1)
        assert agg._trade_count == 5

    def test_has_live_trade_set(self):
        agg = make_aggregator()
        assert not agg._has_live_trade
        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        assert agg._has_live_trade

    def test_first_and_last_trade_ts(self):
        agg = make_aggregator()
        t1 = dt(9, 1, 10)
        t2 = dt(9, 2, 30)
        agg.on_tick(t1, price=1300.0, volume=1)
        agg.on_tick(t2, price=1310.0, volume=1)
        assert agg._bar_first_trade_ts == t1
        assert agg._bar_last_trade_ts == t2

    def test_current_bucket_set_correctly(self):
        agg = make_aggregator(freq_minutes=5)
        agg.on_tick(dt(9, 7, 30), price=1300.0, volume=1)
        assert agg._current_bucket == datetime(2024, 1, 15, 9, 5, 0)


# ---------------------------------------------------------------------------
# on_tick - bucket rollover
# ---------------------------------------------------------------------------


class TestOnTickBucketRollover:
    def test_new_bucket_resets_accumulators(self):
        agg = make_aggregator(freq_minutes=5)
        # First bucket: 09:00–09:05
        agg.on_tick(dt(9, 1), price=1300.0, volume=10)
        agg.on_tick(dt(9, 2), price=1320.0, volume=5)

        # Second bucket: 09:05–09:10
        agg.on_tick(dt(9, 6), price=1310.0, volume=3)

        assert agg._open == 1310.0
        assert agg._high == 1310.0
        assert agg._low == 1310.0
        assert agg._close == 1310.0
        assert agg._volume == 3
        assert agg._trade_count == 1

    def test_new_bucket_updates_current_bucket(self):
        agg = make_aggregator(freq_minutes=5)
        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 6), price=1310.0, volume=1)
        assert agg._current_bucket == datetime(2024, 1, 15, 9, 5, 0)

    def test_rollover_calls_emit_bar(self):
        agg = make_aggregator(freq_minutes=5)
        agg._emit_bar = MagicMock()

        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 6), price=1310.0, volume=1)

        agg._emit_bar.assert_called_once()

    def test_no_emit_within_same_bucket(self):
        agg = make_aggregator(freq_minutes=5)
        agg._emit_bar = MagicMock()

        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 2), price=1310.0, volume=1)
        agg.on_tick(dt(9, 3), price=1320.0, volume=1)

        agg._emit_bar.assert_not_called()

    def test_multiple_rollovers(self):
        agg = make_aggregator(freq_minutes=5)
        emit_calls = []
        original_emit = agg._emit_bar

        def tracking_emit():
            emit_calls.append(agg._current_bucket)
            original_emit()

        agg._emit_bar = tracking_emit

        agg.on_tick(dt(9, 1), price=1300.0, volume=1)  # bucket 09:00
        agg.on_tick(dt(9, 6), price=1310.0, volume=1)  # bucket 09:05 → emit 09:00
        agg.on_tick(dt(9, 11), price=1320.0, volume=1)  # bucket 09:10 → emit 09:05

        assert len(emit_calls) == 2


# ---------------------------------------------------------------------------
# check_time - clock-based rollover
# ---------------------------------------------------------------------------


class TestCheckTime:
    def test_no_emit_when_no_accumulated_data(self):
        agg = make_aggregator(freq_minutes=5)
        agg._emit_bar = MagicMock()

        with patch("src.paper.bar_aggregator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 9, 6, 0)
            agg.check_time()

        agg._emit_bar.assert_not_called()

    def test_no_emit_when_same_bucket(self):
        agg = make_aggregator(freq_minutes=5)
        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg._emit_bar = MagicMock()

        # Wall clock still in same bucket (09:00)
        with patch("src.paper.bar_aggregator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 9, 3, 0)
            agg.check_time()

        agg._emit_bar.assert_not_called()

    def test_emits_when_clock_crosses_bucket_boundary(self):
        agg = make_aggregator(freq_minutes=5)
        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg._emit_bar = MagicMock()

        # Wall clock is now in the next bucket (09:05)
        with patch("src.paper.bar_aggregator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 9, 6, 0)
            agg.check_time()

        agg._emit_bar.assert_called_once()

    def test_no_emit_when_no_live_trade(self):
        """check_time should not emit if _has_live_trade is False."""
        agg = make_aggregator(freq_minutes=5)
        # Manually set a stale bucket without live trade
        agg._current_bucket = datetime(2024, 1, 15, 9, 0, 0)
        agg._has_live_trade = False
        agg._emit_bar = MagicMock()

        with patch("src.paper.bar_aggregator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 9, 6, 0)
            agg.check_time()

        agg._emit_bar.assert_not_called()


# ---------------------------------------------------------------------------
# _reset_accumulators
# ---------------------------------------------------------------------------


class TestResetAccumulators:
    def test_all_fields_reset(self):
        agg = make_aggregator()
        agg.on_tick(dt(9, 1), price=1300.0, volume=10)
        agg._reset_accumulators()

        assert agg._current_bucket is None
        assert agg._open is None
        assert agg._high is None
        assert agg._low is None
        assert agg._close is None
        assert agg._volume == 0.0
        assert agg._bar_first_trade_ts is None
        assert agg._bar_last_trade_ts is None
        assert agg._trade_count == 0
        assert not agg._has_live_trade
