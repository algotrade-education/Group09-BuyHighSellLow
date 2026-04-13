"""Unit tests for BarAggregator bar emission with deterministic timestamps.

Tests verify bar emission behavior and quality assessment:
- Deterministic bucket timestamps (floored to frequency boundary)
- Database bar merging based on quality assessment
- Reference time calculation for end-gap detection
- Quality reason propagation to merge logic
- History buffer management
- Callback invocation

Test organization:
- Deterministic timestamps: bucket_start vs datetime.now()
- DB merge integration: Quality assessment and merge workflow
- History buffer: Append and trim operations
- Callback invocation: on_bar callback execution

Note: DB merge logic tests use data_quality.maybe_merge_db_bar() directly
for better isolation and clarity.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.paper.bar_aggregator import BarAggregator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_aggregator(
    freq_minutes: int = 5,
    fallback_bar_provider=None,
    runtime_config: dict | None = None,
) -> BarAggregator:
    from src.engine.session.base import AlwaysOpenSession

    return BarAggregator(
        freq_minutes=freq_minutes,
        atr_period=14,
        fallback_bar_provider=fallback_bar_provider,
        runtime_config=runtime_config or {},
        session_manager=AlwaysOpenSession(),
    )


def dt(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2024, 1, 15, hour, minute, second)


# ---------------------------------------------------------------------------
# _emit_bar - deterministic timestamp (Requirement 3.1)
# ---------------------------------------------------------------------------


class TestEmitBarTimestamp:
    def test_bar_datetime_equals_current_bucket(self):
        """bar['datetime'] must be _current_bucket, not datetime.now()."""
        agg = make_aggregator(freq_minutes=5)
        received = []
        agg.set_on_bar(received.append)

        # Tick at 09:07 → bucket is 09:05
        agg.on_tick(dt(9, 7), price=1300.0, volume=10)
        # Trigger rollover with a tick in the next bucket
        agg.on_tick(dt(9, 11), price=1310.0, volume=5)

        assert len(received) == 1
        assert received[0]["datetime"] == datetime(2024, 1, 15, 9, 5, 0)

    def test_bar_datetime_is_bucket_start_not_tick_time(self):
        """Even if tick arrives late, datetime is the bucket start."""
        agg = make_aggregator(freq_minutes=5)
        received = []
        agg.set_on_bar(received.append)

        # Tick at 09:04:59 → bucket 09:00
        agg.on_tick(dt(9, 4, 59), price=1300.0, volume=1)
        # Rollover
        agg.on_tick(dt(9, 5, 1), price=1310.0, volume=1)

        assert received[0]["datetime"] == datetime(2024, 1, 15, 9, 0, 0)

    def test_multiple_bars_have_correct_bucket_datetimes(self):
        """Each emitted bar has its own bucket start as datetime."""
        agg = make_aggregator(freq_minutes=5)
        received = []
        agg.set_on_bar(received.append)

        agg.on_tick(dt(9, 1), price=1300.0, volume=1)  # bucket 09:00
        agg.on_tick(dt(9, 6), price=1310.0, volume=1)  # bucket 09:05 → emit 09:00
        agg.on_tick(dt(9, 11), price=1320.0, volume=1)  # bucket 09:10 → emit 09:05

        assert received[0]["datetime"] == datetime(2024, 1, 15, 9, 0, 0)
        assert received[1]["datetime"] == datetime(2024, 1, 15, 9, 5, 0)


# ---------------------------------------------------------------------------
# _emit_bar - OHLCV values
# ---------------------------------------------------------------------------


class TestEmitBarOHLCV:
    def test_emitted_bar_has_correct_ohlcv(self):
        agg = make_aggregator(freq_minutes=5)
        received = []
        agg.set_on_bar(received.append)

        agg.on_tick(dt(9, 1), price=1300.0, volume=10)
        agg.on_tick(dt(9, 2), price=1320.0, volume=5)
        agg.on_tick(dt(9, 3), price=1290.0, volume=3)
        agg.on_tick(dt(9, 4), price=1310.0, volume=7)
        # Rollover
        agg.on_tick(dt(9, 6), price=1315.0, volume=1)

        bar = received[0]
        assert bar["open"] == 1300.0
        assert bar["high"] == 1320.0
        assert bar["low"] == 1290.0
        assert bar["close"] == 1310.0
        assert bar["volume"] == 25.0


# ---------------------------------------------------------------------------
# _emit_bar - history buffer
# ---------------------------------------------------------------------------


class TestEmitBarHistory:
    def test_history_buffer_grows_on_emit(self):
        agg = make_aggregator(freq_minutes=5)
        agg.set_on_bar(lambda b: None)

        assert len(agg._history) == 0

        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 6), price=1310.0, volume=1)  # emit bucket 09:00

        assert len(agg._history) == 1

        agg.on_tick(dt(9, 11), price=1320.0, volume=1)  # emit bucket 09:05

        assert len(agg._history) == 2

    def test_history_bar_has_correct_datetime(self):
        agg = make_aggregator(freq_minutes=5)
        agg.set_on_bar(lambda b: None)

        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 6), price=1310.0, volume=1)

        assert agg._history[0]["datetime"] == datetime(2024, 1, 15, 9, 0, 0)


# ---------------------------------------------------------------------------
# _emit_bar - callback invocation
# ---------------------------------------------------------------------------


class TestEmitBarCallback:
    def test_callback_called_once_per_bar(self):
        agg = make_aggregator(freq_minutes=5)
        cb = MagicMock()
        agg.set_on_bar(cb)

        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 6), price=1310.0, volume=1)
        agg.on_tick(dt(9, 11), price=1320.0, volume=1)

        assert cb.call_count == 2

    def test_no_callback_when_no_on_bar_registered(self):
        """Should not raise even if no callback is registered."""
        agg = make_aggregator(freq_minutes=5)
        # No set_on_bar call
        agg.on_tick(dt(9, 1), price=1300.0, volume=1)
        agg.on_tick(dt(9, 6), price=1310.0, volume=1)  # should not raise

    def test_no_emit_when_no_live_trade(self):
        """_emit_bar should not call callback if _has_live_trade is False."""
        agg = make_aggregator(freq_minutes=5)
        cb = MagicMock()
        agg.set_on_bar(cb)

        # Manually set state without live trade
        agg._current_bucket = datetime(2024, 1, 15, 9, 0, 0)
        agg._has_live_trade = False
        agg._emit_bar()

        cb.assert_not_called()


# ---------------------------------------------------------------------------
# _emit_bar - accumulators reset after emit
# ---------------------------------------------------------------------------


class TestEmitBarResetsAccumulators:
    def test_accumulators_reset_after_emit(self):
        agg = make_aggregator(freq_minutes=5)
        agg.set_on_bar(lambda b: None)

        agg.on_tick(dt(9, 1), price=1300.0, volume=10)
        agg.on_tick(dt(9, 6), price=1310.0, volume=5)  # triggers emit of 09:00 bucket

        # After emit, accumulators should reflect the new bucket (09:05), not the old one
        assert agg._current_bucket == datetime(2024, 1, 15, 9, 5, 0)
        assert agg._open == 1310.0
        assert agg._volume == 5


# ---------------------------------------------------------------------------
# DB merge - reference time uses bucket_end (Requirement 3.2)
# ---------------------------------------------------------------------------


class TestDbMergeReferenceTime:
    def test_emit_bar_passes_bucket_end_as_reference_time(self):
        """Verify _emit_bar passes bucket_end to maybe_merge_db_bar."""
        captured_calls = []

        def capturing_provider(bucket):
            # We don't need to capture here, just return None
            return None

        # Patch at the source - in bar_aggregator's imported module
        from src.paper import bar_aggregator as ba_module
        from src.paper.data_quality import maybe_merge_db_bar as real_mmdb

        def capturing_mmdb(bar, bar_state, reference_time, config, provider):
            captured_calls.append(reference_time)
            return real_mmdb(bar, bar_state, reference_time, config, provider)

        original_mmdb = ba_module.maybe_merge_db_bar
        ba_module.maybe_merge_db_bar = capturing_mmdb

        try:
            agg = make_aggregator(
                freq_minutes=5,
                fallback_bar_provider=capturing_provider,
                runtime_config={"stale_trade_seconds": 60.0, "min_live_updates": 1},
            )
            agg.set_on_bar(lambda b: None)

            agg.on_tick(dt(9, 1), price=1300.0, volume=1)
            agg.on_tick(dt(9, 6), price=1310.0, volume=1)  # triggers emit of 09:00

        finally:
            ba_module.maybe_merge_db_bar = original_mmdb

        assert len(captured_calls) == 1
        expected_ref = datetime(2024, 1, 15, 9, 5, 0)  # 09:00 + 5min
        assert captured_calls[0] == expected_ref


# ---------------------------------------------------------------------------
# DB merge - integration tests (test through data_quality module directly)
# ---------------------------------------------------------------------------


class TestDbMergeIntegration:
    """Test DB merge logic through data_quality.maybe_merge_db_bar() directly."""

    def test_no_merge_when_no_provider(self):
        from src.paper.data_quality import BarState, DataQualityConfig, maybe_merge_db_bar

        bar = {
            "datetime": dt(9, 0),
            "open": 1300.0,
            "high": 1320.0,
            "low": 1290.0,
            "close": 1310.0,
            "volume": 10.0,
        }
        bar_state = BarState(
            has_live_trade=True,
            trade_count=1,
            first_trade_ts=dt(9, 0, 1),
            last_trade_ts=dt(9, 0, 2),
            max_gap_seconds=0.0,
            bucket_start=dt(9, 0),
        )
        config = DataQualityConfig(stale_trade_seconds=60.0, min_live_updates=100, freq_minutes=5)

        result = maybe_merge_db_bar(bar, bar_state, dt(9, 5), config, None)
        assert result == bar

    def test_no_merge_when_quality_ok(self):
        from src.paper.data_quality import BarState, DataQualityConfig, maybe_merge_db_bar

        provider = MagicMock(
            return_value={
                "open": 1200.0,
                "high": 1400.0,
                "low": 1100.0,
                "close": 1250.0,
                "volume": 100.0,
            }
        )

        bar = {
            "datetime": dt(9, 0),
            "open": 1300.0,
            "high": 1320.0,
            "low": 1290.0,
            "close": 1310.0,
            "volume": 10.0,
        }
        bar_state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=dt(9, 0, 1),
            last_trade_ts=dt(9, 4, 59),
            max_gap_seconds=0.0,
            bucket_start=dt(9, 0),
        )
        config = DataQualityConfig(stale_trade_seconds=3600.0, min_live_updates=1, freq_minutes=5)

        result = maybe_merge_db_bar(bar, bar_state, dt(9, 5), config, provider)

        provider.assert_not_called()
        assert result == bar

    def test_merge_too_few_updates_uses_max_high_min_low(self):
        from src.paper.data_quality import BarState, DataQualityConfig, maybe_merge_db_bar

        db_bar = {"open": 1295.0, "high": 1325.0, "low": 1285.0, "close": 1305.0, "volume": 50.0}
        provider = MagicMock(return_value=db_bar)

        bar = {
            "datetime": dt(9, 0),
            "open": 1300.0,
            "high": 1320.0,
            "low": 1290.0,
            "close": 1310.0,
            "volume": 10.0,
        }
        bar_state = BarState(
            has_live_trade=True,
            trade_count=1,
            first_trade_ts=dt(9, 0, 1),
            last_trade_ts=dt(9, 0, 2),
            max_gap_seconds=0.0,
            bucket_start=dt(9, 0),
        )
        config = DataQualityConfig(stale_trade_seconds=3600.0, min_live_updates=100, freq_minutes=5)

        result = maybe_merge_db_bar(bar, bar_state, dt(9, 5), config, provider)

        assert result["high"] == 1325.0
        assert result["low"] == 1285.0
        assert result["volume"] == 50.0

    def test_merge_returns_bar_when_provider_returns_none(self):
        from src.paper.data_quality import BarState, DataQualityConfig, maybe_merge_db_bar

        provider = MagicMock(return_value=None)

        bar = {
            "datetime": dt(9, 0),
            "open": 1300.0,
            "high": 1320.0,
            "low": 1290.0,
            "close": 1310.0,
            "volume": 10.0,
        }
        bar_state = BarState(
            has_live_trade=True,
            trade_count=1,
            first_trade_ts=dt(9, 0, 1),
            last_trade_ts=dt(9, 0, 2),
            max_gap_seconds=0.0,
            bucket_start=dt(9, 0),
        )
        config = DataQualityConfig(stale_trade_seconds=3600.0, min_live_updates=100, freq_minutes=5)

        result = maybe_merge_db_bar(bar, bar_state, dt(9, 5), config, provider)
        assert result == bar

    def test_merge_provider_exception_returns_original_bar(self):
        from src.paper.data_quality import BarState, DataQualityConfig, maybe_merge_db_bar

        def failing_provider(bucket):
            raise RuntimeError("DB connection failed")

        bar = {
            "datetime": dt(9, 0),
            "open": 1300.0,
            "high": 1320.0,
            "low": 1290.0,
            "close": 1310.0,
            "volume": 10.0,
        }
        bar_state = BarState(
            has_live_trade=True,
            trade_count=1,
            first_trade_ts=dt(9, 0, 1),
            last_trade_ts=dt(9, 0, 2),
            max_gap_seconds=0.0,
            bucket_start=dt(9, 0),
        )
        config = DataQualityConfig(stale_trade_seconds=3600.0, min_live_updates=100, freq_minutes=5)

        result = maybe_merge_db_bar(bar, bar_state, dt(9, 5), config, failing_provider)
        assert result == bar

    def test_zero_volume_db_bar_still_merges_ohlc(self):
        from src.paper.data_quality import BarState, DataQualityConfig, maybe_merge_db_bar

        db_bar = {"open": 1295.0, "high": 1325.0, "low": 1285.0, "close": 1305.0, "volume": 0}
        provider = MagicMock(return_value=db_bar)

        bar = {
            "datetime": dt(9, 0),
            "open": 1300.0,
            "high": 1320.0,
            "low": 1290.0,
            "close": 1310.0,
            "volume": 10.0,
        }
        bar_state = BarState(
            has_live_trade=True,
            trade_count=1,
            first_trade_ts=dt(9, 0, 1),
            last_trade_ts=dt(9, 0, 2),
            max_gap_seconds=0.0,
            bucket_start=dt(9, 0),
        )
        config = DataQualityConfig(stale_trade_seconds=3600.0, min_live_updates=100, freq_minutes=5)

        result = maybe_merge_db_bar(bar, bar_state, dt(9, 5), config, provider)

        assert result["high"] == 1325.0
        assert result["low"] == 1285.0
        assert result["volume"] == 10.0


# ---------------------------------------------------------------------------
# _emit_bar - pipeline branch
# ---------------------------------------------------------------------------


class TestEmitBarPipeline:
    def test_pipeline_receives_history_dataframe(self):
        """If runtime_config has a 'pipeline', it must be called with a DataFrame of history."""
        import pandas as pd

        received_dfs = []

        def fake_pipeline_run(df):
            received_dfs.append(df)
            return df  # pass-through

        pipeline = MagicMock()
        pipeline.run.side_effect = fake_pipeline_run

        agg = make_aggregator(runtime_config={"pipeline": pipeline})
        agg.set_on_bar(lambda b: None)

        # Preload warmup history so pipeline gets called
        warmup_bars = [
            {
                "datetime": datetime(2024, 1, 15, 9, 0) + timedelta(minutes=i * 5),
                "open": 1300.0,
                "high": 1320.0,
                "low": 1290.0,
                "close": 1310.0,
                "volume": 100.0,
            }
            for i in range(15)  # More than atr_period + 1
        ]
        agg.preload_history(pd.DataFrame(warmup_bars))

        agg.on_tick(dt(10, 16), price=1300.0, volume=1)
        agg.on_tick(dt(10, 21), price=1310.0, volume=1)  # triggers emit

        pipeline.run.assert_called_once()
        df_arg = received_dfs[0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) >= 15

    def test_pipeline_enriched_bar_passed_to_callback(self):
        """The bar passed to on_bar must include indicator columns added by the pipeline."""
        import pandas as pd

        def enriching_pipeline(df):
            df = df.copy()
            df["atr_14"] = 7.5
            return df

        pipeline = MagicMock()
        pipeline.run.side_effect = enriching_pipeline

        received = []
        agg = make_aggregator(runtime_config={"pipeline": pipeline})
        agg.set_on_bar(received.append)

        # Preload warmup history
        warmup_bars = [
            {
                "datetime": datetime(2024, 1, 15, 9, 0) + timedelta(minutes=i * 5),
                "open": 1300.0,
                "high": 1320.0,
                "low": 1290.0,
                "close": 1310.0,
                "volume": 100.0,
            }
            for i in range(15)
        ]
        agg.preload_history(pd.DataFrame(warmup_bars))

        agg.on_tick(dt(10, 16), price=1300.0, volume=1)
        agg.on_tick(dt(10, 21), price=1310.0, volume=1)

        assert len(received) == 1
        assert received[0].get("atr_14") == pytest.approx(7.5)

    def test_pipeline_exception_is_swallowed_bar_still_emitted(self):
        """If pipeline raises, the exception must be caught and the bar still emitted."""
        import pandas as pd

        pipeline = MagicMock()
        pipeline.run.side_effect = RuntimeError("indicator explosion")

        received = []
        agg = make_aggregator(runtime_config={"pipeline": pipeline})
        agg.set_on_bar(received.append)

        # Preload warmup history
        warmup_bars = [
            {
                "datetime": datetime(2024, 1, 15, 9, 0) + timedelta(minutes=i * 5),
                "open": 1300.0,
                "high": 1320.0,
                "low": 1290.0,
                "close": 1310.0,
                "volume": 100.0,
            }
            for i in range(15)
        ]
        agg.preload_history(pd.DataFrame(warmup_bars))

        # Must not raise
        agg.on_tick(dt(10, 16), price=1300.0, volume=1)
        agg.on_tick(dt(10, 21), price=1310.0, volume=1)

        # Bar was still emitted even though pipeline failed
        assert len(received) == 1
        assert received[0]["close"] == 1300.0  # original values intact
