"""Unit tests for merge_db_bar() - pure function merge logic.

Tests the DB bar merge logic in isolation without needing a BarAggregator instance.
This demonstrates the benefit of extracting merge logic to data_quality.py.
"""

from __future__ import annotations

from src.paper.data_quality import merge_db_bar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_live_bar(
    open_: float = 1300.0,
    high: float = 1320.0,
    low: float = 1290.0,
    close: float = 1310.0,
    volume: float = 10.0,
) -> dict:
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def make_db_bar(
    open_: float = 1295.0,
    high: float = 1325.0,
    low: float = 1285.0,
    close: float = 1305.0,
    volume: float = 50.0,
) -> dict:
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


# ---------------------------------------------------------------------------
# No quality issues - no merge
# ---------------------------------------------------------------------------


class TestNoMergeNeeded:
    def test_empty_reasons_returns_live_bar_unchanged(self):
        """When reasons is empty, live bar should be returned as-is."""
        live = make_live_bar()
        db = make_db_bar()
        result = merge_db_bar(live, db, reasons=[])

        assert result == live

    def test_no_reasons_preserves_all_live_values(self):
        live = make_live_bar(open_=1300.0, high=1320.0, low=1290.0, close=1310.0, volume=10.0)
        db = make_db_bar(open_=1250.0, high=1400.0, low=1200.0, close=1350.0, volume=100.0)
        result = merge_db_bar(live, db, reasons=[])

        assert result["open"] == 1300.0
        assert result["high"] == 1320.0
        assert result["low"] == 1290.0
        assert result["close"] == 1310.0
        assert result["volume"] == 10.0


# ---------------------------------------------------------------------------
# no_live_trade - replace all OHLC
# ---------------------------------------------------------------------------


class TestNoLiveTradeMerge:
    def test_replaces_all_ohlc_with_db_values(self):
        """When no_live_trade, all OHLC should come from DB."""
        live = make_live_bar(open_=1300.0, high=1320.0, low=1290.0, close=1310.0, volume=10.0)
        db = make_db_bar(open_=1250.0, high=1370.0, low=1230.0, close=1305.0, volume=80.0)
        result = merge_db_bar(live, db, reasons=["no_live_trade"])

        assert result["open"] == 1250.0
        assert result["high"] == 1370.0
        assert result["low"] == 1230.0
        assert result["close"] == 1305.0

    def test_volume_is_max_of_live_and_db(self):
        """Volume should always be max(live, db) even with no_live_trade."""
        live = make_live_bar(volume=100.0)
        db = make_db_bar(volume=50.0)
        result = merge_db_bar(live, db, reasons=["no_live_trade"])

        assert result["volume"] == 100.0  # max(100, 50)

    def test_no_live_trade_with_missing_db_keys_uses_live_fallback(self):
        """If DB bar is missing keys, use live values as fallback."""
        live = make_live_bar(open_=1300.0, high=1320.0, low=1290.0, close=1310.0)
        db = {"open": 1250.0}  # Missing high, low, close
        result = merge_db_bar(live, db, reasons=["no_live_trade"])

        assert result["open"] == 1250.0  # from DB
        assert result["high"] == 1320.0  # fallback to live
        assert result["low"] == 1290.0  # fallback to live
        assert result["close"] == 1310.0  # fallback to live


# ---------------------------------------------------------------------------
# start_gap - replace open only
# ---------------------------------------------------------------------------


class TestStartGapMerge:
    def test_replaces_open_with_db_open(self):
        """start_gap should replace only the open price."""
        live = make_live_bar(open_=1300.0, high=1320.0, low=1290.0, close=1310.0)
        db = make_db_bar(open_=1250.0, high=1325.0, low=1285.0, close=1305.0)
        result = merge_db_bar(live, db, reasons=["start_gap"])

        assert result["open"] == 1250.0  # replaced by DB
        assert result["close"] == 1310.0  # NOT replaced (no end_gap)

    def test_start_gap_updates_high_and_low(self):
        """start_gap should still apply high=max, low=min logic."""
        live = make_live_bar(high=1320.0, low=1290.0)
        db = make_db_bar(high=1325.0, low=1285.0)
        result = merge_db_bar(live, db, reasons=["start_gap"])

        assert result["high"] == 1325.0  # max(1320, 1325)
        assert result["low"] == 1285.0  # min(1290, 1285)

    def test_start_gap_updates_volume(self):
        """Volume should be max(live, db) with start_gap."""
        live = make_live_bar(volume=10.0)
        db = make_db_bar(volume=50.0)
        result = merge_db_bar(live, db, reasons=["start_gap"])

        assert result["volume"] == 50.0  # max(10, 50)


# ---------------------------------------------------------------------------
# end_gap - replace close only
# ---------------------------------------------------------------------------


class TestEndGapMerge:
    def test_replaces_close_with_db_close(self):
        """end_gap should replace only the close price."""
        live = make_live_bar(open_=1300.0, high=1320.0, low=1290.0, close=1310.0)
        db = make_db_bar(open_=1295.0, high=1325.0, low=1285.0, close=1308.0)
        result = merge_db_bar(live, db, reasons=["end_gap"])

        assert result["close"] == 1308.0  # replaced by DB
        assert result["open"] == 1300.0  # NOT replaced (no start_gap)

    def test_end_gap_updates_high_and_low(self):
        """end_gap should still apply high=max, low=min logic."""
        live = make_live_bar(high=1320.0, low=1290.0)
        db = make_db_bar(high=1325.0, low=1285.0)
        result = merge_db_bar(live, db, reasons=["end_gap"])

        assert result["high"] == 1325.0  # max(1320, 1325)
        assert result["low"] == 1285.0  # min(1290, 1285)


# ---------------------------------------------------------------------------
# Multiple reasons
# ---------------------------------------------------------------------------


class TestMultipleReasonsMerge:
    def test_start_gap_and_end_gap_replaces_both_open_and_close(self):
        """Both start_gap and end_gap should replace open and close."""
        live = make_live_bar(open_=1300.0, close=1310.0)
        db = make_db_bar(open_=1250.0, close=1305.0)
        result = merge_db_bar(live, db, reasons=["start_gap", "end_gap"])

        assert result["open"] == 1250.0  # replaced by DB
        assert result["close"] == 1305.0  # replaced by DB

    def test_too_few_updates_applies_high_low_volume_logic(self):
        """too_few_updates should apply high=max, low=min, volume=max."""
        live = make_live_bar(high=1320.0, low=1290.0, volume=10.0)
        db = make_db_bar(high=1325.0, low=1285.0, volume=50.0)
        result = merge_db_bar(live, db, reasons=["too_few_updates"])

        assert result["high"] == 1325.0
        assert result["low"] == 1285.0
        assert result["volume"] == 50.0

    def test_large_internal_gap_applies_high_low_volume_logic(self):
        """large_internal_gap should apply high=max, low=min, volume=max."""
        live = make_live_bar(high=1320.0, low=1290.0, volume=10.0)
        db = make_db_bar(high=1315.0, low=1295.0, volume=30.0)
        result = merge_db_bar(live, db, reasons=["large_internal_gap"])

        assert result["high"] == 1320.0  # max(1320, 1315) - live wins
        assert result["low"] == 1290.0  # min(1290, 1295) - live wins
        assert result["volume"] == 30.0  # max(10, 30) - DB wins

    def test_all_reasons_except_no_live_trade(self):
        """Multiple reasons should combine their effects."""
        live = make_live_bar(open_=1300.0, high=1320.0, low=1290.0, close=1310.0, volume=10.0)
        db = make_db_bar(open_=1250.0, high=1325.0, low=1285.0, close=1305.0, volume=50.0)
        result = merge_db_bar(
            live, db, reasons=["start_gap", "end_gap", "too_few_updates", "large_internal_gap"]
        )

        assert result["open"] == 1250.0  # start_gap
        assert result["close"] == 1305.0  # end_gap
        assert result["high"] == 1325.0  # max
        assert result["low"] == 1285.0  # min
        assert result["volume"] == 50.0  # max


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMergeEdgeCases:
    def test_db_bar_missing_volume_key(self):
        """If DB bar has no volume key, use live volume."""
        live = make_live_bar(volume=10.0)
        db = {"open": 1295.0, "high": 1325.0, "low": 1285.0, "close": 1305.0}
        result = merge_db_bar(live, db, reasons=["start_gap"])

        assert result["volume"] == 10.0  # max(10, 10) - DB missing, uses live

    def test_db_bar_zero_volume(self):
        """DB bar with volume=0 should still allow OHLC merge."""
        live = make_live_bar(volume=10.0)
        db = make_db_bar(volume=0.0)
        result = merge_db_bar(live, db, reasons=["start_gap"])

        assert result["open"] == 1295.0  # DB open used
        assert result["volume"] == 10.0  # max(10, 0) - live wins

    def test_live_bar_higher_high_than_db(self):
        """When live high > DB high, live should win."""
        live = make_live_bar(high=1350.0)
        db = make_db_bar(high=1325.0)
        result = merge_db_bar(live, db, reasons=["too_few_updates"])

        assert result["high"] == 1350.0  # max(1350, 1325)

    def test_live_bar_lower_low_than_db(self):
        """When live low < DB low, live should win."""
        live = make_live_bar(low=1280.0)
        db = make_db_bar(low=1285.0)
        result = merge_db_bar(live, db, reasons=["too_few_updates"])

        assert result["low"] == 1280.0  # min(1280, 1285)

    def test_preserves_datetime_if_present(self):
        """merge_db_bar should preserve datetime and other keys."""
        from datetime import datetime

        live = make_live_bar()
        live["datetime"] = datetime(2024, 1, 15, 9, 5, 0)
        live["custom_field"] = "test"
        db = make_db_bar()

        result = merge_db_bar(live, db, reasons=["start_gap"])

        assert result["datetime"] == datetime(2024, 1, 15, 9, 5, 0)
        assert result["custom_field"] == "test"


# ---------------------------------------------------------------------------
# Reason priority
# ---------------------------------------------------------------------------


class TestReasonPriority:
    def test_no_live_trade_overrides_other_reasons(self):
        """no_live_trade should replace all OHLC regardless of other reasons."""
        live = make_live_bar(open_=1300.0, high=1320.0, low=1290.0, close=1310.0)
        db = make_db_bar(open_=1250.0, high=1370.0, low=1230.0, close=1305.0)

        # Even with start_gap and end_gap, no_live_trade takes precedence
        result = merge_db_bar(live, db, reasons=["no_live_trade", "start_gap", "end_gap"])

        assert result["open"] == 1250.0
        assert result["high"] == 1370.0
        assert result["low"] == 1230.0
        assert result["close"] == 1305.0

    def test_start_gap_without_end_gap_preserves_close(self):
        """start_gap alone should not affect close."""
        live = make_live_bar(open_=1300.0, close=1310.0)
        db = make_db_bar(open_=1250.0, close=1240.0)
        result = merge_db_bar(live, db, reasons=["start_gap"])

        assert result["open"] == 1250.0  # replaced
        assert result["close"] == 1310.0  # preserved

    def test_end_gap_without_start_gap_preserves_open(self):
        """end_gap alone should not affect open."""
        live = make_live_bar(open_=1300.0, close=1310.0)
        db = make_db_bar(open_=1250.0, close=1305.0)
        result = merge_db_bar(live, db, reasons=["end_gap"])

        assert result["open"] == 1300.0  # preserved
        assert result["close"] == 1305.0  # replaced
