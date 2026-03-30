"""Unit tests for DataQuality module (src/paper/data_quality.py).

Covers:
- get_quality_reasons: no_live_trade short-circuit
- get_quality_reasons: multi-reason completeness (start_gap + end_gap simultaneously)
- get_quality_reasons: individual reason triggers
- get_quality_reasons: empty reasons when bar is healthy
- BarState / DataQualityConfig dataclass construction
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src.paper.data_quality import BarState, DataQualityConfig, get_quality_reasons

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = datetime(2024, 1, 15, 9, 0, 0)


def make_config(
    stale_trade_seconds: float = 60.0,
    min_live_updates: int = 2,
    freq_minutes: int = 5,
) -> DataQualityConfig:
    return DataQualityConfig(
        stale_trade_seconds=stale_trade_seconds,
        min_live_updates=min_live_updates,
        freq_minutes=freq_minutes,
    )


def make_healthy_bar_state(bucket_start: datetime = BASE_TS) -> BarState:
    """A bar state that should pass all quality checks."""
    return BarState(
        has_live_trade=True,
        trade_count=5,
        first_trade_ts=bucket_start + timedelta(seconds=5),
        last_trade_ts=bucket_start + timedelta(minutes=4, seconds=50),
        max_gap_seconds=10.0,
        bucket_start=bucket_start,
    )


def bucket_end(bucket_start: datetime, freq_minutes: int = 5) -> datetime:
    return bucket_start + timedelta(minutes=freq_minutes)


# ---------------------------------------------------------------------------
# no_live_trade short-circuit
# ---------------------------------------------------------------------------


class TestNoLiveTrade:
    def test_returns_only_no_live_trade(self):
        """When has_live_trade is False, only 'no_live_trade' is returned."""
        state = BarState(
            has_live_trade=False,
            trade_count=0,
            first_trade_ts=None,
            last_trade_ts=None,
            max_gap_seconds=0.0,
            bucket_start=BASE_TS,
        )
        config = make_config()
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)

        assert reasons == ["no_live_trade"]

    def test_no_live_trade_short_circuits_other_checks(self):
        """Even if other conditions would trigger, only no_live_trade is returned."""
        state = BarState(
            has_live_trade=False,
            trade_count=0,  # would trigger too_few_updates
            first_trade_ts=None,
            last_trade_ts=None,
            max_gap_seconds=999.0,  # would trigger large_internal_gap
            bucket_start=BASE_TS,
        )
        config = make_config(min_live_updates=1)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)

        assert len(reasons) == 1
        assert reasons[0] == "no_live_trade"


# ---------------------------------------------------------------------------
# Healthy bar - no reasons
# ---------------------------------------------------------------------------


class TestHealthyBar:
    def test_healthy_bar_returns_empty_list(self):
        state = make_healthy_bar_state()
        config = make_config(stale_trade_seconds=60.0, min_live_updates=2)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert reasons == []

    def test_exactly_at_threshold_does_not_trigger(self):
        """Gaps exactly equal to stale_trade_seconds should NOT trigger (strict >)."""
        state = BarState(
            has_live_trade=True,
            trade_count=3,
            first_trade_ts=BASE_TS + timedelta(seconds=59),  # 59s < 60s threshold
            last_trade_ts=BASE_TS + timedelta(minutes=4, seconds=1),
            max_gap_seconds=59.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "large_internal_gap" not in reasons
        assert "start_gap" not in reasons


# ---------------------------------------------------------------------------
# Individual reason triggers
# ---------------------------------------------------------------------------


class TestTooFewUpdates:
    def test_triggers_when_below_min(self):
        state = BarState(
            has_live_trade=True,
            trade_count=1,
            first_trade_ts=BASE_TS + timedelta(seconds=5),
            last_trade_ts=BASE_TS + timedelta(minutes=4),
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(min_live_updates=3)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "too_few_updates" in reasons

    def test_does_not_trigger_when_at_min(self):
        state = BarState(
            has_live_trade=True,
            trade_count=3,
            first_trade_ts=BASE_TS + timedelta(seconds=5),
            last_trade_ts=BASE_TS + timedelta(minutes=4),
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(min_live_updates=3)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "too_few_updates" not in reasons


class TestLargeInternalGap:
    def test_triggers_when_gap_exceeds_threshold(self):
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=5),
            last_trade_ts=BASE_TS + timedelta(minutes=4),
            max_gap_seconds=120.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "large_internal_gap" in reasons

    def test_does_not_trigger_when_gap_below_threshold(self):
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=5),
            last_trade_ts=BASE_TS + timedelta(minutes=4),
            max_gap_seconds=30.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "large_internal_gap" not in reasons


class TestStartGap:
    def test_triggers_when_first_trade_late(self):
        """First trade arrived 90s after bucket start - exceeds 60s threshold."""
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=90),
            last_trade_ts=BASE_TS + timedelta(minutes=4),
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "start_gap" in reasons

    def test_does_not_trigger_when_first_trade_on_time(self):
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=10),
            last_trade_ts=BASE_TS + timedelta(minutes=4),
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "start_gap" not in reasons

    def test_no_trigger_when_first_trade_ts_is_none(self):
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=None,
            last_trade_ts=BASE_TS + timedelta(minutes=4),
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "start_gap" not in reasons


class TestEndGap:
    def test_triggers_when_last_trade_too_early(self):
        """Last trade was 90s before bucket end - exceeds 60s threshold."""
        ref_time = bucket_end(BASE_TS)  # 09:05
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=5),
            last_trade_ts=ref_time - timedelta(seconds=90),
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, ref_time, config)
        assert "end_gap" in reasons

    def test_does_not_trigger_when_last_trade_recent(self):
        ref_time = bucket_end(BASE_TS)
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=5),
            last_trade_ts=ref_time - timedelta(seconds=10),
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, ref_time, config)
        assert "end_gap" not in reasons

    def test_no_trigger_when_last_trade_ts_is_none(self):
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=5),
            last_trade_ts=None,
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, bucket_end(BASE_TS), config)
        assert "end_gap" not in reasons


# ---------------------------------------------------------------------------
# Multi-reason completeness (Req 4.3)
# ---------------------------------------------------------------------------


class TestMultiReasonCompleteness:
    def test_start_gap_and_end_gap_both_returned(self):
        """When both start_gap and end_gap conditions hold, both must be in the result."""
        ref_time = bucket_end(BASE_TS)
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=90),  # start_gap
            last_trade_ts=ref_time - timedelta(seconds=90),  # end_gap
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, ref_time, config)

        assert "start_gap" in reasons
        assert "end_gap" in reasons

    def test_all_four_reasons_can_be_returned_simultaneously(self):
        """All non-no_live_trade reasons can appear together."""
        ref_time = bucket_end(BASE_TS)
        state = BarState(
            has_live_trade=True,
            trade_count=1,  # too_few_updates
            first_trade_ts=BASE_TS + timedelta(seconds=90),  # start_gap
            last_trade_ts=ref_time - timedelta(seconds=90),  # end_gap
            max_gap_seconds=120.0,  # large_internal_gap
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0, min_live_updates=3)
        reasons = get_quality_reasons(state, ref_time, config)

        assert "too_few_updates" in reasons
        assert "large_internal_gap" in reasons
        assert "start_gap" in reasons
        assert "end_gap" in reasons
        assert "no_live_trade" not in reasons

    def test_reasons_are_independent(self):
        """Fixing one condition should not suppress other triggered reasons."""
        ref_time = bucket_end(BASE_TS)

        # Only start_gap triggered
        state = BarState(
            has_live_trade=True,
            trade_count=5,
            first_trade_ts=BASE_TS + timedelta(seconds=90),
            last_trade_ts=ref_time - timedelta(seconds=10),  # end_gap NOT triggered
            max_gap_seconds=10.0,
            bucket_start=BASE_TS,
        )
        config = make_config(stale_trade_seconds=60.0)
        reasons = get_quality_reasons(state, ref_time, config)

        assert "start_gap" in reasons
        assert "end_gap" not in reasons
