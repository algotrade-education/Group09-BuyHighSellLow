"""Unit tests for RiskHandler (src/paper/handlers/risk_handler.py).

Covers:
- on_bar returns False when position is flat
- on_bar calls risk_manager.get_exit_trigger and submits exit when triggered
- on_bar calls risk_manager.apply_trailing_stop when position is open
- _check_force_flat priority order (ATC > preclose > last candle > session boundary)
- _submit_exit_or_defer defers exit when outside session and defer_exit_outside_session=True
- _submit_exit_or_defer submits immediately when inside session
"""

from __future__ import annotations

from datetime import datetime, time
from unittest.mock import Mock

from src.paper.handlers.risk_handler import RiskHandler, RiskHandlerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(
    force_flat_on_session_close: bool = True,
    force_flat_preclose_seconds: float = 15.0,
    force_flat_on_last_candle: bool = True,
    defer_exit_outside_session: bool = True,
    freq_minutes: int = 5,
) -> RiskHandlerConfig:
    return RiskHandlerConfig(
        force_flat_on_session_close=force_flat_on_session_close,
        force_flat_preclose_seconds=force_flat_preclose_seconds,
        force_flat_on_last_candle=force_flat_on_last_candle,
        defer_exit_outside_session=defer_exit_outside_session,
        freq_minutes=freq_minutes,
    )


def make_tracker(is_flat: bool = False) -> Mock:
    tracker = Mock()
    tracker.is_flat = is_flat
    tracker.position = Mock()
    return tracker


def make_order_manager() -> Mock:
    order_manager = Mock()
    order_manager.submit_exit = Mock()
    return order_manager


def make_risk_manager(exit_trigger: str | None = None) -> Mock:
    risk_manager = Mock()
    risk_manager.get_exit_trigger = Mock(return_value=exit_trigger)
    risk_manager.apply_trailing_stop = Mock()
    return risk_manager


def make_session_manager(
    is_trading_hours: bool = True,
    get_force_close_reason: str | None = None,
    is_atc: bool = False,
) -> Mock:
    session_manager = Mock()
    session_manager.is_trading_hours = Mock(return_value=is_trading_hours)
    session_manager.get_force_close_reason = Mock(return_value=get_force_close_reason)
    session_manager.is_atc = Mock(return_value=is_atc)
    return session_manager


def make_engine() -> Mock:
    engine = Mock()
    engine._deferred_exit_reason = None
    return engine


def make_bar(
    open_: float = 1300.0,
    high: float = 1320.0,
    low: float = 1280.0,
    close: float = 1310.0,
) -> dict:
    return {"open": open_, "high": high, "low": low, "close": close}


# ---------------------------------------------------------------------------
# on_bar - flat position
# ---------------------------------------------------------------------------


class TestOnBarFlat:
    def test_returns_false_when_position_flat(self):
        """When position is flat, on_bar returns False immediately."""
        tracker = make_tracker(is_flat=True)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager()
        session_manager = make_session_manager()
        config = make_config()
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar()
        bar_time = datetime(2024, 1, 15, 9, 5)

        result = handler.on_bar(bar, bar_time)

        assert result is False
        risk_manager.get_exit_trigger.assert_not_called()
        risk_manager.apply_trailing_stop.assert_not_called()
        order_manager.submit_exit.assert_not_called()


# ---------------------------------------------------------------------------
# on_bar - SL/TP trigger
# ---------------------------------------------------------------------------


class TestOnBarSLTP:
    def test_returns_true_when_sl_triggered(self):
        """When SL triggers, on_bar returns True and submits exit."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger="Stop Loss")
        session_manager = make_session_manager(is_trading_hours=True)
        config = make_config()
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar(close=1310.0)
        bar_time = datetime(2024, 1, 15, 9, 5)

        result = handler.on_bar(bar, bar_time)

        assert result is True
        risk_manager.get_exit_trigger.assert_called_once_with(tracker.position, bar)
        order_manager.submit_exit.assert_called_once_with(
            reason="Stop Loss",
            price=1310.0,
            ord_type="LIMIT",
            timestamp=bar_time,
        )

    def test_applies_trailing_stop_when_no_exit_trigger(self):
        """When no exit trigger, applies trailing stop and checks force-flat."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger=None)
        session_manager = make_session_manager(is_trading_hours=True)
        config = make_config()
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar()
        bar_time = datetime(2024, 1, 15, 9, 5)

        result = handler.on_bar(bar, bar_time)

        assert result is False
        risk_manager.apply_trailing_stop.assert_called_once_with(tracker.position, bar)


# ---------------------------------------------------------------------------
# _check_force_flat - priority order
# ---------------------------------------------------------------------------


class TestCheckForceFlatPriority:
    def test_atc_safety_close_has_highest_priority(self):
        """ATC safety close (14:30) triggers before other checks."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger=None)
        session_manager = make_session_manager(is_trading_hours=True, is_atc=True)
        config = make_config()
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar(close=1310.0)
        bar_time = datetime(2024, 1, 15, 14, 30)  # ATC start

        result = handler.on_bar(bar, bar_time)

        assert result is True
        order_manager.submit_exit.assert_called_once()
        call_args = order_manager.submit_exit.call_args
        assert call_args[1]["reason"] == "ATC Safety Close"

    def test_preclose_window_triggers_when_no_atc(self):
        """Session preclose window triggers when not in ATC."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger=None)
        session_manager = make_session_manager(
            is_trading_hours=True,
            get_force_close_reason="Session preclose (10s remaining)",
        )
        config = make_config(force_flat_preclose_seconds=15.0)
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar(close=1310.0)
        bar_time = datetime(2024, 1, 15, 11, 29, 50)  # 10s before 11:30

        result = handler.on_bar(bar, bar_time)

        assert result is True
        session_manager.get_force_close_reason.assert_called_once_with(
            dt=bar_time,
            preclose_seconds=15.0,
        )

    def test_last_candle_triggers_when_bar_close_exits_session(self):
        """Last candle heuristic triggers when bar close time exits session."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger=None)

        # Session manager: bar_time is in session, but bar_close_time - 1s is not
        # Morning session ends at 11:30, so 11:29:59 should still be in session
        # but 11:30:00 should not be
        def is_trading_hours_side_effect(dt):
            # Morning session: 09:00 - 11:30 (exclusive end)
            if time(9, 0) <= dt.time() < time(11, 30):
                return True

            # Afternoon session: 13:00 - 14:30 (exclusive end)
            return time(13, 0) <= dt.time() < time(14, 30)

        session_manager = Mock()
        session_manager.is_trading_hours = Mock(side_effect=is_trading_hours_side_effect)
        session_manager.get_force_close_reason = Mock(return_value=None)
        session_manager.is_atc = Mock(return_value=False)

        config = make_config(force_flat_on_last_candle=True, freq_minutes=5)
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar(close=1310.0)
        # Bar at 11:25, closes at 11:30
        # Check at 11:29:59 should return False (outside session)
        bar_time = datetime(2024, 1, 15, 11, 25)

        result = handler.on_bar(bar, bar_time)

        assert result is True
        call_args = order_manager.submit_exit.call_args
        assert call_args[1]["reason"] == "Last Candle Close"


# ---------------------------------------------------------------------------
# _submit_exit_or_defer
# ---------------------------------------------------------------------------


class TestSubmitExitOrDefer:
    def test_submits_immediately_when_in_trading_hours(self):
        """When in trading hours, submits exit immediately."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger="Stop Loss")
        session_manager = make_session_manager(is_trading_hours=True)
        config = make_config()
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar(close=1310.0)
        bar_time = datetime(2024, 1, 15, 9, 5)

        handler.on_bar(bar, bar_time)

        order_manager.submit_exit.assert_called_once()
        assert engine._deferred_exit_reason is None

    def test_defers_exit_when_outside_session_and_defer_enabled(self):
        """When outside session and defer_exit_outside_session=True, stores reason in engine."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger="Stop Loss")
        session_manager = make_session_manager(is_trading_hours=False)
        config = make_config(defer_exit_outside_session=True)
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar(close=1310.0)
        bar_time = datetime(2024, 1, 15, 12, 0)  # outside session (lunch break)

        handler.on_bar(bar, bar_time)

        order_manager.submit_exit.assert_not_called()
        assert engine._deferred_exit_reason == "Stop Loss"

    def test_skips_exit_when_outside_session_and_defer_disabled(self):
        """When outside session and defer_exit_outside_session=False, skips exit."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger="Stop Loss")
        session_manager = make_session_manager(is_trading_hours=False)
        config = make_config(defer_exit_outside_session=False)
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar(close=1310.0)
        bar_time = datetime(2024, 1, 15, 12, 0)

        handler.on_bar(bar, bar_time)

        order_manager.submit_exit.assert_not_called()
        assert engine._deferred_exit_reason is None

    def test_allows_exit_during_atc_session(self):
        """Exits are allowed during ATC session even if not in standard trading hours."""
        tracker = make_tracker(is_flat=False)
        order_manager = make_order_manager()
        risk_manager = make_risk_manager(exit_trigger="Stop Loss")

        session_manager = Mock()
        session_manager.is_trading_hours = Mock(return_value=False)  # ATC not in standard hours
        session_manager.is_atc = Mock(return_value=True)  # But is ATC

        config = make_config()
        engine = make_engine()

        handler = RiskHandler(
            tracker=tracker,
            order_manager=order_manager,
            risk_manager=risk_manager,
            session_manager=session_manager,
            config=config,
            on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
        )

        bar = make_bar(close=1310.0)
        bar_time = datetime(2024, 1, 15, 14, 35)  # ATC period

        handler.on_bar(bar, bar_time)

        order_manager.submit_exit.assert_called_once()
        assert engine._deferred_exit_reason is None
