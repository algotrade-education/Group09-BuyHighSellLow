"""
Risk handler for paper trading.

Checks SL/TP, trailing stop, and force-flat triggers on each bar.
This is the second handler in the pipeline (after BarHandler, before SignalHandler).

Key responsibilities:
- Check stop loss and take profit triggers (SL checked before TP)
- Apply trailing stop updates when position is open
- Check force-flat triggers (ATC, preclose window, last candle, session boundary)
- Defer exits outside session hours when configured

Force-flat priority order:
1. ATC safety close (bar_time >= 14:30)
2. Session preclose window (force_flat_preclose_seconds)
3. Last candle heuristic (bar close time exits session)
4. Outside session with open position (safety net)

Deferred exit flow:
When a risk trigger fires outside session hours and defer_exit_outside_session=True,
on_bar returns (True, reason) so the engine can store the deferred reason.
The BarHandler will submit it when the session reopens.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.session.base import SessionManager
    from src.paper.account.tracker import Tracker
    from src.paper.execution.order_manager import OrderManager
    from src.paper.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class RiskHandlerConfig:
    """Configuration for risk handler behavior."""

    force_flat_on_session_close: bool = True
    force_flat_preclose_seconds: float = 15.0
    force_flat_on_last_candle: bool = True
    defer_exit_outside_session: bool = True
    freq_minutes: int = 5


class RiskHandler:
    """Handles risk-based exit triggers and force-flat logic.

    This handler runs after BarHandler and before SignalHandler in the pipeline.
    It checks for SL/TP triggers, applies trailing stops, and enforces session
    boundary rules to ensure positions are closed before market close.

    Return value of on_bar: (exit_triggered, deferred_reason)
    - exit_triggered: True if an exit was submitted or deferred (skip SignalHandler)
    - deferred_reason: non-None string when an exit was deferred to next session open;
      the engine stores this and BarHandler submits it when the session reopens.
    """

    def __init__(
        self,
        tracker: Tracker,
        order_manager: OrderManager,
        risk_manager: RiskManager,
        session_manager: SessionManager,
        config: RiskHandlerConfig,
        on_deferred_exit: Callable[[str | None], None] | None = None,
    ) -> None:
        """Initialize the risk handler.

        Args:
            tracker: Account tracker for position state.
            order_manager: Order manager for submitting exits.
            risk_manager: Risk manager for SL/TP/trailing stop checks.
            session_manager: Session manager for trading hours checks.
            config: Risk handler configuration.
            on_deferred_exit: Optional callback invoked with the deferred reason
                (or None to clear) when an exit cannot be submitted immediately.
                If None, deferred exits are silently dropped.
        """
        self._tracker = tracker
        self._order_manager = order_manager
        self._risk_manager = risk_manager
        self._session_manager = session_manager
        self._config = config
        self._on_deferred_exit = on_deferred_exit

    def set_deferred_exit_callback(self, callback: Callable[[str | None], None]) -> None:
        """Set or replace the deferred exit callback after construction.

        Allows the engine to wire itself in after both the handler and engine
        are constructed, avoiding the need for a placeholder None reference.

        Args:
            callback: Callable invoked with the deferred reason (or None to clear).
        """
        self._on_deferred_exit = callback

    def on_bar(self, bar: dict, bar_time: datetime) -> bool:
        """Process risk checks on a new bar.

        Checks SL/TP, applies trailing stop, and checks force-flat triggers.
        Returns True if an exit was triggered or deferred (skip signal generation).

        Args:
            bar: Bar dict containing OHLC and indicator data.
            bar_time: Bar timestamp (bucket start time).

        Returns:
            True if an exit was triggered or deferred, False otherwise.
        """
        if self._tracker.is_flat:
            return False

        # Check SL/TP triggers (SL checked before TP - conservative)
        if self._check_sl_tp(bar, bar_time):
            return True

        # Apply trailing stop (mutates position.stop_loss in place)
        self._risk_manager.apply_trailing_stop(self._tracker.position, bar)

        # Check force-flat triggers
        close = float(bar.get("close", 0.0))
        return bool(self._check_force_flat(bar_time, close))

    def _check_sl_tp(self, bar: dict, bar_time: datetime) -> bool:
        """Check for stop loss or take profit triggers.

        SL is checked before TP (conservative assumption - SL wins when both
        would trigger within the same bar).

        Args:
            bar: Bar dict containing OHLC data.
            bar_time: Bar timestamp.

        Returns:
            True if an exit was triggered, False otherwise.
        """
        exit_trigger = self._risk_manager.get_exit_trigger(self._tracker.position, bar)
        if exit_trigger:
            close = float(bar.get("close", 0.0))
            self._submit_exit_or_defer(
                reason=exit_trigger,
                price=close,
                bar_time=bar_time,
                ord_type="LIMIT",
            )
            return True
        return False

    def _check_force_flat(self, bar_time: datetime, close: float) -> bool:
        """Check force-flat triggers in priority order.

        Priority order:
        1. ATC safety close (bar_time >= 14:30)
        2. Session preclose window (force_flat_preclose_seconds)
        3. Last candle heuristic (bar close time exits session)
        4. Outside session with open position (safety net)

        Args:
            bar_time: Bar timestamp (bucket start time).
            close: Bar close price.

        Returns:
            True if a force-flat exit was triggered, False otherwise.
        """
        if not self._config.force_flat_on_session_close:
            return False

        reason: str | None = None

        # 1. ATC safety close - delegate to session_manager instead of hardcoding time(14, 30)
        if self._session_manager.is_atc(bar_time):
            reason = "ATC Safety Close"

        # 2. Session preclose window
        if reason is None and self._config.force_flat_preclose_seconds > 0:
            reason = self._session_manager.get_force_close_reason(
                dt=bar_time,
                preclose_seconds=self._config.force_flat_preclose_seconds,
            )

        # 3. Last candle heuristic (bar close time exits session)
        if reason is None and self._config.force_flat_on_last_candle:
            bar_close_time = bar_time + timedelta(minutes=self._config.freq_minutes)
            # Check if bar close time is outside session
            if not self._session_manager.is_trading_hours(bar_close_time):
                reason = "Last Candle Close"

        # 4. Outside session with open position (safety net)
        # Only check this if we haven't already found a reason
        if reason is None and not self._session_manager.is_trading_hours(bar_time):
            reason = "Session Boundary Close"

        if reason:
            logger.info(
                "Force-flat trigger: %s | bar_time=%s",
                reason,
                bar_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            self._submit_exit_or_defer(
                reason=reason,
                price=close,
                bar_time=bar_time,
                ord_type="LIMIT",
            )
            return True

        return False

    def _submit_exit_or_defer(
        self,
        reason: str,
        price: float,
        bar_time: datetime,
        ord_type: str,
    ) -> None:
        """Submit exit immediately or defer to next session open via callback.

        If outside session and defer_exit_outside_session is True, invokes
        on_deferred_exit(reason) so the engine can store it. BarHandler will
        submit the exit when the session reopens.

        Args:
            reason: Exit reason string.
            price: Exit price.
            bar_time: Bar timestamp.
            ord_type: Order type ("MARKET" or "LIMIT").
        """
        can_exit_now = self._session_manager.is_trading_hours(bar_time)

        # Also allow exits during ATC even if not in standard trading hours
        if not can_exit_now and self._session_manager.is_atc(bar_time):
            can_exit_now = True

        if can_exit_now:
            self._order_manager.submit_exit(
                reason=reason,
                price=price,
                ord_type=ord_type,
                timestamp=bar_time,
            )
            # Clear any pending deferred reason
            if self._on_deferred_exit is not None:
                self._on_deferred_exit(None)
            return

        # Outside session - defer or drop
        if self._config.defer_exit_outside_session:
            logger.warning(
                "Deferring exit (%s): bar_time=%s is outside trading session.",
                reason,
                bar_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            if self._on_deferred_exit is not None:
                self._on_deferred_exit(reason)
        else:
            logger.warning(
                "Skipping exit (%s): bar_time=%s is outside trading session.",
                reason,
                bar_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
