"""
Bar event handler for paper trading.

Processes each new bar by updating account state and handling deferred exits.
This is the first handler in the pipeline, ensuring equity tracking and P&L
updates occur before risk checks and signal generation.

Key responsibilities:
- Update unrealized P&L based on current bar close price
- Update daily P&L tracking (resets on new trading days)
- Record equity snapshots for performance metrics
- Submit deferred exit orders when session reopens

Deferred exits occur when a risk trigger fires outside trading hours.
The exit reason is stored in the engine's _deferred_exit_reason field,
and this handler submits the order as soon as the session opens.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.session.base import SessionManager
    from src.paper.account.tracker import Tracker
    from src.paper.execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


class BarHandler:
    """Handles bar-level account updates and deferred exit submission.

    This handler runs first in the pipeline to ensure equity and P&L state
    is current before downstream handlers (RiskHandler, SignalHandler) make
    decisions based on that state.

    The deferred exit mechanism allows risk triggers that fire outside trading
    hours to be queued and submitted when the market reopens, preventing
    stale exit orders from being rejected by the broker.
    """

    def __init__(
        self,
        tracker: Tracker,
        order_manager: OrderManager,
        session_manager: SessionManager,
    ) -> None:
        """Initialize the bar handler.

        Args:
            tracker: Account tracker for equity and P&L updates.
            order_manager: Order manager for submitting deferred exits.
            session_manager: Session manager for trading hours checks.
        """
        self._tracker = tracker
        self._order_manager = order_manager
        self._session_manager = session_manager

    def on_bar(
        self,
        bar: dict,
        bar_time: datetime,
        deferred_exit_reason: str | None,
    ) -> tuple[bool, str | None]:
        """Process a new bar: update equity, handle deferred exits.

        This method performs three core operations on every bar:
        1. Update unrealized P&L based on current close price
        2. Update daily P&L tracking (resets on new trading day)
        3. Record equity snapshot for performance metrics

        If a deferred exit is pending and the session is open, submits the
        exit order and clears the deferred reason.

        Args:
            bar: Bar dict containing OHLC and indicator data.
            bar_time: Bar timestamp (bucket start time).
            deferred_exit_reason: Exit reason deferred from previous bar (or None).

        Returns:
            Tuple of (exit_submitted, remaining_deferred_reason):
            - exit_submitted: True if a deferred exit was submitted
            - remaining_deferred_reason: None if exit was submitted, otherwise
              the original deferred_exit_reason (to be stored back in engine)
        """
        close = float(bar.get("close", 0.0))

        # Update account state on every bar
        self._tracker.update_unrealized(close)
        self._tracker.update_daily_pnl(bar_time)
        self._tracker.equity_snapshot(bar_time)

        # Check for deferred exit submission
        if deferred_exit_reason and self._session_manager.is_trading_hours(bar_time):
            logger.info(
                "on_bar: submitting deferred exit (reason=%s) at bar_time=%s close=%.2f",
                deferred_exit_reason,
                bar_time,
                close,
            )
            self._order_manager.submit_exit(
                reason=deferred_exit_reason,
                price=close,
                ord_type="LIMIT",
                timestamp=bar_time,
            )
            return True, None  # Exit submitted, clear deferred reason

        # No deferred exit submitted - pass through the deferred reason unchanged
        return False, deferred_exit_reason
