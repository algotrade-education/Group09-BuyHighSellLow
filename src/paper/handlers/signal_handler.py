"""
Signal handler for paper trading.

Converts strategy signals into order submissions with proper guard checks.
This is the third handler in the pipeline (after BarHandler and RiskHandler).

Key responsibilities:
- Generate strategy signals by calling strategy.generate_signal()
- Apply entry guards in order: session skip → daily loss → entry cutoff
- Submit entry orders for LONG/SHORT signals when flat
- Submit exit orders for EXIT signals when not flat
- CLOSE signals bypass daily loss and entry cutoff guards

Guard order:
1. session_manager.should_skip_signal_generation() → skip all signals
2. risk_manager.is_daily_loss_hit() → skip entries only (CLOSE bypasses)
3. session_manager.is_entry_blocked() → skip entries only (CLOSE bypasses)

Signal-to-order mapping:
- LONG/SHORT when flat → order_manager.submit_entry()
- EXIT when not flat → order_manager.submit_exit()
- HOLD → no action
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from config.constants import VN30F_CONTRACT_MULTIPLIER

if TYPE_CHECKING:
    from src.engine.account.sizer import PositionSizer
    from src.engine.session.base import SessionManager
    from src.paper.account.tracker import Tracker
    from src.paper.execution.order_manager import OrderManager
    from src.paper.risk_manager import RiskManager
    from src.strategy.base import StrategyBase

logger = logging.getLogger(__name__)


@dataclass
class SignalHandlerConfig:
    """Configuration for signal handler behavior."""

    entry_cutoff_seconds: float = 0.0
    allow_late_entry: bool = False
    contract_multiplier: float = VN30F_CONTRACT_MULTIPLIER


class SignalHandler:
    """Handles strategy signal generation and order submission.

    This handler runs after BarHandler and RiskHandler in the pipeline.
    It generates strategy signals and converts them to orders, applying
    entry guards to prevent trading in unfavorable conditions.

    The handler applies three guards in order:
    1. Session skip check (blocks all signals)
    2. Daily loss check (blocks entries only)
    3. Entry cutoff check (blocks entries only)

    CLOSE signals bypass guards 2 and 3, allowing exits even when
    entries are blocked.
    """

    def __init__(
        self,
        strategy: StrategyBase,
        tracker: Tracker,
        order_manager: OrderManager,
        risk_manager: RiskManager,
        session_manager: SessionManager,
        position_sizer: PositionSizer,
        config: SignalHandlerConfig,
    ) -> None:
        """Initialize the signal handler.

        Args:
            strategy: Trading strategy for signal generation.
            tracker: Account tracker for position state.
            order_manager: Order manager for submitting orders.
            risk_manager: Risk manager for daily loss checks.
            session_manager: Session manager for trading hours checks.
            position_sizer: Position sizer for calculating order quantities.
            config: Signal handler configuration.
        """
        self._strategy = strategy
        self._tracker = tracker
        self._order_manager = order_manager
        self._risk_manager = risk_manager
        self._session_manager = session_manager
        self._position_sizer = position_sizer
        self._config = config

    def on_bar(self, bar: dict, bar_time: datetime) -> None:
        """Process strategy signal generation on a new bar.

        Generates a signal from the strategy and applies entry guards before
        submitting orders. CLOSE signals bypass daily loss and entry cutoff
        guards to ensure positions can be exited even when entries are blocked.

        Args:
            bar: Bar dict containing OHLC and indicator data.
            bar_time: Bar timestamp (bucket start time).
        """
        # Guard 1: Session skip check (blocks all signals)
        if self._session_manager.should_skip_signal(bar_time):
            logger.debug(
                "on_bar: skipping signal generation (session skip) at bar_time=%s",
                bar_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return

        # Generate signal from strategy
        position_snapshot = self._tracker.position.to_snapshot()
        signal = self._strategy.generate_signal(bar, position_snapshot, is_warmup=False)

        # HOLD signal - no action
        if signal.is_hold:
            return

        # EXIT signal - submit exit if not flat (bypasses guards 2 and 3)
        if signal.is_exit:
            if not self._tracker.is_flat:
                logger.info(
                    "on_bar: EXIT signal received at bar_time=%s reason=%s",
                    bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                    signal.reason or "strategy exit",
                )
                self._order_manager.submit_exit(
                    reason=signal.reason or "Signal Exit",
                    price=signal.entry_price if signal.entry_price > 0 else None,
                    ord_type=signal.ord_type,
                    timestamp=bar_time,
                )
            else:
                logger.debug(
                    "on_bar: EXIT signal ignored (already flat) at bar_time=%s",
                    bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            return

        # LONG/SHORT entry signals - apply guards 2 and 3
        if signal.is_entry:
            # Guard 2: Daily loss check (blocks entries only)
            if self._risk_manager.is_daily_loss_hit(self._tracker.daily_pnl):
                logger.warning(
                    "on_bar: skipping %s entry (daily loss limit hit) at bar_time=%s daily_pnl=%.2f",
                    signal.signal.upper(),
                    bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                    self._tracker.daily_pnl,
                )
                return

            # Guard 3: Entry cutoff check (blocks entries only)
            if self._session_manager.is_entry_blocked(
                bar_time,
                self._config.entry_cutoff_seconds,
                self._config.allow_late_entry,
            ):
                logger.warning(
                    "on_bar: skipping %s entry (entry cutoff window) at bar_time=%s",
                    signal.signal.upper(),
                    bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                return

            # All guards passed - submit entry if flat
            if self._tracker.is_flat:
                # Calculate position size
                atr_val = bar.get("atr_14")
                atr_float = float(atr_val) if atr_val is not None else 0.0

                qty = self._position_sizer.calculate_size(
                    equity=self._tracker.equity,
                    entry_price=signal.entry_price or float(bar.get("close", 0.0)),
                    stop_loss=signal.stop_loss if signal.stop_loss > 0 else None,
                    contract_multiplier=self._config.contract_multiplier,
                    atr=atr_float,
                )

                logger.info(
                    "on_bar: %s entry signal at bar_time=%s qty=%d entry_price=%.2f sl=%.2f tp=%.2f reason=%s",
                    signal.signal.upper(),
                    bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                    qty,
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.reason or "strategy entry",
                )

                self._order_manager.submit_entry(
                    signal=signal,
                    qty=qty,
                    bar=bar,
                    timestamp=bar_time,
                )
            else:
                logger.debug(
                    "on_bar: %s entry signal ignored (position already open) at bar_time=%s",
                    signal.signal.upper(),
                    bar_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            return

        # Unknown signal type - log warning
        logger.warning(
            "on_bar: unknown signal type %s at bar_time=%s",
            signal.signal,
            bar_time.strftime("%Y-%m-%d %H:%M:%S"),
        )
