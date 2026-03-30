"""
Risk management for live paper trading.

Evaluates position and portfolio risk constraints during live trading:
- Stop loss and take profit trigger detection
- Trailing stop updates based on ATR
- Maximum daily loss enforcement
- Per-trade maximum loss limits

All risk checks are performed on each bar update. Trailing stops move in favor
of the trade using ATR-based distance calculation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RiskManager:
    """Evaluates position and portfolio risk constraints during live trading."""

    def __init__(
        self,
        use_trailing_stop: bool = False,
        trailing_atr_multiplier: float = 2.0,
        max_daily_loss_pct: float = 0.0,
        initial_capital: float = 0.0,
        max_loss_per_trade_pct: float = 0.0,
    ) -> None:
        """Initialize the risk manager.

        Args:
            use_trailing_stop: Enable ATR-based trailing stop updates.
            trailing_atr_multiplier: ATR multiplier for trailing stop distance.
            max_daily_loss_pct: Maximum daily loss as percentage of initial capital.
            initial_capital: Starting capital for loss limit calculations.
            max_loss_per_trade_pct: Maximum loss per trade as percentage of initial capital.
        """
        self.use_trailing_stop = use_trailing_stop
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.max_daily_loss_pct = max_daily_loss_pct
        self.initial_capital = initial_capital
        self.max_loss_per_trade_pct = max_loss_per_trade_pct
        self._max_daily_loss_amount = initial_capital * (max_daily_loss_pct / 100.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exit_trigger(self, position: Any, bar: dict[str, Any]) -> str | None:
        """Check whether SL, TP, or per-trade max loss should trigger on this bar.

        Checks stop loss before take profit (conservative assumption - SL wins
        when both would trigger within the same bar).

        Why SL before TP?
        Without tick-level data, we can't know which level was hit first when both
        trigger in the same bar. Checking SL first is conservative - it assumes the
        worst case (stopped out) rather than best case (profit taken). This prevents
        overly optimistic backtest results.

        Args:
            position: Current position object with is_flat, is_long, is_short properties.
            bar: Current bar dict with low, high, and optional unrealized_pnl.

        Returns:
            "Stop Loss", "Take Profit", "Max Trade Loss", or None.
        """
        if position.is_flat:
            return None

        bar_low = float(bar.get("low", 0.0) or 0.0)
        bar_high = float(bar.get("high", 0.0) or 0.0)

        # Check SL first (conservative - assume worst case when both trigger)
        if position.is_long:
            if position.stop_loss is not None and bar_low <= position.stop_loss:
                return "Stop Loss"
            if position.take_profit is not None and bar_high >= position.take_profit:
                return "Take Profit"

        elif position.is_short:
            if position.stop_loss is not None and bar_high >= position.stop_loss:
                return "Stop Loss"
            if position.take_profit is not None and bar_low <= position.take_profit:
                return "Take Profit"

        if self.max_loss_per_trade_pct != 0.0:
            threshold = self.initial_capital * self.max_loss_per_trade_pct / 100.0
            unrealized_loss = -position.unrealized_pnl  # positive when losing
            if unrealized_loss > threshold:
                return "Max Trade Loss"

        return None

    def apply_trailing_stop(self, position: Any, bar: dict[str, Any]) -> None:
        """Move stop loss in favor of the trade using ATR trailing distance.

        Updates position.stop_loss in place when the trailing stop calculation
        produces a more favorable level than the current stop loss.

        Trailing stop logic:
        - LONG: SL = close - (ATR * multiplier), only moves UP (ratchets)
        - SHORT: SL = close + (ATR * multiplier), only moves DOWN (ratchets)

        The stop "trails" the price at a fixed ATR distance, locking in profits
        as the trade moves favorably. It never moves against the trade direction.

        Args:
            position: Current position object with stop_loss, is_long, is_short properties.
            bar: Current bar dict containing close price and ATR indicator values.
        """
        if not self.use_trailing_stop or position.is_flat or position.stop_loss is None:
            return

        # Extract ATR from bar (supports any atr_* key like atr_14, atr_20, etc.)
        atr = 0.0
        for key, value in bar.items():
            if str(key).startswith("atr_") and value and value > 0:
                atr = float(value)
                break

        if atr <= 0:
            return

        trail_distance = self.trailing_atr_multiplier * atr
        close = float(bar.get("close", 0.0) or 0.0)

        # LONG: Trail below price, only move UP (lock in profits)
        if position.is_long:
            new_sl = close - trail_distance
            if new_sl > position.stop_loss:  # Only update if moving in our favor
                logger.info(
                    "Trailing stop updated: LONG SL %.2f -> %.2f",
                    position.stop_loss,
                    new_sl,
                )
                position.stop_loss = new_sl
        # SHORT: Trail above price, only move DOWN (lock in profits)
        elif position.is_short:
            new_sl = close + trail_distance
            if new_sl < position.stop_loss:  # Only update if moving in our favor
                logger.info(
                    "Trailing stop updated: SHORT SL %.2f -> %.2f",
                    position.stop_loss,
                    new_sl,
                )
                position.stop_loss = new_sl

    def is_daily_loss_hit(self, daily_pnl: float) -> bool:
        """Return True when the portfolio has breached its daily loss limit.

        Args:
            daily_pnl: Current daily P&L (negative values indicate losses).

        Returns:
            True if daily loss limit is exceeded, False otherwise.
        """
        if self._max_daily_loss_amount <= 0:
            return False
        return daily_pnl <= -self._max_daily_loss_amount
