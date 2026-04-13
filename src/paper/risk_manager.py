"""
Risk management for live paper trading.

Owns only what is unique to live/paper trading:
- Maximum daily loss enforcement
- Per-trade maximum loss limits

SL/TP detection and trailing stop updates are delegated to the Position object
(check_stop_loss / check_take_profit) and AccountState._update_trailing_stop,
which already implement this logic for the backtesting engine.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RiskManager:
    """Evaluates portfolio-level risk constraints during live trading.

    Intentionally does NOT reimplement SL/TP or trailing stop logic —
    those live on Position (check_stop_loss/check_take_profit) and are
    called directly by RiskHandler.
    """

    def __init__(
        self,
        use_trailing_stop: bool = False,
        trailing_atr_multiplier: float = 2.0,
        max_daily_loss_fraction: float = 0.0,
        initial_capital: float = 0.0,
        max_loss_per_trade_fraction: float = 0.0,
    ) -> None:
        """Initialize the risk manager.

        Args:
            use_trailing_stop: Enable ATR-based trailing stop updates.
            trailing_atr_multiplier: ATR multiplier for trailing stop distance.
            max_daily_loss_fraction: Maximum daily loss as a fraction of initial capital
                (e.g. 0.02 = 2%). Matches RiskConfig.max_daily_loss schema units.
            initial_capital: Starting capital for loss limit calculations.
            max_loss_per_trade_fraction: Maximum loss per trade as a fraction of initial
                capital (e.g. 0.01 = 1%). 0.0 disables the check.
        """
        self.use_trailing_stop = use_trailing_stop
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.max_daily_loss_fraction = max_daily_loss_fraction
        self.initial_capital = initial_capital
        self.max_loss_per_trade_fraction = max_loss_per_trade_fraction
        self._max_daily_loss_amount = initial_capital * max_daily_loss_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exit_trigger(self, position: Any, bar: dict[str, Any]) -> str | None:
        """Check whether SL, TP, or per-trade max loss should trigger on this bar.

        Delegates SL/TP detection to Position.check_stop_loss / check_take_profit,
        which are the canonical implementations shared with the backtesting engine.
        SL is checked before TP (conservative — assumes worst case when both trigger
        in the same bar without tick-level data).

        Args:
            position: Current Position object.
            bar: Current bar dict with low, high, and optional unrealized_pnl.

        Returns:
            "Stop Loss", "Take Profit", "Max Trade Loss", or None.
        """
        if position.is_flat:
            return None

        bar_low = float(bar.get("low", 0.0) or 0.0)
        bar_high = float(bar.get("high", 0.0) or 0.0)

        # SL: check low for long, high for short (same logic as AccountState.check_sl_tp)
        sl_price = bar_low if position.is_long else bar_high
        if position.check_stop_loss(sl_price):
            return "Stop Loss"

        # TP: check high for long, low for short
        tp_price = bar_high if position.is_long else bar_low
        if position.check_take_profit(tp_price):
            return "Take Profit"

        if self.max_loss_per_trade_fraction != 0.0:
            threshold = self.initial_capital * self.max_loss_per_trade_fraction
            unrealized_loss = -position.unrealized_pnl  # positive when losing
            if unrealized_loss > threshold:
                return "Max Trade Loss"

        return None

    def apply_trailing_stop(self, position: Any, bar: dict[str, Any]) -> None:
        """Move stop loss in favor of the trade using ATR trailing distance.

        Reuses the same ATR extraction and ratchet logic as
        AccountState._update_trailing_stop. Updates position.stop_loss in place.

        Args:
            position: Current Position object.
            bar: Current bar dict containing close price and ATR indicator values.
        """
        if not self.use_trailing_stop or position.is_flat or position.stop_loss is None:
            return

        # Extract ATR from bar (supports any atr_* key: atr_14, atr_20, etc.)
        atr = next(
            (float(v) for k, v in bar.items() if str(k).startswith("atr_") and v and float(v) > 0),
            0.0,
        )
        if atr <= 0:
            return

        trail_distance = self.trailing_atr_multiplier * atr
        close = float(bar.get("close", 0.0) or 0.0)

        if position.is_long:
            new_sl = close - trail_distance
            if new_sl > position.stop_loss:
                logger.info(
                    "Trailing stop updated: LONG SL %.2f -> %.2f", position.stop_loss, new_sl
                )
                position.stop_loss = new_sl
        elif position.is_short:
            new_sl = close + trail_distance
            if new_sl < position.stop_loss:
                logger.info(
                    "Trailing stop updated: SHORT SL %.2f -> %.2f", position.stop_loss, new_sl
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
