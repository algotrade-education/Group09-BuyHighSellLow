"""
Risk Management for live paper trading.

Handles trailing stops, SL/TP triggers, and maximum daily loss constraints.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """Evaluates position and portfolio risk constraints during live trading."""

    def __init__(
        self,
        use_trailing_stop: bool = False,
        trailing_atr_multiplier: float = 2.0,
        max_daily_loss_pct: float = 0.0,
        initial_capital: float = 0.0,
    ):
        self.use_trailing_stop = use_trailing_stop
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.max_daily_loss_amount = initial_capital * (max_daily_loss_pct / 100.0)
        self.max_daily_loss_pct = max_daily_loss_pct

    def is_daily_loss_hit(self, daily_pnl: float) -> bool:
        """Check if the portfolio has breached its daily loss limit."""
        if self.max_daily_loss_amount <= 0:
            return False
        return daily_pnl <= -self.max_daily_loss_amount

    def apply_trailing_stop(self, position: Any, bar: Dict[str, Any]) -> None:
        """
        Move stop loss in favor of the trade using ATR trailing distance.
        Mutates the position's stop_loss in place.
        """
        if not self.use_trailing_stop or position.is_flat or position.stop_loss is None:
            return

        atr = 0.0
        for key, value in bar.items():
            if str(key).startswith("atr_") and value and value > 0:
                atr = float(value)
                break

        if atr <= 0:
            return

        trail_distance = self.trailing_atr_multiplier * atr
        close = float(bar.get("close", 0.0) or 0.0)

        if position.is_long:
            new_sl = close - trail_distance
            if new_sl > position.stop_loss:
                logger.info(
                    "Trailing stop updated: LONG SL %.2f -> %.2f",
                    position.stop_loss,
                    new_sl,
                )
                position.stop_loss = new_sl
        elif position.is_short:
            new_sl = close + trail_distance
            if new_sl < position.stop_loss:
                logger.info(
                    "Trailing stop updated: SHORT SL %.2f -> %.2f",
                    position.stop_loss,
                    new_sl,
                )
                position.stop_loss = new_sl

    def get_exit_trigger(self, position: Any, bar: Dict[str, Any]) -> Optional[str]:
        """
        Check whether SL or TP should be triggered on this bar.

        Checks the bar's low (for long SL) and high (for long TP) - same
        conservative logic as the backtester.

        Returns:
            'Stop Loss', 'Take Profit', or None.
        """
        if position.is_flat:
            return None

        bar_low = float(bar.get("low", 0.0) or 0.0)
        bar_high = float(bar.get("high", 0.0) or 0.0)

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

        return None
