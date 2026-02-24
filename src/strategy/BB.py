"""
Bollinger Band Mean Reversion Strategy with ADX Filter.

Entry logic:
    - Long: Close <= BB Lower AND ADX < threshold (range-bound) AND RSI < oversold
    - Short: Close >= BB Upper AND ADX < threshold AND RSI > overbought

Exit logic:
    - Take Profit: BB Middle band (mean reversion target)
    - Stop Loss: ATR-based (atr_sl_multiplier * ATR beyond entry)
    - EOD close handled by the engine's session manager

All parameters are constructor kwargs for easy optimization via GridSearch.
"""

import logging
from typing import Any, Dict, Optional

from src.strategy.base import Signal, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class BollingerMeanReversion(Strategy):
    """
    Bollinger Band Mean Reversion strategy.

    Enters when price touches outer bands in a range-bound market
    (low ADX) and RSI confirms oversold/overbought.
    Targets the middle band (SMA) as the mean reversion exit.
    Uses ATR-based stop loss for adaptive risk management.
    """

    # Required indicator columns for bar validation
    REQUIRED_FIELDS = [
        "close",
        "bb_upper",
        "bb_lower",
        "bb_middle",
    ]

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        **kwargs,
    ):
        """
        Initialize the Bollinger Mean Reversion strategy.

        Args:
            bb_period: Bollinger Bands SMA period
            bb_std: Bollinger Bands standard deviation multiplier
            adx_period: ADX indicator period
            adx_threshold: Max ADX value to allow entries (lower = more range-bound)
            rsi_period: RSI indicator period
            rsi_oversold: RSI level below which the market is oversold (long entry)
            rsi_overbought: RSI level above which the market is overbought (short entry)
            atr_period: ATR indicator period
            atr_sl_multiplier: ATR multiplier for stop loss distance
        """
        super().__init__(name="BollingerMeanReversion")

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier

        # Store all params for serialization / optimization
        self._params = {
            "bb_period": bb_period,
            "bb_std": bb_std,
            "adx_period": adx_period,
            "adx_threshold": adx_threshold,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "atr_period": atr_period,
            "atr_sl_multiplier": atr_sl_multiplier,
        }

        logger.info("Strategy params: %s", self._params)

    def generate_signal(
        self,
        bar: Dict[str, Any],
        current_position: Optional[Any] = None,
    ) -> TradeSignal:
        """
        Generate a trading signal based on BB, ADX, and RSI conditions.

        Args:
            bar: Dict containing OHLC + indicator values for the current bar.
                 Expected keys: close, bb_upper, bb_lower, bb_middle,
                 adx_{period}, rsi_{period}, atr_{period}
            current_position: Current position object from TradeManager

        Returns:
            TradeSignal with entry/exit details
        """
        # Validate required fields
        if not self.validate_bar(bar, self.REQUIRED_FIELDS):
            return TradeSignal(signal=Signal.HOLD)

        close = bar["close"]
        bb_upper = bar["bb_upper"]
        bb_lower = bar["bb_lower"]
        bb_middle = bar["bb_middle"]

        # Get indicator values with safe defaults
        adx_col = f"adx_{self.adx_period}"
        rsi_col = f"rsi_{self.rsi_period}"
        atr_col = f"atr_{self.atr_period}"

        adx = bar.get(adx_col, 50.0)  # Default high ADX = no trade
        rsi = bar.get(rsi_col, 50.0)  # Default neutral RSI
        atr = bar.get(atr_col, 0.0)

        # If already in a position, hold (exits via SL/TP/EOD)
        if current_position is not None and not current_position.is_flat:
            return TradeSignal(signal=Signal.HOLD)

        # --- ENTRY CONDITIONS ---
        is_range_bound = adx < self.adx_threshold

        if not is_range_bound:
            return TradeSignal(signal=Signal.HOLD, reason="ADX too high (trending)")

        # Guard: ATR must be positive for stop loss calculation
        if atr <= 0:
            return TradeSignal(signal=Signal.HOLD, reason="ATR is zero or negative")

        sl_distance = self.atr_sl_multiplier * atr

        # --- LONG ENTRY ---
        if close <= bb_lower and rsi < self.rsi_oversold:
            stop_loss = close - sl_distance
            take_profit = bb_middle  # Mean reversion target

            # Sanity: TP must be above entry
            if take_profit <= close:
                return TradeSignal(signal=Signal.HOLD, reason="TP <= entry (skip)")

            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0.0,  # 0 = market order (fill at next open)
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"BB Lower touch (RSI={rsi:.1f}, ADX={adx:.1f})",
                metadata={
                    "bb_lower": bb_lower,
                    "bb_middle": bb_middle,
                    "atr": atr,
                    "adx": adx,
                    "rsi": rsi,
                },
            )

        # --- SHORT ENTRY ---
        if close >= bb_upper and rsi > self.rsi_overbought:
            stop_loss = close + sl_distance
            take_profit = bb_middle  # Mean reversion target

            # Sanity: TP must be below entry
            if take_profit >= close:
                return TradeSignal(signal=Signal.HOLD, reason="TP >= entry (skip)")

            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=0.0,  # Market order
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"BB Upper touch (RSI={rsi:.1f}, ADX={adx:.1f})",
                metadata={
                    "bb_upper": bb_upper,
                    "bb_middle": bb_middle,
                    "atr": atr,
                    "adx": adx,
                    "rsi": rsi,
                },
            )

        # No conditions met
        return TradeSignal(signal=Signal.HOLD)
