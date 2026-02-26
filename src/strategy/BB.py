"""
Bollinger Band Mean Reversion Strategy v2 with ADX Filter.

Key improvements over v1:
    - Entry uses BB %B instead of exact band touch (more flexible)
    - BB Bandwidth filter to avoid squeeze breakouts
    - Take Profit targets BB Middle (true mean reversion) with ATR fallback
    - Post-trade cooldown to avoid overtrading
    - Optional volume confirmation

Entry logic:
    - Long: %B < bb_pctb_entry AND ADX < threshold AND RSI < oversold
    - Short: %B > (1 - bb_pctb_entry) AND ADX < threshold AND RSI > overbought
    - Bandwidth must be above bb_bandwidth_min (skip squeeze)

Exit logic:
    - Take Profit: BB Middle (mean) or ATR-based fallback if middle too close
    - Stop Loss: ATR-based (atr_sl_multiplier * ATR beyond entry)
    - EOD close handled by the engine's session manager

All parameters are constructor kwargs for easy optimization via Optuna.
"""

import logging
from typing import Any, Dict, Optional

from src.strategy.base import Signal, Strategy, TradeSignal

logger = logging.getLogger(__name__)


class BollingerMeanReversion(Strategy):
    """
    Bollinger Band Mean Reversion strategy v2.

    Enters when price is near outer bands (%B-based) in a range-bound market
    (low ADX) and RSI confirms oversold/overbought.
    Targets the middle band (SMA) as the mean reversion exit.
    Uses ATR-based stop loss for adaptive risk management.
    Includes bandwidth filter and post-trade cooldown.
    """

    # Required indicator columns for bar validation
    REQUIRED_FIELDS = [
        "close",
        "bb_upper",
        "bb_lower",
        "bb_middle",
        "bb_pctb",
        "bb_bandwidth",
    ]

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_pctb_entry: float = 0.2,
        bb_bandwidth_min: float = 0.005,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 2.0,
        cooldown_bars: int = 3,
        use_volume_filter: bool = False,
        **kwargs,
    ):
        """
        Initialize the Bollinger Mean Reversion strategy v2.

        Args:
            bb_period: Bollinger Bands SMA period
            bb_std: Bollinger Bands standard deviation multiplier
            bb_pctb_entry: %B threshold for entry (< for long, > 1-val for short)
            bb_bandwidth_min: Min bandwidth to allow entries (skip squeeze)
            adx_period: ADX indicator period
            adx_threshold: Max ADX value to allow entries (lower = more range-bound)
            rsi_period: RSI indicator period
            rsi_oversold: RSI level below which the market is oversold (long entry)
            rsi_overbought: RSI level above which the market is overbought (short entry)
            atr_period: ATR indicator period
            atr_sl_multiplier: ATR multiplier for stop loss distance
            atr_tp_multiplier: ATR multiplier for take profit (fallback if BB middle too close)
            cooldown_bars: Bars to skip after a trade closes before new entries
            use_volume_filter: If True, require volume > volume_ma for entry
        """
        super().__init__(name="BollingerMeanReversion")

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_pctb_entry = bb_pctb_entry
        self.bb_bandwidth_min = bb_bandwidth_min
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.cooldown_bars = cooldown_bars
        self.use_volume_filter = use_volume_filter

        # Cooldown state tracking
        self._bars_since_flat = 0
        self._was_in_position = False

        # Store all params for serialization / optimization
        self._params = {
            "bb_period": bb_period,
            "bb_std": bb_std,
            "bb_pctb_entry": bb_pctb_entry,
            "bb_bandwidth_min": bb_bandwidth_min,
            "adx_period": adx_period,
            "adx_threshold": adx_threshold,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "atr_period": atr_period,
            "atr_sl_multiplier": atr_sl_multiplier,
            "atr_tp_multiplier": atr_tp_multiplier,
            "cooldown_bars": cooldown_bars,
            "use_volume_filter": use_volume_filter,
        }

        logger.info("Strategy params: %s", self._params)

    def generate_signal(
        self,
        bar: Dict[str, Any],
        current_position: Optional[Any] = None,
    ) -> TradeSignal:
        """
        Generate a trading signal based on BB %B, ADX, RSI, and bandwidth.

        Args:
            bar: Dict containing OHLC + indicator values for the current bar.
                 Expected keys: close, bb_upper, bb_lower, bb_middle, bb_pctb,
                 bb_bandwidth, adx_{period}, rsi_{period}, atr_{period}
            current_position: Current position object from TradeManager

        Returns:
            TradeSignal with entry/exit details
        """
        # Validate required fields
        if not self.validate_bar(bar, self.REQUIRED_FIELDS):
            return TradeSignal(signal=Signal.HOLD)

        close = bar["close"]
        bb_middle = bar["bb_middle"]
        bb_pctb = bar["bb_pctb"]
        bb_bandwidth = bar["bb_bandwidth"]

        # Get indicator values with safe defaults
        adx_col = f"adx_{self.adx_period}"
        rsi_col = f"rsi_{self.rsi_period}"
        atr_col = f"atr_{self.atr_period}"

        adx = bar.get(adx_col, 50.0)  # Default high ADX = no trade
        rsi = bar.get(rsi_col, 50.0)  # Default neutral RSI
        atr = bar.get(atr_col, 0.0)

        # --- POSITION STATE TRACKING (for cooldown) ---
        is_flat = current_position is None or current_position.is_flat

        if is_flat:
            if self._was_in_position:
                # Just closed a position — start cooldown
                self._bars_since_flat = 0
                self._was_in_position = False
            self._bars_since_flat += 1
        else:
            # Currently in a position
            self._was_in_position = True
            return TradeSignal(signal=Signal.HOLD)

        # --- COOLDOWN CHECK ---
        if self.cooldown_bars > 0 and self._bars_since_flat <= self.cooldown_bars:
            return TradeSignal(
                signal=Signal.HOLD,
                reason=f"Cooldown ({self._bars_since_flat}/{self.cooldown_bars})",
            )

        # --- FILTER CONDITIONS ---
        # ADX filter: only trade in range-bound markets
        is_range_bound = adx < self.adx_threshold
        if not is_range_bound:
            return TradeSignal(signal=Signal.HOLD, reason="ADX too high (trending)")

        # Bandwidth filter: skip when BB is too narrow (squeeze → potential breakout)
        if bb_bandwidth < self.bb_bandwidth_min:
            return TradeSignal(signal=Signal.HOLD, reason="BB squeeze (low bandwidth)")

        # Guard: ATR must be positive for stop loss calculation
        if atr <= 0:
            return TradeSignal(signal=Signal.HOLD, reason="ATR is zero or negative")

        # Guard: %B and bandwidth must be valid numbers
        if bb_pctb != bb_pctb or bb_bandwidth != bb_bandwidth:  # NaN check
            return TradeSignal(signal=Signal.HOLD, reason="Invalid %B or bandwidth")

        # Volume filter (optional)
        if self.use_volume_filter:
            volume = bar.get("volume", 0)
            volume_ma = bar.get(f"volume_ma_20", 0)
            if volume_ma > 0 and volume < volume_ma:
                return TradeSignal(signal=Signal.HOLD, reason="Volume below average")

        sl_distance = self.atr_sl_multiplier * atr
        tp_atr_distance = self.atr_tp_multiplier * atr

        # --- LONG ENTRY ---
        if bb_pctb < self.bb_pctb_entry and rsi < self.rsi_oversold:
            stop_loss = close - sl_distance

            # TP = BB Middle (mean reversion target)
            # Fallback to ATR-based TP if BB Middle is too close to entry
            tp_to_middle = bb_middle - close
            min_tp_distance = 0.3 * atr  # At least 0.3 ATR profit

            if tp_to_middle > min_tp_distance:
                take_profit = bb_middle
            else:
                take_profit = close + tp_atr_distance

            # Sanity: TP must be above entry
            if take_profit <= close:
                return TradeSignal(signal=Signal.HOLD, reason="TP <= entry (skip)")

            return TradeSignal(
                signal=Signal.LONG,
                entry_price=0.0,  # 0 = market order (fill at next open)
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"BB Long (%B={bb_pctb:.2f}, RSI={rsi:.1f}, ADX={adx:.1f})",
                metadata={
                    "bb_pctb": bb_pctb,
                    "bb_bandwidth": bb_bandwidth,
                    "bb_middle": bb_middle,
                    "atr": atr,
                    "adx": adx,
                    "rsi": rsi,
                    "tp_type": "bb_middle" if tp_to_middle > min_tp_distance else "atr",
                },
            )

        # --- SHORT ENTRY ---
        if bb_pctb > (1.0 - self.bb_pctb_entry) and rsi > self.rsi_overbought:
            stop_loss = close + sl_distance

            # TP = BB Middle (mean reversion target)
            tp_to_middle = close - bb_middle
            min_tp_distance = 0.3 * atr

            if tp_to_middle > min_tp_distance:
                take_profit = bb_middle
            else:
                take_profit = close - tp_atr_distance

            # Sanity: TP must be below entry
            if take_profit >= close:
                return TradeSignal(signal=Signal.HOLD, reason="TP >= entry (skip)")

            return TradeSignal(
                signal=Signal.SHORT,
                entry_price=0.0,  # Market order
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=f"BB Short (%B={bb_pctb:.2f}, RSI={rsi:.1f}, ADX={adx:.1f})",
                metadata={
                    "bb_pctb": bb_pctb,
                    "bb_bandwidth": bb_bandwidth,
                    "bb_middle": bb_middle,
                    "atr": atr,
                    "adx": adx,
                    "rsi": rsi,
                    "tp_type": "bb_middle" if tp_to_middle > min_tp_distance else "atr",
                },
            )

        # No conditions met
        return TradeSignal(signal=Signal.HOLD)
