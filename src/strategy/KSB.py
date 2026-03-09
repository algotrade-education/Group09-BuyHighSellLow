"""
Keltner Squeeze Breakout (KSB) Strategy for VN30F1M.

Concept (TTM Squeeze variant):
    - Detects volatility compression: Bollinger Bands contracting inside Keltner Channels
    - When the squeeze releases, trades the directional breakout
    - Designed for medium-frequency intraday trading (30-60+ trades/month)

Squeeze detection:
    - squeeze_on:  bb_lower > kc_lower AND bb_upper < kc_upper
    - squeeze_off: NOT squeeze_on (BB expand back outside KC)

Entry logic:
    - Wait for squeeze to release (ON -> OFF transition)
    - Require squeeze to have lasted >= min_squeeze_bars
    - Signal window: entry allowed for signal_window bars after the release
      (not just the exact release bar)
    - Direction determined by:
        LONG:  mom > 0 AND close > kc_middle  (bullish momentum + above trend)
        SHORT: mom < 0 AND close < kc_middle  (bearish momentum + below trend)
    - Optional volume filter: volume > vol_mult * volume_ma

Exit logic:
    - Stop Loss: ATR-based (atr_sl_mult * ATR)
    - Take Profit: ATR-based (atr_tp_mult * ATR)
    - EOD close handled by the engine's session manager
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Dict, Optional

from src.strategy.base import Signal, Strategy, TradeSignal

logger = logging.getLogger(__name__)

MORNING_START = time(9, 0)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(14, 30)


@dataclass(frozen=True)
class ParsedBar:
    """Parsed bar fields used by KSB decision flow."""

    dt: datetime
    close: float
    high: float
    low: float
    volume: float


class KeltnerSqueezeBreakout(Strategy):
    """
    Keltner Squeeze Breakout strategy for VN30F1M.

    Detects volatility compression (BB inside KC) then trades the
    directional breakout once the squeeze releases.
    """

    REQUIRED_FIELDS = ["close", "high", "low", "open", "volume"]

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        mom_period: int = 12,
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 2.5,
        min_squeeze_bars: int = 3,
        signal_window: int = 3,
        cooldown_bars: int = 2,
        long_only: bool = False,
        use_volume_filter: bool = False,
        vol_mult: float = 1.5,
        vol_ma_period: int = 20,
        **kwargs,
    ):
        """
        Args:
            bb_period: Bollinger Band SMA period
            bb_std: Bollinger Band standard-deviation multiplier
            kc_period: Keltner Channel EMA period
            kc_mult: Keltner Channel ATR multiplier
            mom_period: Momentum oscillator lookback
            atr_period: ATR period for SL/TP and Keltner width
            atr_sl_mult: ATR multiplier for stop loss distance
            atr_tp_mult: ATR multiplier for take profit distance
            min_squeeze_bars: Minimum bars the squeeze must last before valid release
            signal_window: Number of bars after release where entry is still valid
            cooldown_bars: Bars to wait after closing a position
            long_only: If True, skip short entries
            use_volume_filter: Require volume > vol_mult * volume_ma
            vol_mult: Volume multiplier threshold (when filter enabled)
            vol_ma_period: Volume moving average period
        """
        super().__init__(name="KeltnerSqueezeBreakout")

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.mom_period = mom_period
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.min_squeeze_bars = max(1, int(min_squeeze_bars))
        self.signal_window = max(1, int(signal_window))
        self.cooldown_bars = max(0, int(cooldown_bars))
        self.long_only = long_only
        self.use_volume_filter = use_volume_filter
        self.vol_mult = vol_mult
        self.vol_ma_period = vol_ma_period

        # Session tracking
        self._current_date: Optional[date] = None
        self._current_session: Optional[str] = None

        # Squeeze state
        self._squeeze_on: bool = False
        self._squeeze_bar_count: int = 0

        # Signal window: bars remaining after a valid squeeze release
        self._window_remaining: int = 0

        # Cooldown state
        self._bars_since_exit: int = 9999
        self._was_flat: bool = True

        self._params = {
            "bb_period": bb_period,
            "bb_std": bb_std,
            "kc_period": kc_period,
            "kc_mult": kc_mult,
            "mom_period": mom_period,
            "atr_period": atr_period,
            "atr_sl_mult": atr_sl_mult,
            "atr_tp_mult": atr_tp_mult,
            "min_squeeze_bars": self.min_squeeze_bars,
            "signal_window": self.signal_window,
            "cooldown_bars": self.cooldown_bars,
            "long_only": long_only,
            "use_volume_filter": use_volume_filter,
            "vol_mult": vol_mult,
            "vol_ma_period": vol_ma_period,
        }

        logger.info("Strategy params: %s", self._params)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _get_session(self, dt: datetime) -> Optional[str]:
        t = dt.time()
        if MORNING_START <= t < MORNING_END:
            return "morning"
        if AFTERNOON_START <= t < AFTERNOON_END:
            return "afternoon"
        return None

    def _reset_squeeze_state(self) -> None:
        self._squeeze_on = False
        self._squeeze_bar_count = 0
        self._window_remaining = 0

    def _update_session_state(self, dt: datetime, session: str) -> None:
        current_date = dt.date()
        if current_date != self._current_date:
            self._current_date = current_date
            self._current_session = session
            self._reset_squeeze_state()
            self._bars_since_exit = 9999
            self._was_flat = True
        elif session != self._current_session:
            self._current_session = session
            self._reset_squeeze_state()

    # ------------------------------------------------------------------
    # Bar parsing
    # ------------------------------------------------------------------

    def _parse_bar(self, bar: Dict[str, Any]) -> Optional[ParsedBar]:
        dt = bar.get("datetime")
        if dt is None:
            return None
        if not isinstance(dt, datetime):
            try:
                dt = datetime.fromisoformat(str(dt))
            except (TypeError, ValueError):
                return None
        try:
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            volume = float(bar["volume"])
        except (KeyError, TypeError, ValueError):
            return None
        return ParsedBar(dt=dt, close=close, high=high, low=low, volume=volume)

    # ------------------------------------------------------------------
    # Signal builders
    # ------------------------------------------------------------------

    def _build_long_signal(
        self, close: float, atr: float, mom: float, session: str,
    ) -> TradeSignal:
        stop_loss = close - (self.atr_sl_mult * atr)
        take_profit = close + (self.atr_tp_mult * atr)

        if take_profit <= close or stop_loss >= close:
            return TradeSignal(signal=Signal.HOLD, reason="Invalid TP/SL (skip)")

        return TradeSignal(
            signal=Signal.LONG,
            entry_price=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"KSB Long (squeeze release, mom={mom:.2f})",
            metadata={"session": session, "mom": mom, "atr": atr},
        )

    def _build_short_signal(
        self, close: float, atr: float, mom: float, session: str,
    ) -> TradeSignal:
        stop_loss = close + (self.atr_sl_mult * atr)
        take_profit = close - (self.atr_tp_mult * atr)

        if take_profit >= close or stop_loss <= close:
            return TradeSignal(signal=Signal.HOLD, reason="Invalid TP/SL (skip)")

        return TradeSignal(
            signal=Signal.SHORT,
            entry_price=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"KSB Short (squeeze release, mom={mom:.2f})",
            metadata={"session": session, "mom": mom, "atr": atr},
        )

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        bar: Dict[str, Any],
        current_position: Optional[Any] = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        """
        Generate a trading signal based on Keltner Squeeze Breakout logic.

        Args:
            bar: Dict with OHLCV + indicator values for the current bar.
            current_position: Current position object from TradeManager.
        """
        # 1. Validation
        if not self.validate_bar(bar, self.REQUIRED_FIELDS):
            return TradeSignal(signal=Signal.HOLD)

        parsed = self._parse_bar(bar)
        if parsed is None:
            return TradeSignal(signal=Signal.HOLD, reason="Invalid bar data")

        session = self._get_session(parsed.dt)
        if session is None:
            return TradeSignal(signal=Signal.HOLD, reason="Outside trading session")

        self._update_session_state(parsed.dt, session)

        # 2. Cooldown tracking
        is_flat = current_position is None or current_position.is_flat

        if not self._was_flat and is_flat:
            # We just exited a trade. Wait, if it's during warmup, 
            # we don't track live exits, but if we did, we'd start cooldown.
            self._bars_since_exit = 0
        self._was_flat = is_flat

        if not is_flat:
            return TradeSignal(signal=Signal.HOLD)

        if self._bars_since_exit < self.cooldown_bars:
            if not is_warmup:
                self._bars_since_exit += 1
            return TradeSignal(
                signal=Signal.HOLD,
                reason=f"Cooldown ({self._bars_since_exit}/{self.cooldown_bars})",
            )
        if self._bars_since_exit < 9999 and not is_warmup:
            self._bars_since_exit += 1

        # 3. Read indicator values
        bb_upper = bar.get("bb_upper", 0.0)
        bb_lower = bar.get("bb_lower", 0.0)
        kc_upper = bar.get("kc_upper", 0.0)
        kc_lower = bar.get("kc_lower", 0.0)
        kc_middle = bar.get("kc_middle", 0.0)

        mom_col = f"mom_{self.mom_period}"
        mom = bar.get(mom_col, 0.0)

        atr_col = f"atr_{self.atr_period}"
        atr = bar.get(atr_col, 0.0)

        vol_ma_col = f"volume_ma_{self.vol_ma_period}"
        vol_ma = bar.get(vol_ma_col, 0.0)

        if atr <= 0 or kc_upper <= 0:
            return TradeSignal(signal=Signal.HOLD, reason="Missing indicators")

        # 4. Squeeze state machine
        currently_squeezed = (bb_lower > kc_lower) and (bb_upper < kc_upper)

        if currently_squeezed:
            if not self._squeeze_on:
                self._squeeze_on = True
                self._squeeze_bar_count = 1
            else:
                self._squeeze_bar_count += 1
            # New squeeze cancels any leftover signal window
            self._window_remaining = 0
            return TradeSignal(signal=Signal.HOLD, reason="Squeeze active")

        # Not squeezed - check for fresh release or active signal window
        if self._squeeze_on:
            # Squeeze just released this bar
            duration = self._squeeze_bar_count
            self._squeeze_on = False
            self._squeeze_bar_count = 0

            if duration >= self.min_squeeze_bars:
                self._window_remaining = self.signal_window
            # else: squeeze too short, no window opened

        # Tick down the window (even if we just opened it above)
        if self._window_remaining <= 0:
            return TradeSignal(signal=Signal.HOLD, reason="No active signal window")

        self._window_remaining -= 1

        # 5. Optional volume filter
        if self.use_volume_filter and vol_ma > 0:
            if parsed.volume <= self.vol_mult * vol_ma:
                return TradeSignal(signal=Signal.HOLD, reason="Volume below threshold")

        # 6. Direction: momentum sign + price vs KC middle
        close = parsed.close

        if mom > 0 and close > kc_middle:
            return self._build_long_signal(close, atr, mom, session)

        if not self.long_only and mom < 0 and close < kc_middle:
            return self._build_short_signal(close, atr, mom, session)

        return TradeSignal(signal=Signal.HOLD, reason="No directional confirmation")
