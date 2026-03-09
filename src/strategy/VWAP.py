"""
VWAP Band Reversion Strategy for VN30F1M.

Concept:
    - Uses session-resetting VWAP as the institutional mean-price anchor
    - When price deviates beyond N standard deviations from VWAP, fade back
    - Take profit when price returns to VWAP (the mean)
    - Designed for medium-frequency intraday mean reversion (40-80+ trades/month)

Entry logic:
    - LONG:  close < vwap - entry_band * vwap_std  (oversold relative to VWAP)
    - SHORT: close > vwap + entry_band * vwap_std  (overbought relative to VWAP)
    - Optional slope filter: only trade in VWAP slope direction
    - Optional volume filter: volume > vol_mult * volume_ma

Exit logic:
    - Take Profit: VWAP (the mean) - the natural reversion target
    - Stop Loss: ATR-based (atr_sl_mult * ATR)
    - Minimum TP distance guard: skip if VWAP is too close (< min_tp_atr * ATR)
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
    """Parsed bar fields used by VWAP decision flow."""

    dt: datetime
    close: float
    high: float
    low: float
    volume: float


class VWAPBandReversion(Strategy):
    """
    VWAP Band Reversion strategy for VN30F1M.

    Fades price extremes back to the session-resetting VWAP (Institutional mean).
    Entry is triggered when price deviates significantly (Standard Deviation bands) 
    from the VWAP, targeting the VWAP itself as the exit point.

    Key stages:
    1. Session Warmup: Prevents entry for the first N bars while VWAP stabilizes.
    2. Trend Detection: Optional slope filter to avoid fading strong trends.
    3. Mean Reversion: Enters when price is overextended.
    """

    REQUIRED_FIELDS = ["close", "high", "low", "open", "volume"]

    def __init__(
        self,
        entry_band: float = 2.0,
        atr_period: int = 14,
        atr_sl_mult: float = 1.5,
        min_tp_atr: float = 0.5,
        cooldown_bars: int = 2,
        session_warmup: int = 10,
        long_only: bool = False,
        use_slope_filter: bool = False,
        slope_period: int = 5,
        use_volume_filter: bool = False,
        vol_mult: float = 1.5,
        vol_ma_period: int = 20,
        **kwargs,
    ):
        """
        Args:
            entry_band: Std-deviation multiplier for entry bands (e.g. 2.0 = 2σ)
            atr_period: ATR period for stop-loss calculation
            atr_sl_mult: ATR multiplier for stop-loss distance
            min_tp_atr: Minimum TP distance in ATR units; skip if VWAP is closer
            cooldown_bars: Bars to wait after closing a position
            session_warmup: Bars from session start before signals are allowed
            long_only: If True, skip short entries
            use_slope_filter: Require VWAP slope alignment for entry direction
            slope_period: Lookback bars for VWAP slope calculation
            use_volume_filter: Require volume > vol_mult * volume_ma
            vol_mult: Volume multiplier threshold (when filter enabled)
            vol_ma_period: Volume moving average period
        """
        super().__init__(name="VWAPBandReversion")

        self.entry_band = entry_band
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.min_tp_atr = min_tp_atr
        self.cooldown_bars = max(0, int(cooldown_bars))
        self.session_warmup = max(0, int(session_warmup))
        self.long_only = long_only
        self.use_slope_filter = use_slope_filter
        self.slope_period = max(1, int(slope_period))
        self.use_volume_filter = use_volume_filter
        self.vol_mult = vol_mult
        self.vol_ma_period = vol_ma_period

        # Session tracking
        self._current_date: Optional[date] = None
        self._current_session: Optional[str] = None
        self._session_bar_count: int = 0

        # VWAP slope history (ring buffer of recent VWAP values)
        self._vwap_history: list[float] = []

        # Cooldown state
        self._bars_since_exit: int = 9999
        self._was_flat: bool = True

        self._params = {
            "entry_band": entry_band,
            "atr_period": atr_period,
            "atr_sl_mult": atr_sl_mult,
            "min_tp_atr": min_tp_atr,
            "cooldown_bars": self.cooldown_bars,
            "session_warmup": self.session_warmup,
            "long_only": long_only,
            "use_slope_filter": use_slope_filter,
            "slope_period": self.slope_period,
            "use_volume_filter": use_volume_filter,
            "vol_mult": vol_mult,
            "vol_ma_period": vol_ma_period,
        }

        logger.info("Strategy params: %s", self._params)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _get_session(self, dt: datetime) -> Optional[str]:
        """
        Determine which trading session the timestamp belongs to.

        Args:
            dt: Current bar timestamp.

        Returns:
            "morning", "afternoon", or None.
        """
        t = dt.time()
        if MORNING_START <= t < MORNING_END:
            return "morning"
        if AFTERNOON_START <= t < AFTERNOON_END:
            return "afternoon"
        return None

    def _update_session_state(self, dt: datetime, session: str) -> None:
        current_date = dt.date()
        if current_date != self._current_date:
            self._current_date = current_date
            self._current_session = session
            self._session_bar_count = 0
            self._vwap_history = []
            self._bars_since_exit = 9999
            self._was_flat = True
        elif session != self._current_session:
            self._current_session = session
            self._session_bar_count = 0
            self._vwap_history = []

        # We will increment the session bar count in generate_signal
        # so we can check if it's a warmup bar or a live bar

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
        self, close: float, vwap: float, atr: float, session: str,
    ) -> TradeSignal:
        """
        Build a LONG mean reversion signal targeting VWAP.

        Args:
            close: Current bar close price.
            vwap: Current VWAP level (Exit target).
            atr: Current ATR for risk/distance filters.
            session: Current session name.
        """
        stop_loss = close - (self.atr_sl_mult * atr)
        take_profit = vwap

        if take_profit <= close or stop_loss >= close:
            return TradeSignal(signal=Signal.HOLD, reason="Invalid TP/SL (skip)")

        tp_distance = take_profit - close
        if tp_distance < self.min_tp_atr * atr:
            return TradeSignal(signal=Signal.HOLD, reason="TP too close to VWAP")

        return TradeSignal(
            signal=Signal.LONG,
            entry_price=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"VWAP Long (close below lower band)",
            metadata={"session": session, "vwap": vwap, "atr": atr},
        )


    def _build_short_signal(
        self, close: float, vwap: float, atr: float, session: str,
    ) -> TradeSignal:
        """
        Build a SHORT mean reversion signal targeting VWAP.

        Args:
            close: Current bar close price.
            vwap: Current VWAP level (Exit target).
            atr: Current ATR for risk/distance filters.
            session: Current session name.
        """
        stop_loss = close + (self.atr_sl_mult * atr)
        take_profit = vwap

        if take_profit >= close or stop_loss <= close:
            return TradeSignal(signal=Signal.HOLD, reason="Invalid TP/SL (skip)")

        tp_distance = close - take_profit
        if tp_distance < self.min_tp_atr * atr:
            return TradeSignal(signal=Signal.HOLD, reason="TP too close to VWAP")

        return TradeSignal(
            signal=Signal.SHORT,
            entry_price=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"VWAP Short (close above upper band)",
            metadata={"session": session, "vwap": vwap, "atr": atr},
        )

    # ------------------------------------------------------------------
    # VWAP slope
    # ------------------------------------------------------------------

    def _vwap_slope(self) -> float:
        """Return VWAP slope over the last slope_period bars (0 if not enough data)."""
        if len(self._vwap_history) < self.slope_period + 1:
            return 0.0
        return self._vwap_history[-1] - self._vwap_history[-(self.slope_period + 1)]

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
        Generate a trading signal based on VWAP Band Reversion logic.

        Decision flow:
        - Check session boundaries and warmup status.
        - Calculate VWAP slope for trend filtering.
        - Monitor for price breakouts beyond the Std-Dev bands.
        - Validate distance to mean (VWAP) for favorable reward/risk.

        Args:
            bar: Dict containing at least: ['close', 'high', 'low', 'open', 'volume']
                 + 'vwap', 'vwap_std', 'atr_{period}', 'volume_ma_{period}'.
            current_position: Current position object from TradeManager.
            is_warmup: Prevents state updates (session bar count) during warmup.
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
        
        if not is_warmup:
            self._session_bar_count += 1

        # 2. Read indicator values (update VWAP history before any early return)
        vwap = bar.get("vwap", 0.0)
        vwap_std = bar.get("vwap_std", 0.0)

        if vwap and vwap > 0:
            self._vwap_history.append(vwap)
            if len(self._vwap_history) > self.slope_period + 5:
                self._vwap_history = self._vwap_history[-(self.slope_period + 5):]

        # 3. Cooldown tracking
        is_flat = current_position is None or current_position.is_flat

        if not self._was_flat and is_flat:
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

        # 4. Session warmup guard
        if self._session_bar_count < self.session_warmup:
            return TradeSignal(signal=Signal.HOLD, reason="Session warmup")

        atr_col = f"atr_{self.atr_period}"
        atr = bar.get(atr_col, 0.0)

        vol_ma_col = f"volume_ma_{self.vol_ma_period}"
        vol_ma = bar.get(vol_ma_col, 0.0)

        if atr <= 0 or vwap <= 0 or vwap_std <= 0:
            return TradeSignal(signal=Signal.HOLD, reason="Missing indicators")

        # 5. Compute entry bands
        upper_entry = vwap + self.entry_band * vwap_std
        lower_entry = vwap - self.entry_band * vwap_std
        close = parsed.close

        # 6. Optional volume filter
        if self.use_volume_filter and vol_ma > 0:
            if parsed.volume <= self.vol_mult * vol_ma:
                return TradeSignal(signal=Signal.HOLD, reason="Volume below threshold")

        # 7. Optional slope filter
        slope = self._vwap_slope()

        # 8. Entry logic - fade extremes back to VWAP
        if close < lower_entry:
            if self.use_slope_filter and slope < 0:
                return TradeSignal(
                    signal=Signal.HOLD, reason="VWAP slope bearish (skip long)"
                )
            return self._build_long_signal(close, vwap, atr, session)

        if not self.long_only and close > upper_entry:
            if self.use_slope_filter and slope > 0:
                return TradeSignal(
                    signal=Signal.HOLD, reason="VWAP slope bullish (skip short)"
                )
            return self._build_short_signal(close, vwap, atr, session)

        return TradeSignal(signal=Signal.HOLD, reason="Price within VWAP bands")
