"""
Opening Range Breakout (ORB) Strategy for VN30F1M.

Concept:
    - Define an "opening range" from the first N minutes of each trading session
    - After the range forms, trade breakouts above the high or below the low
    - VN30 has 2 sessions: morning (09:00-11:30) and afternoon (13:00-14:30)
    - Each session gets its own independent opening range

Entry logic:
    - LONG:  close > range_high + (breakout_buffer * ATR)
    - SHORT: close < range_low  - (breakout_buffer * ATR) (skipped if long_only)
    - Max N trades per session (configurable)

Exit logic:
    - Stop Loss: opposite side of the opening range, or ATR-based fallback
    - Take Profit: ATR-based (atr_tp_multiplier * ATR)
    - EOD close handled by the engine's session manager

Filters:
    - Range size filter: skip if range < min_range_atr or > max_range_atr (in ATR units)
    - Volume filter (optional): require volume > volume_ma for breakout confirmation
    - ADX filter (optional): require ADX > adx_min for trend confirmation
    - Long-only mode (optional): skip all short breakouts

All parameters are constructor kwargs for easy optimization via Optuna.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Dict, Optional

from src.strategy.base import Signal, Strategy, TradeSignal

logger = logging.getLogger(__name__)


# VN30 session boundaries
MORNING_START = time(9, 0)
MORNING_END = time(11, 30)
AFTERNOON_START = time(13, 0)
AFTERNOON_END = time(14, 30)


@dataclass(frozen=True)
class ParsedBar:
    """Parsed bar fields used by ORB decision flow."""

    dt: datetime
    close: float
    high: float
    low: float


class OpeningRangeBreakout(Strategy):
    """
    Opening Range Breakout strategy for VN30F1M.

    Tracks the high/low of the first N minutes of each trading session,
    then trades breakouts beyond the range. Designed for intraday futures
    with 2 sessions per day.
    """

    REQUIRED_FIELDS = ["close", "high", "low", "open"]

    def __init__(
        self,
        orb_minutes: int = 15,
        atr_period: int = 14,
        atr_tp_multiplier: float = 2.0,
        atr_sl_multiplier: float = 1.5,
        breakout_buffer: float = 0.1,
        use_range_sl: bool = True,
        min_range_atr: float = 0.5,
        max_range_atr: float = 3.0,
        long_only: bool = False,
        use_volume_filter: bool = False,
        use_adx_filter: bool = False,
        adx_min: float = 20.0,
        max_trades_per_session: int = 1,
        **kwargs,
    ):
        """
        Initialize the Opening Range Breakout strategy.

        Args:
            orb_minutes: Minutes for opening range formation (e.g., 15, 30)
            atr_period: ATR indicator period
            atr_tp_multiplier: ATR multiplier for take profit distance
            atr_sl_multiplier: ATR multiplier for stop loss (fallback if use_range_sl=False)
            breakout_buffer: Extra buffer beyond range in ATR units for confirmation
            use_range_sl: If True, SL = opposite range boundary; else ATR-based
            min_range_atr: Skip if range < this many ATRs (too narrow → noise)
            max_range_atr: Skip if range > this many ATRs (too wide → risk)
            long_only: If True, skip all short breakouts
            use_volume_filter: If True, require volume > volume_ma for entry
            use_adx_filter: If True, require ADX > adx_min for entry
            adx_min: Minimum ADX value when use_adx_filter is enabled
            max_trades_per_session: Maximum number of entries allowed per session
        """
        super().__init__(name="OpeningRangeBreakout")

        self.orb_minutes = orb_minutes
        self.atr_period = atr_period
        self.atr_tp_multiplier = atr_tp_multiplier
        self.atr_sl_multiplier = atr_sl_multiplier
        self.breakout_buffer = breakout_buffer
        self.use_range_sl = use_range_sl
        self.min_range_atr = min_range_atr
        self.max_range_atr = max_range_atr
        self.long_only = long_only
        self.use_volume_filter = use_volume_filter
        self.use_adx_filter = use_adx_filter
        self.adx_min = adx_min
        self.max_trades_per_session = max(1, int(max_trades_per_session))

        # Session state
        self._current_date: Optional[date] = None
        self._current_session: Optional[str] = None  # "morning" or "afternoon"
        self._range_high: float = 0.0
        self._range_low: float = float("inf")
        self._range_formed: bool = False
        self._trades_this_session: int = 0
        self._range_start_time: Optional[datetime] = None

        # Store params for serialization / optimization
        self._params = {
            "orb_minutes": orb_minutes,
            "atr_period": atr_period,
            "atr_tp_multiplier": atr_tp_multiplier,
            "atr_sl_multiplier": atr_sl_multiplier,
            "breakout_buffer": breakout_buffer,
            "use_range_sl": use_range_sl,
            "min_range_atr": min_range_atr,
            "max_range_atr": max_range_atr,
            "long_only": long_only,
            "use_volume_filter": use_volume_filter,
            "use_adx_filter": use_adx_filter,
            "adx_min": adx_min,
            "max_trades_per_session": self.max_trades_per_session,
        }

        logger.info("Strategy params: %s", self._params)

    def _get_session(self, dt: datetime) -> Optional[str]:
        """Determine which trading session the timestamp belongs to."""
        t = dt.time()
        if MORNING_START <= t < MORNING_END:
            return "morning"
        elif AFTERNOON_START <= t < AFTERNOON_END:
            return "afternoon"
        return None

    def _get_session_start(self, session: str) -> time:
        """Get the start time for a session."""
        if session == "morning":
            return MORNING_START
        return AFTERNOON_START

    def _is_in_formation_window(self, dt: datetime, session: str) -> bool:
        """Check if timestamp is within the opening range formation window."""
        session_start = self._get_session_start(session)
        # Convert to minutes since session start for comparison
        session_start_minutes = session_start.hour * 60 + session_start.minute
        current_minutes = dt.hour * 60 + dt.minute

        elapsed = current_minutes - session_start_minutes
        return elapsed < self.orb_minutes

    def _reset_session_state(self) -> None:
        """Reset state for a new session."""
        self._range_high = 0.0
        self._range_low = float("inf")
        self._range_formed = False
        self._trades_this_session = 0
        self._range_start_time = None

    def _parse_bar(self, bar: Dict[str, Any]) -> Optional[ParsedBar]:
        """Extract and coerce datetime/close/high/low from bar."""
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
        except (KeyError, TypeError, ValueError):
            return None

        return ParsedBar(dt=dt, close=close, high=high, low=low)

    def _update_session_state(self, dt: datetime, session: str) -> None:
        """Reset and initialize state when session/day changes."""
        current_date = dt.date()
        if current_date != self._current_date or session != self._current_session:
            self._current_date = current_date
            self._current_session = session
            self._reset_session_state()
            self._range_start_time = dt

    def _update_formation(self, high: float, low: float) -> None:
        """Expand opening range boundaries during formation phase."""
        if high > self._range_high:
            self._range_high = high
        if low < self._range_low:
            self._range_low = low

    def _check_range_filters(self, atr: float) -> Optional[TradeSignal]:
        """Validate range quality constraints for breakout eligibility."""
        range_size = self._range_high - self._range_low
        if range_size <= 0:
            return TradeSignal(signal=Signal.HOLD, reason="Invalid range (size <= 0)")

        range_in_atr = range_size / atr

        if range_in_atr < self.min_range_atr:
            return TradeSignal(
                signal=Signal.HOLD,
                reason=f"Range too narrow: Size is {range_in_atr:.1f}x ATR (Min {self.min_range_atr}x)",
            )

        if range_in_atr > self.max_range_atr:
            return TradeSignal(
                signal=Signal.HOLD,
                reason=f"Range too wide: Size is {range_in_atr:.1f}x ATR (Max {self.max_range_atr}x)",
            )

        return None

    def _check_optional_filters(self, bar: Dict[str, Any]) -> Optional[TradeSignal]:
        """Run optional volume/ADX filters before breakout checks."""
        if self.use_volume_filter:
            volume = bar.get("volume", 0)
            volume_ma = bar.get("volume_ma_20", 0)
            if volume_ma > 0 and volume < volume_ma:
                return TradeSignal(signal=Signal.HOLD, reason="Volume below average")

        if self.use_adx_filter:
            adx_col = f"adx_{self.atr_period}"
            adx = bar.get(adx_col, 0.0)
            if adx < self.adx_min:
                return TradeSignal(
                    signal=Signal.HOLD,
                    reason=f"ADX too low ({adx:.1f} < {self.adx_min})",
                )

        return None

    def _build_long_signal(
        self,
        close: float,
        atr: float,
        range_size: float,
        session: str,
        is_warmup: bool = False,
    ) -> TradeSignal:
        """Build validated long breakout signal."""
        if self.use_range_sl:
            stop_loss = self._range_low
        else:
            stop_loss = close - (self.atr_sl_multiplier * atr)

        take_profit = close + (self.atr_tp_multiplier * atr)

        if take_profit <= close:
            return TradeSignal(signal=Signal.HOLD, reason="TP <= entry (skip)")
        if stop_loss >= close:
            return TradeSignal(signal=Signal.HOLD, reason="SL >= entry (skip)")

        if not is_warmup:
            self._trades_this_session += 1
        
        range_in_atr = range_size / atr

        return TradeSignal(
            signal=Signal.LONG,
            entry_price=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"ORB Long breakout ({session}, range={range_size:.1f})",
            metadata={
                "session": session,
                "range_high": self._range_high,
                "range_low": self._range_low,
                "range_size": range_size,
                "range_in_atr": range_in_atr,
                "atr": atr,
                "sl_type": "range" if self.use_range_sl else "atr",
            },
        )

    def _build_short_signal(
        self,
        close: float,
        atr: float,
        range_size: float,
        session: str,
        is_warmup: bool = False,
    ) -> TradeSignal:
        """Build validated short breakout signal."""
        if self.use_range_sl:
            stop_loss = self._range_high
        else:
            stop_loss = close + (self.atr_sl_multiplier * atr)

        take_profit = close - (self.atr_tp_multiplier * atr)

        if take_profit >= close:
            return TradeSignal(signal=Signal.HOLD, reason="TP >= entry (skip)")
        if stop_loss <= close:
            return TradeSignal(signal=Signal.HOLD, reason="SL <= entry (skip)")

        if not is_warmup:
            self._trades_this_session += 1
            
        range_in_atr = range_size / atr

        return TradeSignal(
            signal=Signal.SHORT,
            entry_price=0.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"ORB Short breakout ({session}, range={range_size:.1f})",
            metadata={
                "session": session,
                "range_high": self._range_high,
                "range_low": self._range_low,
                "range_size": range_size,
                "range_in_atr": range_in_atr,
                "atr": atr,
                "sl_type": "range" if self.use_range_sl else "atr",
            },
        )

    def generate_signal(
        self,
        bar: Dict[str, Any],
        current_position: Optional[Any] = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        """
        Generate a trading signal based on Opening Range Breakout logic.

        Args:
            bar: Dict containing OHLC + indicator values for the current bar.
                 Expected keys: datetime, open, high, low, close, atr_{period}
            current_position: Current position object from TradeManager

        Returns:
            TradeSignal with entry/exit details
        """
        # Validate required fields
        if not self.validate_bar(bar, self.REQUIRED_FIELDS):
            return TradeSignal(signal=Signal.HOLD)

        if bar.get("datetime") is None:
            return TradeSignal(signal=Signal.HOLD, reason="No datetime in bar")

        parsed = self._parse_bar(bar)
        if parsed is None:
            return TradeSignal(signal=Signal.HOLD, reason="Invalid bar data")

        dt = parsed.dt
        close = parsed.close
        high = parsed.high
        low = parsed.low

        # Determine current session
        session = self._get_session(dt)
        if session is None:
            return TradeSignal(signal=Signal.HOLD, reason="Outside trading session")

        self._update_session_state(dt, session)

        # If already in a position, hold (exits via SL/TP/EOD)
        if current_position is not None and not current_position.is_flat:
            return TradeSignal(signal=Signal.HOLD)

        # Reached session trade limit
        if self._trades_this_session >= self.max_trades_per_session:
            return TradeSignal(
                signal=Signal.HOLD,
                reason=(
                    "Session trade limit reached "
                    f"({self._trades_this_session}/{self.max_trades_per_session})"
                ),
            )

        # --- FORMATION PHASE ---
        if self._is_in_formation_window(dt, session):
            self._update_formation(high, low)

            return TradeSignal(
                signal=Signal.HOLD,
                reason=f"Forming range ({session}): H={self._range_high:.1f} L={self._range_low:.1f}",
            )

        # --- RANGE JUST FORMED ---
        if not self._range_formed:
            self._range_formed = True
            logger.debug(
                "ORB range formed (%s): High=%.2f, Low=%.2f",
                session,
                self._range_high,
                self._range_low,
            )

        # --- BREAKOUT PHASE ---
        # Get ATR
        atr_col = f"atr_{self.atr_period}"
        atr = bar.get(atr_col, 0.0)

        if atr <= 0:
            return TradeSignal(signal=Signal.HOLD, reason="ATR is zero or negative")

        range_filter = self._check_range_filters(atr)
        if range_filter is not None:
            return range_filter

        range_size = self._range_high - self._range_low

        optional_filter = self._check_optional_filters(bar)
        if optional_filter is not None:
            return optional_filter

        # Breakout levels
        buffer = self.breakout_buffer * atr
        breakout_high = self._range_high + buffer
        breakout_low = self._range_low - buffer

        # --- LONG BREAKOUT ---
        if close > breakout_high:
            return self._build_long_signal(close, atr, range_size, session, is_warmup)

        # --- SHORT BREAKOUT ---
        if self.long_only:
            return TradeSignal(signal=Signal.HOLD)

        if close < breakout_low:
            return self._build_short_signal(close, atr, range_size, session, is_warmup)

        # No breakout yet
        return TradeSignal(signal=Signal.HOLD)
