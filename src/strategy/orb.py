"""
Opening Range Breakout (ORB) Strategy.

Concept:
    - Define opening range from the first N minutes of each trading session
    - Trade breakouts above range_high or below range_low (+ buffer)
    - (VN30) Has 2 sessions and dependent each other

Entry:
    LONG:  close > range_high + (breakout_buffer * ATR)
    SHORT: close < range_low  - (breakout_buffer * ATR)  [skip if long_only]

Exit (managed by engine):
    Stop Loss:   opposite side of range, or ATR-based fallback
    Take Profit: ATR-based (atr_tp_multiplier * ATR)
    EOD:         engine's session manager

Filters:
    Range size: skip if range < min_range_atr or > max_range_atr (in ATR units)
    Volume:     require volume > volume_filter_threshold * volume_ma  [optional]
    ADX:        require ADX > adx_min  [optional]
    Confirm:    require open AND close > breakout level  [optional]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from config.schemas.orb import ORBConfig, ORBStrategyConfig
from config.schemas.session import Session, SessionConfig
from src.data.indicators.registry import IndicatorRegistry, IndicatorSpec
from src.strategy.base import PositionSnapshot
from src.strategy.intraday_base import IntradayStrategy
from src.strategy.signal import Signal, TradeSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ParsedBar:
    """Parsed and validated bar fields used in ORB decision flow."""

    dt: datetime
    close: float
    high: float
    low: float


class ORBStrategy(IntradayStrategy):
    """
    Opening Range Breakout strategy.

    Usage:
        config = ORBConfig.from_json("config/strategy_params/orb_default.json")
        strategy = ORBStrategy(config)
        registry = ORBStrategy.build_registry(atr_period=14)
    """

    REQUIRED_FIELDS = ["datetime", "open", "high", "low", "close"]

    def __init__(
        self,
        config: ORBConfig,
        session: SessionConfig | None = None,
    ) -> None:
        """
        Args:
            config:  ORBConfig already validated by Pydantic.
            session: Injected SessionConfig - default VN30.
                     Override to use with a different market.
        """
        super().__init__(
            name="ORBStrategy",
            session=session,
            gap_threshold_minutes=60,
        )
        self._p: ORBStrategyConfig = config.strategy
        self._risk = config.risk

        # Cache indicator column names to avoid formatting per bar
        self._atr_col = f"atr_{self._p.atr_period}"
        self._adx_col = f"adx_{self._p.adx_period}"
        self._vol_ma_col = f"volume_ma_{self._p.volume_ma_period}"
        self._atr_ma_col = (
            f"atr_ma_{self._p.atr_lookback_period}" if self._p.use_adaptive_volatility else None
        )

        # --- Opening range state ---
        self._range_high: float = 0.0
        self._range_low: float = float("inf")
        self._range_formed: bool = False
        self._trades_this_session: int = 0

        logger.info(
            "ORBStrategy initialized: orb=%dmin, atr=%d, tp=%.1fx, sl=%.1fx",
            self._p.orb_minutes,
            self._p.atr_period,
            self._p.atr_tp_multiplier,
            self._p.atr_sl_multiplier,
        )

    # --- Registry ---

    @classmethod
    def build_registry(
        cls,
        atr_period: int = 14,
        adx_period: int = 14,
        volume_ma_period: int = 20,
        atr_lookback_period: int = 20,
        use_adaptive_volatility: bool = False,
        **_: Any,
    ) -> IndicatorRegistry:
        """
        Build IndicatorRegistry for ORB.
        Register only the indicators that are actually needed.
        """
        registry = (
            IndicatorRegistry()
            .register(IndicatorSpec("atr", {"period": atr_period}, f"atr_{atr_period}"))
            .register(IndicatorSpec("adx", {"period": adx_period}, f"adx_{adx_period}"))
            .register(
                IndicatorSpec(
                    "volume_ma", {"period": volume_ma_period}, f"volume_ma_{volume_ma_period}"
                )
            )
        )

        # Add ATR moving average for adaptive volatility
        if use_adaptive_volatility:
            registry.register(
                IndicatorSpec(
                    "sma",
                    {"period": atr_lookback_period, "source_col": f"atr_{atr_period}"},
                    f"atr_ma_{atr_lookback_period}",
                )
            )

        return registry

    # --- Core interface ---

    def generate_signal(
        self,
        bar: dict[str, Any],
        position: PositionSnapshot | None = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        """
        Generate trading signal based on ORB logic.

        Decision flow:
            1. Validate bar fields
            2. Determine session - HOLD if outside trading hours
            3. Reset state on new session/day/gap
            4. HOLD if a position is already open (exit via SL/TP/EOD)
            5. HOLD if trade limit is reached
            6. Formation phase: update range
            7. Breakout phase: check filters + generate signal
        """
        # --- 1. Validate ---
        if not self.validate_bar(bar, self.REQUIRED_FIELDS):
            return TradeSignal(Signal.HOLD, reason="Invalid bar fields")

        parsed = self._parse_bar(bar)
        if parsed is None:
            return TradeSignal(Signal.HOLD, reason="Failed to parse bar")

        # --- 2. Session check ---
        session = self._get_session(parsed.dt)
        if session not in (Session.MORNING, Session.AFTERNOON):
            return TradeSignal(Signal.HOLD, reason=f"Outside trading session ({session})")

        # --- 3. Session state update ---
        self._update_session_state(parsed.dt, session)

        # --- 4. Position check ---
        if position is not None and not position.is_flat:
            return TradeSignal(Signal.HOLD, reason="Position already open")

        # --- 5. Trade limit ---
        if self._trades_this_session >= self._p.max_trades_per_session:
            return TradeSignal(
                Signal.HOLD,
                reason=f"Session trade limit ({self._trades_this_session}/{self._p.max_trades_per_session})",
            )

        # --- 6. Formation phase ---
        if self._is_in_formation_window(parsed.dt, session):
            self._update_formation(parsed.high, parsed.low)
            return TradeSignal(
                Signal.HOLD,
                reason=f"Forming range: H={self._range_high:.1f} L={self._range_low:.1f}",
            )

        # Mark range as formed on first bar outside formation window
        if not self._range_formed:
            self._range_formed = True
            logger.debug(
                "ORB range formed (%s): H=%.2f L=%.2f size=%.2f",
                session.value,
                self._range_high,
                self._range_low,
                self._range_high - self._range_low,
            )

        # --- 7. Breakout phase ---
        atr = self._get_atr(bar)
        if atr is None:
            return TradeSignal(Signal.HOLD, reason="ATR not available")

        # Range quality filters (with adaptive adjustment)
        range_filter = self._check_range_filters(bar, atr)
        if range_filter is not None:
            return range_filter

        # Optional filters (volume, ADX)
        optional_filter = self._check_optional_filters(bar, atr)
        if optional_filter is not None:
            return optional_filter

        # Get volatility regime for adaptive buffer
        regime, _, buffer_multiplier = self._get_volatility_regime(bar, atr)
        adjusted_buffer = self._p.breakout_buffer * buffer_multiplier

        # Breakout levels
        buffer = adjusted_buffer * atr
        breakout_high = self._range_high + buffer
        breakout_low = self._range_low - buffer
        range_size = self._range_high - self._range_low

        # Breakout validation check
        is_long_breakout = False
        is_short_breakout = False

        if self._p.require_close_confirmation:
            is_long_breakout = parsed.close > breakout_high
            is_short_breakout = parsed.close < breakout_low
        else:
            is_long_breakout = parsed.high > breakout_high
            is_short_breakout = parsed.low < breakout_low

        # Debug logging for breakout checks
        logger.debug(
            "Breakout check: H=%.2f L=%.2f C=%.2f | range[%.2f-%.2f] buffer=%.2f | "
            "breakout_high=%.2f breakout_low=%.2f | long=%s short=%s | regime=%s",
            parsed.high,
            parsed.low,
            parsed.close,
            self._range_low,
            self._range_high,
            buffer,
            breakout_high,
            breakout_low,
            is_long_breakout,
            is_short_breakout,
            regime,
        )

        # Long breakout
        if is_long_breakout:
            # For LIMIT orders: use breakout level (more conservative, may miss some trades)
            # For MARKET orders: use close (will fill at next bar open)
            if self._risk.entry_ord_type == "LIMIT":
                entry_price = breakout_high
            else:
                entry_price = parsed.close if self._p.require_close_confirmation else breakout_high
            return self._build_long_signal(entry_price, atr, range_size, session, is_warmup)

        # Short breakout
        if not self._p.long_only and is_short_breakout:
            # For LIMIT orders: use breakout level (more conservative, may miss some trades)
            # For MARKET orders: use close (will fill at next bar open)
            if self._risk.entry_ord_type == "LIMIT":
                entry_price = breakout_low
            else:
                entry_price = parsed.close if self._p.require_close_confirmation else breakout_low
            return self._build_short_signal(entry_price, atr, range_size, session, is_warmup)

        # No breakout - provide detailed reason
        if self._p.require_close_confirmation:
            # Using close confirmation
            if parsed.close <= breakout_high and parsed.close >= breakout_low:
                reason = (
                    f"No breakout: C={parsed.close:.2f} in range "
                    f"[{breakout_low:.2f}-{breakout_high:.2f}] "
                    f"(need C>{breakout_high:.2f} or C<{breakout_low:.2f})"
                )
            elif parsed.close > self._range_high and parsed.close <= breakout_high:
                reason = (
                    f"No breakout: C={parsed.close:.2f} above range_high={self._range_high:.2f} "
                    f"but below breakout_high={breakout_high:.2f} (buffer={buffer:.2f})"
                )
            elif parsed.close < self._range_low and parsed.close >= breakout_low:
                reason = (
                    f"No breakout: C={parsed.close:.2f} below range_low={self._range_low:.2f} "
                    f"but above breakout_low={breakout_low:.2f} (buffer={buffer:.2f})"
                )
            else:
                reason = f"No breakout: C={parsed.close:.2f} not beyond breakout levels"
        else:
            # Using high/low confirmation
            if parsed.high <= breakout_high and parsed.low >= breakout_low:
                reason = (
                    f"No breakout: H={parsed.high:.2f} L={parsed.low:.2f} in range "
                    f"[{breakout_low:.2f}-{breakout_high:.2f}] "
                    f"(need H>{breakout_high:.2f} or L<{breakout_low:.2f})"
                )
            elif parsed.high > self._range_high and parsed.high <= breakout_high:
                reason = (
                    f"No breakout: H={parsed.high:.2f} above range_high={self._range_high:.2f} "
                    f"but below breakout_high={breakout_high:.2f} (buffer={buffer:.2f})"
                )
            elif parsed.low < self._range_low and parsed.low >= breakout_low:
                reason = (
                    f"No breakout: L={parsed.low:.2f} below range_low={self._range_low:.2f} "
                    f"but above breakout_low={breakout_low:.2f} (buffer={buffer:.2f})"
                )
            else:
                reason = f"No breakout: H={parsed.high:.2f} L={parsed.low:.2f} not beyond breakout levels"

        return TradeSignal(Signal.HOLD, reason=reason)

    # --- Signal builders ---

    def _build_long_signal(
        self,
        entry_price: float,
        atr: float,
        range_size: float,
        session: Session,
        is_warmup: bool,
    ) -> TradeSignal:
        stop_loss = (
            self._range_low
            if self._p.use_range_sl
            else entry_price - self._p.atr_sl_multiplier * atr
        )
        take_profit = entry_price + self._p.atr_tp_multiplier * atr

        if take_profit <= entry_price:
            return TradeSignal(Signal.HOLD, reason="Invalid Long: TP <= entry")
        if stop_loss >= entry_price:
            return TradeSignal(Signal.HOLD, reason="Invalid Long: SL >= entry")

        if not is_warmup:
            self._trades_this_session += 1

        return TradeSignal(
            signal=Signal.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ord_type=self._risk.entry_ord_type,
            reason=f"ORB Long ({session.value}, range={range_size:.1f})",
            metadata={
                "session": session.value,
                "range_high": self._range_high,
                "range_low": self._range_low,
                "range_size": range_size,
                "range_in_atr": range_size / atr,
                "atr": atr,
                "sl_type": "range" if self._p.use_range_sl else "atr",
            },
        )

    def _build_short_signal(
        self,
        entry_price: float,
        atr: float,
        range_size: float,
        session: Session,
        is_warmup: bool,
    ) -> TradeSignal:
        stop_loss = (
            self._range_high
            if self._p.use_range_sl
            else entry_price + self._p.atr_sl_multiplier * atr
        )
        take_profit = entry_price - self._p.atr_tp_multiplier * atr

        if take_profit >= entry_price:
            return TradeSignal(Signal.HOLD, reason="Invalid Short: TP >= entry")
        if stop_loss <= entry_price:
            return TradeSignal(Signal.HOLD, reason="Invalid Short: SL <= entry")

        if not is_warmup:
            self._trades_this_session += 1

        return TradeSignal(
            signal=Signal.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            ord_type=self._risk.entry_ord_type,
            reason=f"ORB Short ({session.value}, range={range_size:.1f})",
            metadata={
                "session": session.value,
                "range_high": self._range_high,
                "range_low": self._range_low,
                "range_size": range_size,
                "range_in_atr": range_size / atr,
                "atr": atr,
                "sl_type": "range" if self._p.use_range_sl else "atr",
            },
        )

    # --- Filters ---

    def _get_volatility_regime(self, bar: dict[str, Any], atr: float) -> tuple[str, float, float]:
        """
        Determine volatility regime and return adjusted multipliers.

        Returns:
            tuple: (regime_name, range_multiplier, buffer_multiplier)
                - regime_name: "low", "normal", or "high"
                - range_multiplier: adjustment for min_range_atr
                - buffer_multiplier: adjustment for breakout_buffer
        """
        if not self._p.use_adaptive_volatility or self._atr_ma_col is None:
            return ("normal", 1.0, 1.0)

        atr_ma = float(bar.get(self._atr_ma_col, 0.0) or 0.0)
        if atr_ma <= 0:
            return ("normal", 1.0, 1.0)

        atr_ratio = atr / atr_ma

        if atr_ratio < self._p.volatility_low_threshold:
            # Low volatility: relax range filter, tighten buffer
            return (
                "low",
                self._p.low_vol_range_multiplier,
                self._p.low_vol_buffer_multiplier,
            )
        elif atr_ratio > self._p.volatility_high_threshold:
            # High volatility: tighten range filter, widen buffer
            return (
                "high",
                self._p.high_vol_range_multiplier,
                self._p.high_vol_buffer_multiplier,
            )
        else:
            # Normal volatility
            return ("normal", 1.0, 1.0)

    def _check_range_filters(self, bar: dict[str, Any], atr: float) -> TradeSignal | None:
        """None = pass, TradeSignal(HOLD) = filtered out."""
        range_size = self._range_high - self._range_low
        if range_size <= 0:
            return TradeSignal(Signal.HOLD, reason="Invalid Range size <= 0")

        range_in_atr = range_size / atr

        # Get volatility regime and adjusted multipliers
        regime, range_multiplier, _ = self._get_volatility_regime(bar, atr)
        adjusted_min_range = self._p.min_range_atr * range_multiplier
        adjusted_max_range = self._p.max_range_atr * range_multiplier

        if self._p.min_range_atr > 0 and range_in_atr < adjusted_min_range:
            reason = f"Range too narrow: {range_in_atr:.2f}x ATR (min {adjusted_min_range:.2f}x)"
            if regime != "normal":
                reason += f" [vol={regime}]"
            return TradeSignal(Signal.HOLD, reason=reason)

        if self._p.max_range_atr > 0 and range_in_atr > adjusted_max_range:
            reason = f"Range too wide: {range_in_atr:.2f}x ATR (max {adjusted_max_range:.2f}x)"
            if regime != "normal":
                reason += f" [vol={regime}]"
            return TradeSignal(Signal.HOLD, reason=reason)

        return None

    def _check_optional_filters(self, bar: dict[str, Any], atr: float) -> TradeSignal | None:
        """Volume and ADX filters. None = pass."""
        if self._p.use_volume_filter:
            volume = float(bar.get("volume", 0.0) or 0.0)
            volume_ma = float(bar.get(self._vol_ma_col, 0.0) or 0.0)
            threshold = self._p.volume_filter_threshold

            if volume_ma > 0 and volume < threshold * volume_ma:
                return TradeSignal(
                    Signal.HOLD,
                    reason=(
                        f"Volume below threshold: {volume:.0f} < {threshold:.1f} * {volume_ma:.0f}"
                    ),
                )

        if self._p.use_adx_filter:
            adx = float(bar.get(self._adx_col, 0.0) or 0.0)
            if adx < self._p.adx_min:
                return TradeSignal(
                    Signal.HOLD,
                    reason=f"ADX too low: {adx:.1f} < {self._p.adx_min}",
                )

        return None

    # --- Helpers ---

    def _get_atr(self, bar: dict[str, Any]) -> float | None:
        """Get ATR value from bar. Returns None if missing or <= 0."""
        val = bar.get(self._atr_col)
        if val is None:
            return None
        try:
            atr = float(val)
            return atr if atr > 0 else None
        except (TypeError, ValueError):
            return None

    def _get_orb_minutes(self) -> int:
        return self._p.orb_minutes

    def _update_formation(self, high: float, low: float) -> None:
        if high > self._range_high:
            self._range_high = high
        if low < self._range_low:
            self._range_low = low

    def _is_in_formation_window(self, dt: datetime, session: Session) -> bool:
        """Check if current datetime is within the opening range formation window."""
        orb_minutes = self._get_orb_minutes()
        if session == Session.MORNING:
            session_start = self._session_cfg.MORNING_START
        elif session == Session.AFTERNOON:
            session_start = self._session_cfg.AFTERNOON_START
        else:
            return False  # Should not happen due to earlier session check

        start_minutes = session_start.hour * 60 + session_start.minute
        current_minutes = dt.hour * 60 + dt.minute
        return (current_minutes - start_minutes) < orb_minutes

    @staticmethod
    def _parse_bar(bar: dict[str, Any]) -> _ParsedBar | None:
        """Parse and coerce bar fields. Returns None if invalid."""
        dt = bar.get("datetime")
        if dt is None:
            return None
        if not isinstance(dt, datetime):
            try:
                dt = datetime.fromisoformat(str(dt))
            except (TypeError, ValueError):
                return None
        try:
            return _ParsedBar(
                dt=dt,
                close=float(bar["close"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    # --- Session reset ---

    def _on_session_reset(self, session: Session) -> None:
        """
        Reset opening range state when a new session starts.

        Args:
            session: The new session (or Session.CLOSED during full reset)
        """
        self._range_high = 0.0
        self._range_low = float("inf")
        self._range_formed = False
        self._trades_this_session = 0

    # --- State serialization ---

    def _get_strategy_state(self) -> dict[str, Any]:
        return {
            "range_high": self._range_high,
            "range_low": self._range_low,
            "range_formed": self._range_formed,
            "trades_this_session": self._trades_this_session,
        }

    def _set_strategy_state(self, state: dict[str, Any]) -> None:
        self._range_high = float(state.get("range_high", 0.0))
        self._range_low = float(state.get("range_low", float("inf")))
        self._range_formed = bool(state.get("range_formed", False))
        self._trades_this_session = int(state.get("trades_this_session", 0))

    # --- Properties (read-only access for engine/debug) ---

    @property
    def range_high(self) -> float:
        return self._range_high

    @property
    def range_low(self) -> float:
        return self._range_low

    @property
    def range_formed(self) -> bool:
        return self._range_formed

    @property
    def trades_this_session(self) -> int:
        return self._trades_this_session
