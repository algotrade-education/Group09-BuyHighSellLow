"""
Unit tests for the OpeningRangeBreakout (ORB) strategy.

Tests the signal generation logic end-to-end using synthetic bar dicts,
covering every code path in generate_signal().

VN30 session schedule:
    Morning   09:00 - 11:30
    Afternoon 13:00 - 14:30

Strategy rules tested:
    - Formation bars (within first N minutes) → HOLD and update range
    - Long breakout: close > range_high + buffer * ATR → LONG
    - Short breakout: close < range_low  - buffer * ATR → SHORT
    - Range size filter (too narrow / too wide) → HOLD
    - Already-traded-this-session guard → HOLD
    - Session boundary: new session resets state
    - long_only mode: no SHORT signals
    - Volume / ADX optional filters
    - Out-of-session bars → HOLD
    - ATR == 0 → HOLD
    - Position already open → HOLD

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_strategy_orb.py -v
"""

from datetime import datetime, date
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from src.strategy.ORB import OpeningRangeBreakout
from src.strategy.base import Signal


# ── Helpers ────────────────────────────────────────────────────────────────


def _bar(
    dt: datetime,
    open_: float = 1300.0,
    high: float = 1310.0,
    low: float = 1290.0,
    close: float = 1300.0,
    atr: float = 10.0,
    atr_period: int = 14,
    volume: float = 200.0,
    volume_ma: float = 150.0,
    adx: float = 25.0,
) -> Dict[str, Any]:
    """Build a minimal bar dict for a given datetime."""
    return {
        "datetime": dt,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        f"atr_{atr_period}": atr,
        "volume": volume,
        "volume_ma_20": volume_ma,
        f"adx_{atr_period}": adx,
    }


def _morning(minute: int, **kw) -> Dict[str, Any]:
    """Bar in the morning session on a fixed date."""
    dt = datetime(2025, 1, 6, 9, minute, 0)
    return _bar(dt, **kw)


def _afternoon(minute: int, **kw) -> Dict[str, Any]:
    """Bar in the afternoon session (13:XX) on the same date."""
    dt = datetime(2025, 1, 6, 13, minute, 0)
    return _bar(dt, **kw)


def _flat_position():
    """Return a mock position that reports is_flat=True."""
    pos = MagicMock()
    pos.is_flat = True
    return pos


def _open_position():
    """Return a mock position that reports is_flat=False (position is open)."""
    pos = MagicMock()
    pos.is_flat = False
    return pos


def _strategy(**kwargs) -> OpeningRangeBreakout:
    """Create an ORB strategy with sensible test defaults."""
    defaults = dict(
        orb_minutes=15,
        atr_period=14,
        atr_tp_multiplier=2.0,
        atr_sl_multiplier=1.5,
        breakout_buffer=0.0,  # no buffer so breakout = exact range boundary
        use_range_sl=True,
        min_range_atr=0.5,
        max_range_atr=3.0,
        long_only=False,
    )
    defaults.update(kwargs)
    return OpeningRangeBreakout(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# Formation phase
# ═══════════════════════════════════════════════════════════════════════════


class TestORBFormationPhase:
    """Bars within the first orb_minutes should HOLD and update the range."""

    def test_formation_bar_returns_hold(self):
        """A bar inside the opening window must return HOLD."""
        strat = _strategy()
        bar = _morning(0)  # 09:00 — inside 15-min window
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.HOLD

    def test_formation_extends_range_high(self):
        """Each formation bar should extend _range_high to the bar's high."""
        strat = _strategy()
        strat.generate_signal(_morning(0, high=1310.0))
        strat.generate_signal(_morning(5, high=1325.0))
        assert strat._range_high == 1325.0

    def test_formation_extends_range_low(self):
        """Each formation bar should contract _range_low to the bar's low."""
        strat = _strategy()
        strat.generate_signal(_morning(0, low=1290.0))
        strat.generate_signal(_morning(5, low=1280.0))
        assert strat._range_low == 1280.0

    def test_formation_does_not_fire_signal(self):
        """No LONG or SHORT during formation — only HOLD."""
        strat = _strategy()
        for minute in range(0, 15, 5):
            sig = strat.generate_signal(_morning(minute))
            assert sig.signal == Signal.HOLD, (
                f"Unexpected {sig.signal} at minute {minute}"
            )

    def test_out_of_session_returns_hold(self):
        """Bars outside both sessions (e.g. 12:00 lunch) must return HOLD."""
        strat = _strategy()
        dt = datetime(2025, 1, 6, 12, 0, 0)  # lunch break
        sig = strat.generate_signal(_bar(dt))
        assert sig.signal == Signal.HOLD

    def test_formation_window_exactly_at_boundary(self):
        """Bar at minute==orb_minutes is NOT in the formation window."""
        strat = _strategy(orb_minutes=15)  # window is [0, 15)
        # Feed 09:00-09:10 as formation
        strat.generate_signal(_morning(0, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(5, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(10, high=1310.0, low=1290.0))
        # 09:15 is the first breakout bar (elapsed=15 == orb_minutes → NOT in window)
        bar = _morning(15, close=1315.0, high=1315.0, low=1305.0, atr=10.0)
        sig = strat.generate_signal(bar)
        # Should not be HOLD due to formation (may be LONG or HOLD for other reasons)
        assert strat._range_formed  # range must be marked formed now


# ═══════════════════════════════════════════════════════════════════════════
# Long breakout
# ═══════════════════════════════════════════════════════════════════════════


class TestORBLongBreakout:
    """Close > range_high (+ buffer) → LONG signal with correct SL/TP."""

    def _setup_range(
        self,
        strat: OpeningRangeBreakout,
        high: float = 1310.0,
        low: float = 1290.0,
    ):
        """Feed two formation bars to establish the range."""
        strat.generate_signal(_morning(0, high=high, low=low))
        strat.generate_signal(_morning(5, high=high, low=low))
        strat.generate_signal(_morning(10, high=high, low=low))

    def test_long_signal_on_upward_breakout(self):
        """close > range_high + buffer → LONG."""
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        self._setup_range(strat, high=1310.0, low=1290.0)

        # close must be strictly > 1310 (range_high)
        bar = _morning(15, close=1311.0, high=1312.0, low=1305.0, atr=10.0)
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.LONG

    def test_long_signal_stop_loss_at_range_low(self):
        """use_range_sl=True → stop_loss == range_low."""
        strat = _strategy(
            breakout_buffer=0.0,
            use_range_sl=True,
            min_range_atr=0.5,
            max_range_atr=10.0,
        )
        self._setup_range(strat, high=1310.0, low=1290.0)

        bar = _morning(15, close=1311.0, high=1312.0, low=1305.0, atr=10.0)
        sig = strat.generate_signal(bar)

        assert sig.signal == Signal.LONG
        assert sig.stop_loss == pytest.approx(1290.0)

    def test_long_signal_take_profit_atr_based(self):
        """take_profit = close + atr_tp_multiplier * ATR."""
        strat = _strategy(
            breakout_buffer=0.0,
            atr_tp_multiplier=2.0,
            use_range_sl=True,
            min_range_atr=0.5,
            max_range_atr=10.0,
        )
        self._setup_range(strat, high=1310.0, low=1290.0)

        close = 1311.0
        atr = 10.0
        bar = _morning(15, close=close, high=1312.0, low=1305.0, atr=atr)
        sig = strat.generate_signal(bar)

        assert sig.take_profit == pytest.approx(close + 2.0 * atr)

    def test_long_signal_with_atr_sl(self):
        """use_range_sl=False → stop_loss = close - atr_sl_multiplier * ATR."""
        strat = _strategy(
            breakout_buffer=0.0,
            use_range_sl=False,
            atr_sl_multiplier=1.5,
            min_range_atr=0.5,
            max_range_atr=10.0,
        )
        self._setup_range(strat, high=1310.0, low=1290.0)

        close = 1311.0
        atr = 10.0
        bar = _morning(15, close=close, high=1312.0, low=1305.0, atr=atr)
        sig = strat.generate_signal(bar)

        assert sig.stop_loss == pytest.approx(close - 1.5 * atr)

    def test_close_exactly_at_range_high_does_not_signal(self):
        """close == range_high (not strictly above) → no LONG (still HOLD)."""
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        self._setup_range(strat, high=1310.0, low=1290.0)

        bar = _morning(15, close=1310.0, high=1310.0, low=1300.0, atr=10.0)
        sig = strat.generate_signal(bar)
        assert sig.signal != Signal.LONG

    def test_long_with_breakout_buffer(self):
        """With breakout_buffer=0.5: breakout_high = range_high + 0.5*ATR."""
        atr = 10.0
        strat = _strategy(breakout_buffer=0.5, min_range_atr=0.5, max_range_atr=10.0)
        self._setup_range(strat, high=1310.0, low=1290.0)

        # breakout_high = 1310 + 0.5*10 = 1315; close=1314 → still in range → HOLD
        bar_below = _morning(15, close=1314.0, high=1315.0, low=1305.0, atr=atr)
        sig = strat.generate_signal(bar_below)
        assert sig.signal != Signal.LONG

        # close=1316 → above breakout_high → LONG
        strat2 = _strategy(breakout_buffer=0.5, min_range_atr=0.5, max_range_atr=10.0)
        self._setup_range(strat2, high=1310.0, low=1290.0)
        bar_above = _morning(15, close=1316.0, high=1317.0, low=1305.0, atr=atr)
        sig2 = strat2.generate_signal(bar_above)
        assert sig2.signal == Signal.LONG


# ═══════════════════════════════════════════════════════════════════════════
# Short breakout
# ═══════════════════════════════════════════════════════════════════════════


class TestORBShortBreakout:
    """Close < range_low (- buffer) → SHORT signal."""

    def _setup_range(self, strat, high=1310.0, low=1290.0):
        strat.generate_signal(_morning(0, high=high, low=low))
        strat.generate_signal(_morning(5, high=high, low=low))
        strat.generate_signal(_morning(10, high=high, low=low))

    def test_short_signal_on_downward_breakout(self):
        """close < range_low → SHORT."""
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        self._setup_range(strat, high=1310.0, low=1290.0)

        bar = _morning(15, close=1289.0, high=1295.0, low=1288.0, atr=10.0)
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.SHORT

    def test_short_stop_loss_at_range_high(self):
        """use_range_sl=True for SHORT → stop_loss == range_high."""
        strat = _strategy(
            breakout_buffer=0.0,
            use_range_sl=True,
            min_range_atr=0.5,
            max_range_atr=10.0,
        )
        self._setup_range(strat, high=1310.0, low=1290.0)

        bar = _morning(15, close=1289.0, high=1295.0, low=1288.0, atr=10.0)
        sig = strat.generate_signal(bar)
        assert sig.stop_loss == pytest.approx(1310.0)

    def test_short_take_profit_atr_based(self):
        """take_profit = close - atr_tp_multiplier * ATR."""
        strat = _strategy(
            breakout_buffer=0.0,
            atr_tp_multiplier=2.0,
            use_range_sl=True,
            min_range_atr=0.5,
            max_range_atr=10.0,
        )
        self._setup_range(strat, high=1310.0, low=1290.0)

        close = 1289.0
        atr = 10.0
        bar = _morning(15, close=close, high=1295.0, low=1288.0, atr=atr)
        sig = strat.generate_signal(bar)
        assert sig.take_profit == pytest.approx(close - 2.0 * atr)

    def test_long_only_suppresses_short(self):
        """With long_only=True, downward breakout returns HOLD not SHORT."""
        strat = _strategy(
            breakout_buffer=0.0, long_only=True, min_range_atr=0.5, max_range_atr=10.0
        )
        self._setup_range(strat, high=1310.0, low=1290.0)

        bar = _morning(15, close=1285.0, high=1295.0, low=1284.0, atr=10.0)
        sig = strat.generate_signal(bar)
        assert sig.signal != Signal.SHORT


# ═══════════════════════════════════════════════════════════════════════════
# Range size filters
# ═══════════════════════════════════════════════════════════════════════════


class TestORBRangeFilters:
    """Range too narrow or too wide → HOLD (filtered out)."""

    def _setup_and_breakout(self, strat, high, low, atr=10.0):
        """Set up range with given high/low, then send a breakout bar."""
        strat.generate_signal(_morning(0, high=high, low=low))
        strat.generate_signal(_morning(5, high=high, low=low))
        strat.generate_signal(_morning(10, high=high, low=low))
        # Send bar clearly above the range
        bar = _morning(15, close=high + 5.0, high=high + 6.0, low=high + 1.0, atr=atr)
        return strat.generate_signal(bar)

    def test_range_too_narrow_is_filtered(self):
        """range < min_range_atr * ATR → HOLD."""
        # ATR=10, range=3 → range_in_atr=0.3 < min_range_atr=0.5
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        sig = self._setup_and_breakout(strat, high=1303.0, low=1300.0, atr=10.0)
        assert sig.signal == Signal.HOLD

    def test_range_too_wide_is_filtered(self):
        """range > max_range_atr * ATR → HOLD."""
        # ATR=10, range=50 → range_in_atr=5 > max_range_atr=3.0
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=3.0)
        sig = self._setup_and_breakout(strat, high=1350.0, low=1300.0, atr=10.0)
        assert sig.signal == Signal.HOLD

    def test_valid_range_size_allows_signal(self):
        """range within [min_range_atr, max_range_atr] allows breakout signal."""
        # ATR=10, range=20 → range_in_atr=2.0 — within [0.5, 3.0]
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=3.0)
        sig = self._setup_and_breakout(strat, high=1320.0, low=1300.0, atr=10.0)
        assert sig.signal == Signal.LONG

    def test_zero_atr_returns_hold(self):
        """ATR == 0 must return HOLD (division guard)."""
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        strat.generate_signal(_morning(0, high=1310.0, low=1290.0, atr=10.0))
        strat.generate_signal(_morning(5, high=1310.0, low=1290.0, atr=10.0))
        strat.generate_signal(_morning(10, high=1310.0, low=1290.0, atr=10.0))
        # Now breakout bar with ATR=0
        bar = _morning(15, close=1315.0, high=1316.0, low=1310.0, atr=0.0)
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# Already-traded guard & session reset
# ═══════════════════════════════════════════════════════════════════════════


class TestORBSessionManagement:
    """Max 1 trade per session; new session resets state."""

    def _do_trade(self, strat: OpeningRangeBreakout, session: str = "morning"):
        """Feed formation + breakout bars to trigger one trade."""
        if session == "morning":
            make = _morning
        else:
            make = _afternoon

        strat.generate_signal(make(0, high=1310.0, low=1290.0))
        strat.generate_signal(make(5, high=1310.0, low=1290.0))
        strat.generate_signal(make(10, high=1310.0, low=1290.0))
        entry = make(15, close=1315.0, high=1316.0, low=1310.0, atr=10.0)
        sig = strat.generate_signal(entry)
        assert sig.signal == Signal.LONG, f"Expected LONG, got {sig.signal}"

    def test_second_breakout_in_same_session_is_held(self):
        """After trading once, further breakouts in the same session → HOLD."""
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        self._do_trade(strat, "morning")

        # Another clear breakout bar in the same session
        bar = _morning(20, close=1320.0, high=1321.0, low=1314.0, atr=10.0)
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.HOLD

    def test_new_session_resets_traded_flag(self):
        """After morning trade, the afternoon session allows a fresh trade."""
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        self._do_trade(strat, "morning")

        # Afternoon formation + breakout
        strat.generate_signal(_afternoon(0, high=1320.0, low=1300.0))
        strat.generate_signal(_afternoon(5, high=1320.0, low=1300.0))
        strat.generate_signal(_afternoon(10, high=1320.0, low=1300.0))
        bar = _afternoon(15, close=1325.0, high=1326.0, low=1318.0, atr=10.0)
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.LONG

    def test_new_date_resets_session_state(self):
        """A bar from a different date starts a fresh session (range reset)."""
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)

        # Day 1 formation
        d1 = datetime(2025, 1, 6, 9, 0)
        strat.generate_signal(_bar(d1, high=1310.0, low=1290.0))

        # Day 2 — state should reset on first bar
        d2 = datetime(2025, 1, 7, 9, 0)
        strat.generate_signal(_bar(d2, high=1310.0, low=1290.0))
        assert strat._current_date == d2.date()
        assert not strat._range_formed  # just started day 2 formation

    def test_position_open_returns_hold(self):
        """If current_position is not flat, always return HOLD."""
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        strat.generate_signal(_morning(0, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(5, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(10, high=1310.0, low=1290.0))

        bar = _morning(15, close=1315.0, high=1316.0, low=1310.0, atr=10.0)
        sig = strat.generate_signal(bar, current_position=_open_position())
        assert sig.signal == Signal.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# Optional filters
# ═══════════════════════════════════════════════════════════════════════════


class TestORBOptionalFilters:
    """Volume and ADX filters can suppress otherwise-valid breakout signals."""

    def _setup_range(self, strat):
        strat.generate_signal(_morning(0, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(5, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(10, high=1310.0, low=1290.0))

    def test_volume_filter_suppresses_breakout_when_below_ma(self):
        """use_volume_filter=True: volume < volume_ma → HOLD."""
        strat = _strategy(
            breakout_buffer=0.0,
            min_range_atr=0.5,
            max_range_atr=10.0,
            use_volume_filter=True,
        )
        self._setup_range(strat)

        # volume (100) < volume_ma (200) → filtered
        bar = _morning(
            15,
            close=1315.0,
            high=1316.0,
            low=1310.0,
            atr=10.0,
            volume=100.0,
            volume_ma=200.0,
        )
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.HOLD

    def test_volume_filter_allows_breakout_when_above_ma(self):
        """use_volume_filter=True: volume > volume_ma → signal passes."""
        strat = _strategy(
            breakout_buffer=0.0,
            min_range_atr=0.5,
            max_range_atr=10.0,
            use_volume_filter=True,
        )
        self._setup_range(strat)

        bar = _morning(
            15,
            close=1315.0,
            high=1316.0,
            low=1310.0,
            atr=10.0,
            volume=300.0,
            volume_ma=200.0,
        )
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.LONG

    def test_adx_filter_suppresses_breakout_when_adx_too_low(self):
        """use_adx_filter=True: ADX < adx_min → HOLD."""
        strat = _strategy(
            breakout_buffer=0.0,
            min_range_atr=0.5,
            max_range_atr=10.0,
            use_adx_filter=True,
            adx_min=25.0,
        )
        self._setup_range(strat)

        bar = _morning(
            15, close=1315.0, high=1316.0, low=1310.0, atr=10.0, adx=15.0
        )  # adx below threshold
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.HOLD

    def test_adx_filter_allows_breakout_when_adx_sufficient(self):
        """use_adx_filter=True: ADX >= adx_min → signal passes."""
        strat = _strategy(
            breakout_buffer=0.0,
            min_range_atr=0.5,
            max_range_atr=10.0,
            use_adx_filter=True,
            adx_min=25.0,
        )
        self._setup_range(strat)

        bar = _morning(15, close=1315.0, high=1316.0, low=1310.0, atr=10.0, adx=30.0)
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.LONG

    def test_both_filters_off_no_suppression(self):
        """With both filters off, valid breakout signals regardless of volume/ADX."""
        strat = _strategy(
            breakout_buffer=0.0,
            min_range_atr=0.5,
            max_range_atr=10.0,
            use_volume_filter=False,
            use_adx_filter=False,
        )
        self._setup_range(strat)

        # Terrible volume and ADX — but filters are off
        bar = _morning(
            15,
            close=1315.0,
            high=1316.0,
            low=1310.0,
            atr=10.0,
            volume=1.0,
            volume_ma=9999.0,
            adx=1.0,
        )
        sig = strat.generate_signal(bar)
        assert sig.signal == Signal.LONG


# ═══════════════════════════════════════════════════════════════════════════
# Signal metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestORBSignalMetadata:
    """Breakout signals should carry correct metadata fields."""

    def test_long_signal_has_expected_metadata_keys(self):
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        strat.generate_signal(_morning(0, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(5, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(10, high=1310.0, low=1290.0))

        bar = _morning(15, close=1315.0, high=1316.0, low=1310.0, atr=10.0)
        sig = strat.generate_signal(bar)

        assert sig.signal == Signal.LONG
        assert sig.metadata is not None
        for key in (
            "session",
            "range_high",
            "range_low",
            "range_size",
            "atr",
            "sl_type",
        ):
            assert key in sig.metadata, f"Missing metadata key: {key}"

    def test_long_signal_session_is_morning(self):
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        strat.generate_signal(_morning(0, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(5, high=1310.0, low=1290.0))
        strat.generate_signal(_morning(10, high=1310.0, low=1290.0))

        bar = _morning(15, close=1315.0, high=1316.0, low=1310.0, atr=10.0)
        sig = strat.generate_signal(bar)
        assert sig.metadata["session"] == "morning"

    def test_short_signal_session_is_afternoon(self):
        strat = _strategy(breakout_buffer=0.0, min_range_atr=0.5, max_range_atr=10.0)
        strat.generate_signal(_afternoon(0, high=1310.0, low=1290.0))
        strat.generate_signal(_afternoon(5, high=1310.0, low=1290.0))
        strat.generate_signal(_afternoon(10, high=1310.0, low=1290.0))

        bar = _afternoon(15, close=1285.0, high=1292.0, low=1284.0, atr=10.0)
        sig = strat.generate_signal(bar)

        assert sig.signal == Signal.SHORT
        assert sig.metadata["session"] == "afternoon"
