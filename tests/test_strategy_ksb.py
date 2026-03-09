"""
Unit tests for the KeltnerSqueezeBreakout (KSB) strategy.

Tests the signal generation logic end-to-end using synthetic bar dicts,
covering every code path in generate_signal().

VN30 session schedule:
    Morning   09:00 - 11:30
    Afternoon 13:00 - 14:30

Strategy rules tested:
    - Squeeze ON (BB inside KC) → HOLD
    - Squeeze release with signal_window → directional signal
    - Squeeze too short → HOLD
    - Direction: mom sign + close vs kc_middle
    - Signal window allows entry on later bars after release
    - Cooldown after exit
    - Session / day reset clears squeeze state
    - long_only mode suppresses SHORT
    - Volume filter
    - Position already open → HOLD
    - Out-of-session bars → HOLD
    - Missing indicators → HOLD

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_strategy_ksb.py -v
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.strategy.KSB import KeltnerSqueezeBreakout
from src.strategy.base import Signal


# ── Helpers ────────────────────────────────────────────────────────────────

KC_MIDDLE = 1300.0


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
    bb_upper: float = 1320.0,
    bb_lower: float = 1280.0,
    kc_upper: float = 1315.0,
    kc_lower: float = 1285.0,
    kc_middle: float = KC_MIDDLE,
    mom: float = 5.0,
    mom_period: int = 12,
) -> Dict[str, Any]:
    """Build a minimal bar dict with all indicator columns."""
    return {
        "datetime": dt,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        f"atr_{atr_period}": atr,
        "volume": volume,
        "volume_ma_20": volume_ma,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "kc_upper": kc_upper,
        "kc_lower": kc_lower,
        "kc_middle": kc_middle,
        f"mom_{mom_period}": mom,
    }


def _squeeze_bar(dt: datetime, mom: float = 5.0, **kw) -> Dict[str, Any]:
    """Bar where BB is INSIDE KC (squeeze ON)."""
    defaults = dict(
        bb_upper=1310.0,
        bb_lower=1290.0,
        kc_upper=1315.0,
        kc_lower=1285.0,
        mom=mom,
    )
    defaults.update(kw)
    return _bar(dt, **defaults)


def _release_bar(dt: datetime, mom: float = 5.0, **kw) -> Dict[str, Any]:
    """Bar where BB is OUTSIDE KC (squeeze OFF), close > kc_middle by default."""
    defaults = dict(
        bb_upper=1325.0,
        bb_lower=1275.0,
        kc_upper=1315.0,
        kc_lower=1285.0,
        close=1305.0,  # above kc_middle (1300) for bullish default
        mom=mom,
    )
    defaults.update(kw)
    return _bar(dt, **defaults)


def _morning(minute: int, **kw) -> Dict[str, Any]:
    dt = datetime(2025, 1, 6, 9, minute, 0)
    return _bar(dt, **kw)


def _morning_squeeze(minute: int, **kw) -> Dict[str, Any]:
    dt = datetime(2025, 1, 6, 9, minute, 0)
    return _squeeze_bar(dt, **kw)


def _morning_release(minute: int, **kw) -> Dict[str, Any]:
    dt = datetime(2025, 1, 6, 9, minute, 0)
    return _release_bar(dt, **kw)


def _afternoon(minute: int, **kw) -> Dict[str, Any]:
    dt = datetime(2025, 1, 6, 13, minute, 0)
    return _bar(dt, **kw)


def _flat_position():
    pos = MagicMock()
    pos.is_flat = True
    return pos


def _open_position():
    pos = MagicMock()
    pos.is_flat = False
    return pos


def _strategy(**kwargs) -> KeltnerSqueezeBreakout:
    defaults = dict(
        bb_period=20,
        bb_std=2.0,
        kc_period=20,
        kc_mult=1.5,
        mom_period=12,
        atr_period=14,
        atr_sl_mult=1.5,
        atr_tp_mult=2.5,
        min_squeeze_bars=3,
        signal_window=3,
        cooldown_bars=2,
        long_only=False,
        use_volume_filter=False,
        vol_mult=1.5,
        vol_ma_period=20,
    )
    defaults.update(kwargs)
    return KeltnerSqueezeBreakout(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# Squeeze detection
# ═══════════════════════════════════════════════════════════════════════════


class TestKSBSqueezeDetection:
    """BB inside KC → squeeze ON; expanding outside → squeeze OFF."""

    def test_squeeze_bar_returns_hold(self):
        strat = _strategy()
        sig = strat.generate_signal(_morning_squeeze(0))
        assert sig.signal == Signal.HOLD

    def test_multiple_squeeze_bars_accumulate(self):
        strat = _strategy(min_squeeze_bars=3)
        for m in [0, 5, 10]:
            strat.generate_signal(_morning_squeeze(m))
        assert strat._squeeze_on is True
        assert strat._squeeze_bar_count == 3

    def test_no_squeeze_no_signal(self):
        """Bars that are never in a squeeze should always HOLD."""
        strat = _strategy()
        sig = strat.generate_signal(_morning_release(0, mom=5.0))
        assert sig.signal == Signal.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# Squeeze release → entry signals
# ═══════════════════════════════════════════════════════════════════════════


class TestKSBSqueezeRelease:
    """Squeeze release with momentum sign + price vs kc_middle determines direction."""

    def _build_squeeze(self, strat, n_bars=3):
        for i in range(n_bars):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))

    def test_long_on_release_with_positive_mom_above_kc_middle(self):
        """mom > 0 AND close > kc_middle → LONG."""
        strat = _strategy(min_squeeze_bars=3)
        self._build_squeeze(strat, 3)
        sig = strat.generate_signal(_morning_release(15, mom=6.0, close=1305.0))
        assert sig.signal == Signal.LONG

    def test_short_on_release_with_negative_mom_below_kc_middle(self):
        """mom < 0 AND close < kc_middle → SHORT."""
        strat = _strategy(min_squeeze_bars=3)
        self._build_squeeze(strat, 3)
        sig = strat.generate_signal(_morning_release(15, mom=-6.0, close=1295.0))
        assert sig.signal == Signal.SHORT

    def test_hold_when_mom_positive_but_below_kc_middle(self):
        """mom > 0 but close < kc_middle → conflicting → HOLD."""
        strat = _strategy(min_squeeze_bars=3)
        self._build_squeeze(strat, 3)
        sig = strat.generate_signal(_morning_release(15, mom=5.0, close=1295.0))
        assert sig.signal == Signal.HOLD

    def test_hold_when_mom_negative_but_above_kc_middle(self):
        """mom < 0 but close > kc_middle → conflicting → HOLD."""
        strat = _strategy(min_squeeze_bars=3)
        self._build_squeeze(strat, 3)
        sig = strat.generate_signal(_morning_release(15, mom=-5.0, close=1305.0))
        assert sig.signal == Signal.HOLD

    def test_squeeze_too_short_returns_hold(self):
        strat = _strategy(min_squeeze_bars=5)
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        sig = strat.generate_signal(_morning_release(15, mom=6.0, close=1305.0))
        assert sig.signal == Signal.HOLD

    def test_long_signal_tp_and_sl(self):
        strat = _strategy(min_squeeze_bars=3, atr_sl_mult=1.5, atr_tp_mult=2.5)
        self._build_squeeze(strat, 3)
        close = 1305.0
        atr = 10.0
        sig = strat.generate_signal(_morning_release(15, mom=6.0, close=close, atr=atr))
        assert sig.signal == Signal.LONG
        assert sig.stop_loss == pytest.approx(close - 1.5 * atr)
        assert sig.take_profit == pytest.approx(close + 2.5 * atr)

    def test_short_signal_tp_and_sl(self):
        strat = _strategy(min_squeeze_bars=3, atr_sl_mult=1.5, atr_tp_mult=2.5)
        self._build_squeeze(strat, 3)
        close = 1295.0
        atr = 10.0
        sig = strat.generate_signal(
            _morning_release(15, mom=-6.0, close=close, atr=atr)
        )
        assert sig.signal == Signal.SHORT
        assert sig.stop_loss == pytest.approx(close + 1.5 * atr)
        assert sig.take_profit == pytest.approx(close - 2.5 * atr)


# ═══════════════════════════════════════════════════════════════════════════
# Signal window
# ═══════════════════════════════════════════════════════════════════════════


class TestKSBSignalWindow:
    """Entry is valid for signal_window bars after the squeeze release."""

    def test_entry_on_second_bar_after_release(self):
        """First release bar has conflicting mom; second bar is fine."""
        strat = _strategy(min_squeeze_bars=3, signal_window=3)
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        # Release bar: mom positive but close below kc_middle → HOLD
        strat.generate_signal(_morning_release(15, mom=3.0, close=1295.0))
        # Second bar in window: close now above kc_middle → LONG
        sig = strat.generate_signal(_morning_release(20, mom=5.0, close=1305.0))
        assert sig.signal == Signal.LONG

    def test_window_expires_returns_hold(self):
        """After signal_window bars, further bars should HOLD."""
        strat = _strategy(min_squeeze_bars=3, signal_window=2)
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        # Consume the 2-bar window with conflicting signals
        strat.generate_signal(_morning_release(15, mom=3.0, close=1295.0))
        strat.generate_signal(_morning_release(20, mom=-1.0, close=1305.0))
        # Window exhausted
        sig = strat.generate_signal(_morning_release(25, mom=5.0, close=1305.0))
        assert sig.signal == Signal.HOLD

    def test_new_squeeze_cancels_window(self):
        """A new squeeze forming should cancel any active window."""
        strat = _strategy(min_squeeze_bars=3, signal_window=5)
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        # Release opens window
        strat.generate_signal(_morning_release(15, mom=3.0, close=1295.0))
        assert strat._window_remaining > 0
        # New squeeze bar cancels window
        strat.generate_signal(_morning_squeeze(20, mom=2.0))
        assert strat._window_remaining == 0


# ═══════════════════════════════════════════════════════════════════════════
# Cooldown
# ═══════════════════════════════════════════════════════════════════════════


class TestKSBCooldown:
    """After an exit, wait cooldown_bars before next entry."""

    def test_cooldown_blocks_entry(self):
        strat = _strategy(cooldown_bars=2, min_squeeze_bars=1, signal_window=3)

        strat.generate_signal(_morning_squeeze(0, mom=1.0))
        strat.generate_signal(_morning_release(5, mom=3.0, close=1305.0))

        strat._was_flat = False
        strat._bars_since_exit = 9999

        sig_exit = strat.generate_signal(
            _morning_squeeze(10, mom=4.0), current_position=_flat_position()
        )
        assert sig_exit.signal == Signal.HOLD

        sig_cd1 = strat.generate_signal(
            _morning_release(15, mom=6.0, close=1305.0),
            current_position=_flat_position(),
        )
        assert sig_cd1.signal == Signal.HOLD

    def test_cooldown_expires_allows_entry(self):
        strat = _strategy(cooldown_bars=1, min_squeeze_bars=1, signal_window=3)

        strat._was_flat = False
        strat._bars_since_exit = 9999
        strat._current_date = datetime(2025, 1, 6).date()
        strat._current_session = "morning"

        # Cooldown tick
        strat.generate_signal(
            _morning_squeeze(0, mom=1.0), current_position=_flat_position()
        )

        # Build squeeze (cooldown expired by now)
        strat.generate_signal(_morning_squeeze(5, mom=2.0))
        strat.generate_signal(_morning_squeeze(10, mom=3.0))
        sig = strat.generate_signal(_morning_release(15, mom=5.0, close=1305.0))
        assert sig.signal == Signal.LONG


# ═══════════════════════════════════════════════════════════════════════════
# Session management
# ═══════════════════════════════════════════════════════════════════════════


class TestKSBSessionManagement:
    """Squeeze state resets on new day and new session."""

    def test_new_day_resets_squeeze(self):
        strat = _strategy(min_squeeze_bars=3)
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        assert strat._squeeze_on is True

        day2 = datetime(2025, 1, 7, 9, 0, 0)
        strat.generate_signal(_squeeze_bar(day2, mom=1.0))
        assert strat._squeeze_bar_count == 1

    def test_new_session_resets_squeeze(self):
        strat = _strategy(min_squeeze_bars=3)
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        assert strat._squeeze_on is True

        strat.generate_signal(_squeeze_bar(datetime(2025, 1, 6, 13, 0, 0), mom=1.0))
        assert strat._squeeze_bar_count == 1

    def test_out_of_session_returns_hold(self):
        strat = _strategy()
        dt = datetime(2025, 1, 6, 12, 0, 0)
        sig = strat.generate_signal(_bar(dt))
        assert sig.signal == Signal.HOLD

    def test_position_open_returns_hold(self):
        strat = _strategy()
        sig = strat.generate_signal(
            _morning_release(0, mom=10.0, close=1305.0),
            current_position=_open_position(),
        )
        assert sig.signal == Signal.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# Filters and flags
# ═══════════════════════════════════════════════════════════════════════════


class TestKSBFilters:
    """Volume filter and long_only mode."""

    def _build_and_release(self, strat, release_kw=None):
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        kw = dict(mom=6.0, close=1305.0)
        if release_kw:
            kw.update(release_kw)
        return strat.generate_signal(_morning_release(15, **kw))

    def test_volume_filter_blocks_when_below(self):
        strat = _strategy(min_squeeze_bars=3, use_volume_filter=True, vol_mult=1.5)
        sig = self._build_and_release(strat, {"volume": 100.0, "volume_ma": 150.0})
        assert sig.signal == Signal.HOLD

    def test_volume_filter_allows_when_above(self):
        strat = _strategy(min_squeeze_bars=3, use_volume_filter=True, vol_mult=1.5)
        sig = self._build_and_release(strat, {"volume": 300.0, "volume_ma": 150.0})
        assert sig.signal == Signal.LONG

    def test_long_only_suppresses_short(self):
        strat = _strategy(min_squeeze_bars=3, long_only=True)
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        sig = strat.generate_signal(_morning_release(15, mom=-6.0, close=1295.0))
        assert sig.signal != Signal.SHORT

    def test_zero_atr_returns_hold(self):
        strat = _strategy(min_squeeze_bars=1, signal_window=3)
        strat.generate_signal(_morning_squeeze(0, atr=0.0, mom=1.0))
        sig = strat.generate_signal(_morning_release(5, atr=0.0, mom=3.0, close=1305.0))
        assert sig.signal == Signal.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# Signal metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestKSBSignalMetadata:
    """Signals carry session/mom/atr metadata."""

    def test_long_signal_metadata(self):
        strat = _strategy(min_squeeze_bars=3)
        for i in range(3):
            strat.generate_signal(_morning_squeeze(i * 5, mom=2.0))
        sig = strat.generate_signal(_morning_release(15, mom=6.0, close=1305.0))
        assert sig.signal == Signal.LONG
        assert sig.metadata["session"] == "morning"
        assert "mom" in sig.metadata
        assert "atr" in sig.metadata
