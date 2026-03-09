"""
Unit tests for the VWAPBandReversion strategy.

Tests the signal generation logic end-to-end using synthetic bar dicts.

VN30 session schedule:
    Morning   09:00 - 11:30
    Afternoon 13:00 - 14:30

Strategy rules tested:
    - Price below lower band → LONG (fade up to VWAP)
    - Price above upper band → SHORT (fade down to VWAP)
    - Price within bands → HOLD
    - Session warmup guard
    - TP = VWAP, SL = ATR-based
    - Min TP distance guard (skip if VWAP too close)
    - Cooldown after exit
    - Session / day reset
    - long_only mode suppresses SHORT
    - VWAP slope filter
    - Volume filter
    - Position already open → HOLD
    - Out-of-session bars → HOLD

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_strategy_vwap.py -v
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.strategy.VWAP import VWAPBandReversion
from src.strategy.base import Signal


# ── Helpers ────────────────────────────────────────────────────────────────

VWAP = 1300.0
VWAP_STD = 5.0  # so 2σ band = 1300 ± 10


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
    vwap: float = VWAP,
    vwap_std: float = VWAP_STD,
) -> Dict[str, Any]:
    return {
        "datetime": dt,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        f"atr_{atr_period}": atr,
        "volume": volume,
        "volume_ma_20": volume_ma,
        "vwap": vwap,
        "vwap_std": vwap_std,
    }


def _morning(minute: int, **kw) -> Dict[str, Any]:
    dt = datetime(2025, 1, 6, 9, minute, 0)
    return _bar(dt, **kw)


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


def _strategy(**kwargs) -> VWAPBandReversion:
    defaults = dict(
        entry_band=2.0,
        atr_period=14,
        atr_sl_mult=1.5,
        min_tp_atr=0.3,
        cooldown_bars=2,
        session_warmup=3,
        long_only=False,
        use_slope_filter=False,
        slope_period=5,
        use_volume_filter=False,
        vol_mult=1.5,
        vol_ma_period=20,
    )
    defaults.update(kwargs)
    return VWAPBandReversion(**defaults)


def _warm_up(strat: VWAPBandReversion, n: int = 3) -> None:
    """Feed n warmup bars to pass the session_warmup guard."""
    for i in range(n):
        strat.generate_signal(_morning(i))


# ═══════════════════════════════════════════════════════════════════════════
# Entry signals
# ═══════════════════════════════════════════════════════════════════════════


class TestVWAPEntrySignals:
    """Price outside VWAP bands triggers mean-reversion entries."""

    def test_long_when_below_lower_band(self):
        """close < vwap - 2*std → LONG."""
        strat = _strategy(session_warmup=3, entry_band=2.0)
        _warm_up(strat)
        # lower band = 1300 - 2*5 = 1290; close=1288 is below
        sig = strat.generate_signal(_morning(15, close=1288.0))
        assert sig.signal == Signal.LONG

    def test_short_when_above_upper_band(self):
        """close > vwap + 2*std → SHORT."""
        strat = _strategy(session_warmup=3, entry_band=2.0)
        _warm_up(strat)
        # upper band = 1300 + 2*5 = 1310; close=1312 is above
        sig = strat.generate_signal(_morning(15, close=1312.0))
        assert sig.signal == Signal.SHORT

    def test_hold_when_within_bands(self):
        """close between bands → HOLD."""
        strat = _strategy(session_warmup=3, entry_band=2.0)
        _warm_up(strat)
        sig = strat.generate_signal(_morning(15, close=1300.0))
        assert sig.signal == Signal.HOLD

    def test_long_tp_is_vwap(self):
        """LONG take profit should be at the current VWAP."""
        strat = _strategy(session_warmup=3, entry_band=2.0)
        _warm_up(strat)
        sig = strat.generate_signal(_morning(15, close=1288.0))
        assert sig.signal == Signal.LONG
        assert sig.take_profit == pytest.approx(VWAP)

    def test_long_sl_is_atr_based(self):
        strat = _strategy(session_warmup=3, atr_sl_mult=1.5)
        _warm_up(strat)
        close = 1288.0
        atr = 10.0
        sig = strat.generate_signal(_morning(15, close=close, atr=atr))
        assert sig.signal == Signal.LONG
        assert sig.stop_loss == pytest.approx(close - 1.5 * atr)

    def test_short_tp_is_vwap(self):
        strat = _strategy(session_warmup=3, entry_band=2.0)
        _warm_up(strat)
        sig = strat.generate_signal(_morning(15, close=1312.0))
        assert sig.signal == Signal.SHORT
        assert sig.take_profit == pytest.approx(VWAP)

    def test_short_sl_is_atr_based(self):
        strat = _strategy(session_warmup=3, atr_sl_mult=1.5)
        _warm_up(strat)
        close = 1312.0
        atr = 10.0
        sig = strat.generate_signal(_morning(15, close=close, atr=atr))
        assert sig.signal == Signal.SHORT
        assert sig.stop_loss == pytest.approx(close + 1.5 * atr)


# ═══════════════════════════════════════════════════════════════════════════
# Min TP distance guard
# ═══════════════════════════════════════════════════════════════════════════


class TestVWAPMinTP:
    """Skip trades where VWAP is too close to entry."""

    def test_skip_long_when_vwap_too_close(self):
        """If TP distance < min_tp_atr * ATR, skip."""
        strat = _strategy(session_warmup=3, min_tp_atr=1.5)
        _warm_up(strat)
        # close=1289, vwap=1300, TP distance=11, ATR=10, min_tp_atr*ATR=15 → too close
        sig = strat.generate_signal(_morning(15, close=1289.0, atr=10.0))
        assert sig.signal == Signal.HOLD

    def test_allow_long_when_tp_far_enough(self):
        strat = _strategy(session_warmup=3, min_tp_atr=0.3)
        _warm_up(strat)
        # TP distance=12, min_tp_atr*ATR=3 → fine
        sig = strat.generate_signal(_morning(15, close=1288.0, atr=10.0))
        assert sig.signal == Signal.LONG


# ═══════════════════════════════════════════════════════════════════════════
# Session warmup
# ═══════════════════════════════════════════════════════════════════════════


class TestVWAPSessionWarmup:
    """No signals during the first session_warmup bars."""

    def test_warmup_blocks_signal(self):
        strat = _strategy(session_warmup=5)
        sig = strat.generate_signal(_morning(0, close=1288.0))
        assert sig.signal == Signal.HOLD

    def test_after_warmup_allows_signal(self):
        strat = _strategy(session_warmup=3)
        _warm_up(strat, 3)
        sig = strat.generate_signal(_morning(15, close=1288.0))
        assert sig.signal == Signal.LONG


# ═══════════════════════════════════════════════════════════════════════════
# Cooldown
# ═══════════════════════════════════════════════════════════════════════════


class TestVWAPCooldown:
    """After an exit, wait cooldown_bars before next entry."""

    def test_cooldown_blocks_entry(self):
        strat = _strategy(session_warmup=3, cooldown_bars=2)
        _warm_up(strat)

        # Simulate exit transition
        strat._was_flat = False
        strat._bars_since_exit = 9999

        sig = strat.generate_signal(
            _morning(15, close=1288.0), current_position=_flat_position()
        )
        assert sig.signal == Signal.HOLD  # cooldown bar 0

    def test_cooldown_expires_allows_entry(self):
        strat = _strategy(session_warmup=3, cooldown_bars=1)
        _warm_up(strat)

        strat._was_flat = False
        strat._bars_since_exit = 9999

        # Cooldown tick
        strat.generate_signal(
            _morning(15, close=1300.0), current_position=_flat_position()
        )
        # Now past cooldown
        sig = strat.generate_signal(
            _morning(20, close=1288.0), current_position=_flat_position()
        )
        assert sig.signal == Signal.LONG


# ═══════════════════════════════════════════════════════════════════════════
# Session management
# ═══════════════════════════════════════════════════════════════════════════


class TestVWAPSessionManagement:

    def test_new_day_resets_warmup(self):
        strat = _strategy(session_warmup=5)
        _warm_up(strat, 5)
        # Now switch to day 2 - warmup should reset
        d2 = datetime(2025, 1, 7, 9, 0, 0)
        sig = strat.generate_signal(_bar(d2, close=1288.0))
        assert sig.signal == Signal.HOLD  # warmup not met yet

    def test_new_session_resets_warmup(self):
        strat = _strategy(session_warmup=5)
        _warm_up(strat, 5)
        # Switch to afternoon - warmup resets
        sig = strat.generate_signal(_afternoon(0, close=1288.0))
        assert sig.signal == Signal.HOLD

    def test_out_of_session_returns_hold(self):
        strat = _strategy()
        dt = datetime(2025, 1, 6, 12, 0, 0)
        sig = strat.generate_signal(_bar(dt, close=1288.0))
        assert sig.signal == Signal.HOLD

    def test_position_open_returns_hold(self):
        strat = _strategy(session_warmup=3)
        _warm_up(strat)
        sig = strat.generate_signal(
            _morning(15, close=1288.0), current_position=_open_position()
        )
        assert sig.signal == Signal.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# Filters and flags
# ═══════════════════════════════════════════════════════════════════════════


class TestVWAPFilters:

    def test_long_only_suppresses_short(self):
        strat = _strategy(session_warmup=3, long_only=True)
        _warm_up(strat)
        sig = strat.generate_signal(_morning(15, close=1312.0))
        assert sig.signal != Signal.SHORT

    def test_volume_filter_blocks_when_below(self):
        strat = _strategy(session_warmup=3, use_volume_filter=True, vol_mult=1.5)
        _warm_up(strat)
        sig = strat.generate_signal(
            _morning(15, close=1288.0, volume=100.0, volume_ma=150.0)
        )
        assert sig.signal == Signal.HOLD

    def test_volume_filter_allows_when_above(self):
        strat = _strategy(session_warmup=3, use_volume_filter=True, vol_mult=1.5)
        _warm_up(strat)
        sig = strat.generate_signal(
            _morning(15, close=1288.0, volume=300.0, volume_ma=150.0)
        )
        assert sig.signal == Signal.LONG

    def test_slope_filter_blocks_long_when_slope_bearish(self):
        strat = _strategy(session_warmup=3, use_slope_filter=True, slope_period=2)
        _warm_up(strat)
        # Inject declining VWAP history
        strat._vwap_history = [1310.0, 1305.0, 1300.0]
        sig = strat.generate_signal(_morning(15, close=1288.0))
        assert sig.signal == Signal.HOLD

    def test_slope_filter_allows_long_when_slope_neutral(self):
        strat = _strategy(session_warmup=3, use_slope_filter=True, slope_period=2)
        _warm_up(strat)
        # Flat VWAP history
        strat._vwap_history = [1300.0, 1300.0, 1300.0]
        sig = strat.generate_signal(_morning(15, close=1288.0))
        assert sig.signal == Signal.LONG

    def test_zero_atr_returns_hold(self):
        strat = _strategy(session_warmup=3)
        _warm_up(strat)
        sig = strat.generate_signal(_morning(15, close=1288.0, atr=0.0))
        assert sig.signal == Signal.HOLD

    def test_zero_vwap_std_returns_hold(self):
        strat = _strategy(session_warmup=3)
        _warm_up(strat)
        sig = strat.generate_signal(_morning(15, close=1288.0, vwap_std=0.0))
        assert sig.signal == Signal.HOLD


# ═══════════════════════════════════════════════════════════════════════════
# Signal metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestVWAPSignalMetadata:

    def test_long_signal_metadata(self):
        strat = _strategy(session_warmup=3)
        _warm_up(strat)
        sig = strat.generate_signal(_morning(15, close=1288.0))
        assert sig.signal == Signal.LONG
        assert sig.metadata["session"] == "morning"
        assert "vwap" in sig.metadata
        assert "atr" in sig.metadata
