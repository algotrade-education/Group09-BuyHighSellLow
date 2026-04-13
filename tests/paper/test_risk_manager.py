"""Unit tests for RiskManager (src/paper/risk_manager.py).

Tests verify risk management functionality including:
- Stop loss and take profit trigger detection for LONG and SHORT positions
- Conservative SL-before-TP ordering when both trigger in same bar
- Per-trade maximum loss enforcement
- Trailing stop updates based on ATR
- Daily loss limit enforcement

Test organization:
- get_exit_trigger: SL/TP detection and priority
- apply_trailing_stop: Trailing stop mechanics
- is_daily_loss_hit: Daily loss limit checks
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.paper.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_risk_manager(
    use_trailing_stop: bool = False,
    trailing_atr_multiplier: float = 2.0,
    max_daily_loss_fraction: float = 0.0,
    initial_capital: float = 100_000.0,
    max_loss_per_trade_fraction: float = 0.0,
) -> RiskManager:
    return RiskManager(
        use_trailing_stop=use_trailing_stop,
        trailing_atr_multiplier=trailing_atr_multiplier,
        max_daily_loss_fraction=max_daily_loss_fraction,
        initial_capital=initial_capital,
        max_loss_per_trade_fraction=max_loss_per_trade_fraction,
    )


def make_position(
    is_flat: bool = False,
    is_long: bool = True,
    is_short: bool = False,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    entry_price: float = 1300.0,
    quantity: int = 1,
    unrealized_pnl: float = 0.0,
) -> MagicMock:
    pos = MagicMock()
    pos.is_flat = is_flat
    pos.is_long = is_long
    pos.is_short = is_short
    pos.stop_loss = stop_loss
    pos.take_profit = take_profit
    pos.entry_price = entry_price
    pos.quantity = quantity
    pos.unrealized_pnl = unrealized_pnl

    # Wire check_stop_loss / check_take_profit to match the real Position logic
    # so get_exit_trigger (which delegates to these) behaves correctly.
    def _check_sl(price: float) -> bool:
        if stop_loss is None or is_flat:
            return False
        return price <= stop_loss if is_long else price >= stop_loss

    def _check_tp(price: float) -> bool:
        if take_profit is None or is_flat:
            return False
        return price >= take_profit if is_long else price <= take_profit

    pos.check_stop_loss = _check_sl
    pos.check_take_profit = _check_tp

    return pos


def make_bar(
    open_: float = 1300.0,
    high: float = 1320.0,
    low: float = 1280.0,
    close: float = 1310.0,
    atr_14: float = 10.0,
) -> dict:
    return {"open": open_, "high": high, "low": low, "close": close, "atr_14": atr_14}


# --- Stop Loss / Take Profit Detection ---
# Requirement 5.3-5.6: SL/TP trigger detection for LONG and SHORT positions


class TestGetExitTriggerFlat:
    """Test exit trigger detection when position is flat."""

    def test_flat_position_returns_none(self):
        """Verify get_exit_trigger returns None when position is flat.

        Test scenario:
        - Position is flat (no open position)
        - Bar has valid OHLC data

        Expected behavior:
        - Returns None (no exit trigger possible)
        """
        rm = make_risk_manager()
        pos = make_position(is_flat=True)
        bar = make_bar()
        assert rm.get_exit_trigger(pos, bar) is None


# --- LONG Position SL/TP Tests ---


class TestGetExitTriggerLong:
    """Test exit trigger detection for LONG positions."""

    def test_sl_triggered_when_low_touches_stop(self):
        """Verify LONG stop loss triggers when bar low touches SL level.

        Requirement 5.3: For LONG positions, stop loss SHALL trigger when
        bar low <= stop_loss price.

        Test scenario:
        - LONG position with SL at 1290.0
        - Bar low exactly at 1290.0

        Expected behavior:
        - Returns "Stop Loss"
        """
        rm = make_risk_manager()
        pos = make_position(is_long=True, stop_loss=1290.0)
        bar = make_bar(low=1290.0)  # exactly at SL
        assert rm.get_exit_trigger(pos, bar) == "Stop Loss"

    def test_sl_triggered_when_low_below_stop(self):
        rm = make_risk_manager()
        pos = make_position(is_long=True, stop_loss=1290.0)
        bar = make_bar(low=1285.0)
        assert rm.get_exit_trigger(pos, bar) == "Stop Loss"

    def test_sl_not_triggered_when_low_above_stop(self):
        rm = make_risk_manager()
        pos = make_position(is_long=True, stop_loss=1290.0)
        bar = make_bar(low=1295.0)
        result = rm.get_exit_trigger(pos, bar)
        assert result != "Stop Loss"

    def test_tp_triggered_when_high_touches_take_profit(self):
        """LONG: TP when bar high >= take_profit (Req 5.4)."""
        rm = make_risk_manager()
        pos = make_position(is_long=True, take_profit=1350.0)
        bar = make_bar(high=1350.0)
        assert rm.get_exit_trigger(pos, bar) == "Take Profit"

    def test_tp_triggered_when_high_above_take_profit(self):
        rm = make_risk_manager()
        pos = make_position(is_long=True, take_profit=1350.0)
        bar = make_bar(high=1360.0)
        assert rm.get_exit_trigger(pos, bar) == "Take Profit"

    def test_tp_not_triggered_when_high_below_take_profit(self):
        rm = make_risk_manager()
        pos = make_position(is_long=True, take_profit=1350.0)
        bar = make_bar(high=1340.0)
        result = rm.get_exit_trigger(pos, bar)
        assert result != "Take Profit"

    def test_sl_wins_over_tp_on_ambiguous_bar(self):
        """Verify SL takes priority over TP when both trigger in same bar.

        Requirement 9.1: When both SL and TP would trigger within the same bar,
        the risk manager SHALL return "Stop Loss" (conservative assumption).

        Rationale: Without tick-level data, we can't know which level was hit
        first. Assuming SL hit first is conservative and prevents overly
        optimistic backtest results.

        Test scenario:
        - LONG position with SL=1280.0, TP=1350.0
        - Bar with low=1275.0 (below SL) and high=1360.0 (above TP)

        Expected behavior:
        - Returns "Stop Loss" (not "Take Profit")
        """
        rm = make_risk_manager()
        pos = make_position(is_long=True, stop_loss=1280.0, take_profit=1350.0)
        bar = make_bar(low=1275.0, high=1360.0)  # both SL and TP hit
        assert rm.get_exit_trigger(pos, bar) == "Stop Loss"

    def test_no_trigger_when_no_sl_tp(self):
        rm = make_risk_manager()
        pos = make_position(is_long=True, stop_loss=None, take_profit=None)
        bar = make_bar(low=1200.0, high=1400.0)
        assert rm.get_exit_trigger(pos, bar) is None


# ---------------------------------------------------------------------------
# get_exit_trigger - SHORT SL/TP
# ---------------------------------------------------------------------------


class TestGetExitTriggerShort:
    def test_sl_triggered_when_high_touches_stop(self):
        """SHORT: SL when bar high >= stop_loss (Req 5.5)."""
        rm = make_risk_manager()
        pos = make_position(is_long=False, is_short=True, stop_loss=1320.0)
        bar = make_bar(high=1320.0)
        assert rm.get_exit_trigger(pos, bar) == "Stop Loss"

    def test_sl_triggered_when_high_above_stop(self):
        rm = make_risk_manager()
        pos = make_position(is_long=False, is_short=True, stop_loss=1320.0)
        bar = make_bar(high=1330.0)
        assert rm.get_exit_trigger(pos, bar) == "Stop Loss"

    def test_sl_not_triggered_when_high_below_stop(self):
        rm = make_risk_manager()
        pos = make_position(is_long=False, is_short=True, stop_loss=1320.0)
        bar = make_bar(high=1310.0)
        result = rm.get_exit_trigger(pos, bar)
        assert result != "Stop Loss"

    def test_tp_triggered_when_low_touches_take_profit(self):
        """SHORT: TP when bar low <= take_profit (Req 5.6)."""
        rm = make_risk_manager()
        pos = make_position(is_long=False, is_short=True, take_profit=1260.0)
        bar = make_bar(low=1260.0)
        assert rm.get_exit_trigger(pos, bar) == "Take Profit"

    def test_tp_triggered_when_low_below_take_profit(self):
        rm = make_risk_manager()
        pos = make_position(is_long=False, is_short=True, take_profit=1260.0)
        bar = make_bar(low=1250.0)
        assert rm.get_exit_trigger(pos, bar) == "Take Profit"

    def test_sl_wins_over_tp_short(self):
        """SHORT: SL wins when both would trigger."""
        rm = make_risk_manager()
        pos = make_position(is_long=False, is_short=True, stop_loss=1320.0, take_profit=1260.0)
        bar = make_bar(low=1250.0, high=1330.0)
        assert rm.get_exit_trigger(pos, bar) == "Stop Loss"


# ---------------------------------------------------------------------------
# get_exit_trigger - per-trade max loss (Req 9.4, 9.5)
# ---------------------------------------------------------------------------


class TestGetExitTriggerMaxTradeLoss:
    def test_triggers_when_loss_exceeds_threshold(self):
        """Returns 'Max Trade Loss' when unrealized loss > initial_capital * fraction."""
        rm = make_risk_manager(initial_capital=100_000.0, max_loss_per_trade_fraction=0.02)
        # threshold = 100_000 * 0.02 = 2_000
        pos = make_position(is_long=True, unrealized_pnl=-2_001.0)
        bar = make_bar()
        assert rm.get_exit_trigger(pos, bar) == "Max Trade Loss"

    def test_does_not_trigger_when_loss_below_threshold(self):
        rm = make_risk_manager(initial_capital=100_000.0, max_loss_per_trade_fraction=0.02)
        pos = make_position(is_long=True, unrealized_pnl=-1_999.0)
        bar = make_bar()
        assert rm.get_exit_trigger(pos, bar) != "Max Trade Loss"

    def test_never_triggers_when_pct_is_zero(self):
        """When max_loss_per_trade_fraction == 0.0, Max Trade Loss must never be returned."""
        rm = make_risk_manager(initial_capital=100_000.0, max_loss_per_trade_fraction=0.0)
        pos = make_position(is_long=True, unrealized_pnl=-999_999.0)
        bar = make_bar()
        assert rm.get_exit_trigger(pos, bar) != "Max Trade Loss"

    def test_sl_takes_priority_over_max_trade_loss(self):
        """SL/TP checks happen before per-trade loss check."""
        rm = make_risk_manager(initial_capital=100_000.0, max_loss_per_trade_fraction=0.01)
        pos = make_position(is_long=True, stop_loss=1290.0, unrealized_pnl=-5_000.0)
        bar = make_bar(low=1285.0)
        assert rm.get_exit_trigger(pos, bar) == "Stop Loss"


# ---------------------------------------------------------------------------
# apply_trailing_stop
# ---------------------------------------------------------------------------


class TestApplyTrailingStop:
    def test_no_update_when_trailing_stop_disabled(self):
        rm = make_risk_manager(use_trailing_stop=False)
        pos = make_position(is_long=True, stop_loss=1280.0)
        bar = make_bar(close=1350.0, atr_14=10.0)
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss == 1280.0

    def test_no_update_when_position_flat(self):
        rm = make_risk_manager(use_trailing_stop=True)
        pos = make_position(is_flat=True, stop_loss=1280.0)
        bar = make_bar(close=1350.0, atr_14=10.0)
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss == 1280.0

    def test_no_update_when_stop_loss_is_none(self):
        rm = make_risk_manager(use_trailing_stop=True)
        pos = make_position(is_long=True, stop_loss=None)
        bar = make_bar(close=1350.0, atr_14=10.0)
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss is None

    def test_long_trailing_stop_moves_up(self):
        """LONG: new SL = close - atr * multiplier, only if higher than current SL."""
        rm = make_risk_manager(use_trailing_stop=True, trailing_atr_multiplier=2.0)
        pos = make_position(is_long=True, stop_loss=1280.0)
        bar = make_bar(close=1350.0, atr_14=10.0)
        # new_sl = 1350 - 2 * 10 = 1330 > 1280 → update
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss == pytest.approx(1330.0)

    def test_long_trailing_stop_does_not_move_down(self):
        """LONG: SL should not move down (only ratchets up)."""
        rm = make_risk_manager(use_trailing_stop=True, trailing_atr_multiplier=2.0)
        pos = make_position(is_long=True, stop_loss=1320.0)
        bar = make_bar(close=1310.0, atr_14=10.0)
        # new_sl = 1310 - 20 = 1290 < 1320 → no update
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss == 1320.0

    def test_short_trailing_stop_moves_down(self):
        """SHORT: new SL = close + atr * multiplier, only if lower than current SL."""
        rm = make_risk_manager(use_trailing_stop=True, trailing_atr_multiplier=2.0)
        pos = make_position(is_long=False, is_short=True, stop_loss=1350.0)
        bar = make_bar(close=1280.0, atr_14=10.0)
        # new_sl = 1280 + 20 = 1300 < 1350 → update
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss == pytest.approx(1300.0)

    def test_short_trailing_stop_does_not_move_up(self):
        """SHORT: SL should not move up."""
        rm = make_risk_manager(use_trailing_stop=True, trailing_atr_multiplier=2.0)
        pos = make_position(is_long=False, is_short=True, stop_loss=1300.0)
        bar = make_bar(close=1320.0, atr_14=10.0)
        # new_sl = 1320 + 20 = 1340 > 1300 → no update
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss == 1300.0

    def test_no_update_when_atr_is_zero(self):
        """No trailing stop update when ATR is 0 or missing."""
        rm = make_risk_manager(use_trailing_stop=True, trailing_atr_multiplier=2.0)
        pos = make_position(is_long=True, stop_loss=1280.0)
        bar = make_bar(close=1350.0, atr_14=0.0)
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss == 1280.0

    def test_no_update_when_bar_has_no_atr_key(self):
        """No trailing stop update when the bar dict has no 'atr_*' key at all."""
        rm = make_risk_manager(use_trailing_stop=True, trailing_atr_multiplier=2.0)
        pos = make_position(is_long=True, stop_loss=1280.0)
        bar = {"open": 1350.0, "high": 1360.0, "low": 1340.0, "close": 1350.0}  # no atr_ key
        rm.apply_trailing_stop(pos, bar)
        assert pos.stop_loss == 1280.0


# ---------------------------------------------------------------------------
# is_daily_loss_hit (Req 5.10)
# ---------------------------------------------------------------------------


class TestIsDailyLossHit:
    def test_returns_false_when_disabled(self):
        """max_daily_loss_fraction == 0.0 → always False."""
        rm = make_risk_manager(max_daily_loss_fraction=0.0, initial_capital=100_000.0)
        assert not rm.is_daily_loss_hit(-999_999.0)

    def test_returns_false_when_pnl_positive(self):
        rm = make_risk_manager(max_daily_loss_fraction=0.02, initial_capital=100_000.0)
        assert not rm.is_daily_loss_hit(500.0)

    def test_returns_false_when_loss_below_threshold(self):
        """threshold = 100_000 * 0.02 = 2_000"""
        rm = make_risk_manager(max_daily_loss_fraction=0.02, initial_capital=100_000.0)
        assert not rm.is_daily_loss_hit(-1_999.0)

    def test_returns_true_when_loss_equals_threshold(self):
        rm = make_risk_manager(max_daily_loss_fraction=0.02, initial_capital=100_000.0)
        assert rm.is_daily_loss_hit(-2_000.0)

    def test_returns_true_when_loss_exceeds_threshold(self):
        rm = make_risk_manager(max_daily_loss_fraction=0.02, initial_capital=100_000.0)
        assert rm.is_daily_loss_hit(-2_500.0)

    def test_threshold_scales_with_capital(self):
        """Threshold is proportional to initial_capital."""
        rm = make_risk_manager(max_daily_loss_fraction=0.01, initial_capital=500_000_000.0)
        # threshold = 500_000_000 * 0.01 = 5_000_000
        assert not rm.is_daily_loss_hit(-4_999_999.0)
        assert rm.is_daily_loss_hit(-5_000_000.0)
