"""
Tests for RiskManager isolated logic: SL/TP triggers, trailing stops, and daily loss limits.
"""

from typing import Any, Dict
from unittest.mock import MagicMock
import pytest

from src.paper.risk_manager import RiskManager

class MockPosition:
    def __init__(self, side: str, stop_loss: float = None, take_profit: float = None):
        self.side_name = side
        self.is_long = side == "LONG"
        self.is_short = side == "SHORT"
        self.is_flat = side == "FLAT"
        self.stop_loss = stop_loss
        self.take_profit = take_profit

@pytest.fixture()
def rm_no_trail():
    return RiskManager(
        use_trailing_stop=False,
        trailing_atr_multiplier=2.0,
        max_daily_loss_pct=1.0,  # 1%
        initial_capital=1_000_000.0,
    )

@pytest.fixture()
def rm_with_trail():
    return RiskManager(
        use_trailing_stop=True,
        trailing_atr_multiplier=2.0,
        max_daily_loss_pct=1.0,
        initial_capital=1_000_000.0,
    )

class TestRiskManagerDailyLoss:
    def test_daily_loss_not_hit(self, rm_no_trail):
        # 1% of 1M = 10,000 loss limit
        assert not rm_no_trail.is_daily_loss_hit(-5000.0)
        assert not rm_no_trail.is_daily_loss_hit(5000.0)
        assert not rm_no_trail.is_daily_loss_hit(0.0)

    def test_daily_loss_hit(self, rm_no_trail):
        assert rm_no_trail.is_daily_loss_hit(-10000.0)
        assert rm_no_trail.is_daily_loss_hit(-15000.0)

    def test_daily_loss_disabled_if_zero_pct(self):
        rm = RiskManager(False, max_daily_loss_pct=0.0, initial_capital=1_000_000.0)
        assert not rm.is_daily_loss_hit(-500000.0)

class TestRiskManagerExitTriggers:
    def test_flat_position_returns_none(self, rm_no_trail):
        pos = MockPosition("FLAT")
        assert rm_no_trail.get_exit_trigger(pos, {"low": 100, "high": 200}) is None

    def test_long_sl_hit_by_low(self, rm_no_trail):
        pos = MockPosition("LONG", stop_loss=100.0, take_profit=200.0)
        assert rm_no_trail.get_exit_trigger(pos, {"low": 99.0, "high": 150.0}) == "Stop Loss"
        
    def test_long_tp_hit_by_high(self, rm_no_trail):
        pos = MockPosition("LONG", stop_loss=100.0, take_profit=200.0)
        assert rm_no_trail.get_exit_trigger(pos, {"low": 110.0, "high": 205.0}) == "Take Profit"

    def test_long_both_hit_prefers_sl_pessimistically(self, rm_no_trail):
        pos = MockPosition("LONG", stop_loss=100.0, take_profit=200.0)
        assert rm_no_trail.get_exit_trigger(pos, {"low": 90.0, "high": 210.0}) == "Stop Loss"

    def test_short_sl_hit_by_high(self, rm_no_trail):
        pos = MockPosition("SHORT", stop_loss=200.0, take_profit=100.0)
        assert rm_no_trail.get_exit_trigger(pos, {"low": 150.0, "high": 205.0}) == "Stop Loss"

    def test_short_tp_hit_by_low(self, rm_no_trail):
        pos = MockPosition("SHORT", stop_loss=200.0, take_profit=100.0)
        assert rm_no_trail.get_exit_trigger(pos, {"low": 95.0, "high": 150.0}) == "Take Profit"

class TestRiskManagerTrailingStop:
    def test_trailing_stop_disabled(self, rm_no_trail):
        pos = MockPosition("LONG", stop_loss=100.0)
        rm_no_trail.apply_trailing_stop(pos, {"close": 150.0, "atr_14": 10.0})
        assert pos.stop_loss == 100.0 # No change

    def test_long_trailing_stop_moves_up_only(self, rm_with_trail):
        pos = MockPosition("LONG", stop_loss=100.0)
        # atr_distance = 2 * 10 = 20
        # new target sl = 150 - 20 = 130 > 100
        rm_with_trail.apply_trailing_stop(pos, {"close": 150.0, "atr_14": 10.0})
        assert pos.stop_loss == 130.0

        # price drops to 120 -> target sl = 120 - 20 = 100 which is < 130! 
        # So should NOT move!
        rm_with_trail.apply_trailing_stop(pos, {"close": 120.0, "atr_14": 10.0})
        assert pos.stop_loss == 130.0

    def test_short_trailing_stop_moves_down_only(self, rm_with_trail):
        pos = MockPosition("SHORT", stop_loss=200.0)
        # atr_distance = 2 * 10 = 20
        # new target sl = 150 + 20 = 170 < 200
        rm_with_trail.apply_trailing_stop(pos, {"close": 150.0, "atr_14": 10.0})
        assert pos.stop_loss == 170.0

        # price pops to 180 -> target sl = 180 + 20 = 200 which is > 170!
        # So should NOT move!
        rm_with_trail.apply_trailing_stop(pos, {"close": 180.0, "atr_14": 10.0})
        assert pos.stop_loss == 170.0

    def test_trailing_stop_robust_to_no_atr(self, rm_with_trail):
        pos = MockPosition("LONG", stop_loss=100.0)
        rm_with_trail.apply_trailing_stop(pos, {"close": 150.0})
        assert pos.stop_loss == 100.0 # No change

        rm_with_trail.apply_trailing_stop(pos, {"close": 150.0, "atr_14": 0.0})
        assert pos.stop_loss == 100.0 # No change

        rm_with_trail.apply_trailing_stop(pos, {"close": 150.0, "atr_14": None})
        assert pos.stop_loss == 100.0 # No change
