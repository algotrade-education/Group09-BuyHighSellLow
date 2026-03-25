"""Tests for StrategyBase and PositionSnapshot."""

from dataclasses import FrozenInstanceError
from datetime import datetime
from typing import Any

import pytest

from src.data.indicators.registry import IndicatorRegistry
from src.strategy.base import PositionSnapshot, StrategyBase
from src.strategy.signal import Signal, TradeSignal


class TestPositionSnapshot:
    """Test PositionSnapshot dataclass."""

    def test_flat_position(self):
        """Test creating a flat position."""
        pos = PositionSnapshot.flat()
        assert pos.is_flat
        assert not pos.is_long
        assert not pos.is_short
        assert pos.quantity == 0
        assert pos.entry_price == 0.0
        assert pos.stop_loss == 0.0
        assert pos.take_profit == 0.0

    def test_long_position(self):
        """Test creating a long position."""
        pos = PositionSnapshot(
            is_flat=False,
            is_long=True,
            is_short=False,
            quantity=10,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        assert not pos.is_flat
        assert pos.is_long
        assert not pos.is_short
        assert pos.quantity == 10

    def test_short_position(self):
        """Test creating a short position."""
        pos = PositionSnapshot(
            is_flat=False,
            is_long=False,
            is_short=True,
            quantity=5,
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
        )
        assert not pos.is_flat
        assert not pos.is_long
        assert pos.is_short
        assert pos.quantity == 5

    def test_invalid_multiple_flags(self):
        """Test that multiple direction flags raise ValueError."""
        with pytest.raises(ValueError, match="Exactly one of is_flat/is_long/is_short"):
            PositionSnapshot(
                is_flat=True,
                is_long=True,
                is_short=False,
                quantity=0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
            )

    def test_invalid_no_flags(self):
        """Test that no direction flags raise ValueError."""
        with pytest.raises(ValueError, match="Exactly one of is_flat/is_long/is_short"):
            PositionSnapshot(
                is_flat=False,
                is_long=False,
                is_short=False,
                quantity=0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
            )

    def test_invalid_flat_with_quantity(self):
        """Test that flat position with quantity > 0 raises ValueError."""
        with pytest.raises(ValueError, match="Flat position must have quantity=0"):
            PositionSnapshot(
                is_flat=True,
                is_long=False,
                is_short=False,
                quantity=10,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
            )

    def test_invalid_long_with_zero_quantity(self):
        """Test that long position with quantity=0 raises ValueError."""
        with pytest.raises(ValueError, match="Non-flat position must have quantity>0"):
            PositionSnapshot(
                is_flat=False,
                is_long=True,
                is_short=False,
                quantity=0,
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
            )

    def test_immutable(self):
        """Test that PositionSnapshot is immutable (frozen)."""
        pos = PositionSnapshot.flat()
        with pytest.raises(FrozenInstanceError):  # FrozenInstanceError
            pos.quantity = 10


class DummyStrategy(StrategyBase):
    """Dummy strategy for testing StrategyBase."""

    def generate_signal(
        self,
        bar: dict[str, Any],
        position: PositionSnapshot | None = None,
        is_warmup: bool = False,
    ) -> TradeSignal:
        return TradeSignal(Signal.HOLD)

    @classmethod
    def build_registry(cls, **params: Any) -> IndicatorRegistry:
        return IndicatorRegistry()


class TestStrategyBase:
    """Test StrategyBase abstract class."""

    def test_init(self):
        """Test strategy initialization."""
        strategy = DummyStrategy(name="TestStrategy")
        assert strategy.name == "TestStrategy"

    def test_repr(self):
        """Test strategy __repr__."""
        strategy = DummyStrategy(name="TestStrategy")
        assert repr(strategy) == "DummyStrategy(name='TestStrategy')"

    def test_reset(self):
        """Test default reset does nothing."""
        strategy = DummyStrategy(name="TestStrategy")
        strategy.reset()  # Should not raise

    def test_save_state(self):
        """Test default save_state returns empty dict."""
        strategy = DummyStrategy(name="TestStrategy")
        state = strategy.save_state()
        assert state == {}

    def test_load_state(self):
        """Test default load_state does nothing."""
        strategy = DummyStrategy(name="TestStrategy")
        strategy.load_state({"key": "value"})  # Should not raise

    # --- validate_bar tests ---

    def test_validate_bar_valid(self):
        """Test validate_bar with valid bar."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "volume": 1000,
        }
        assert StrategyBase.validate_bar(bar, ["datetime", "open", "high", "low", "close"])

    def test_validate_bar_missing_field(self):
        """Test validate_bar with missing field."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            # missing 'close'
        }
        assert not StrategyBase.validate_bar(bar, ["datetime", "open", "high", "low", "close"])

    def test_validate_bar_missing_field_raise(self):
        """Test validate_bar raises on missing field when raise_on_error=True."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
        }
        with pytest.raises(ValueError, match="Missing required field: close"):
            StrategyBase.validate_bar(bar, ["datetime", "open", "close"], raise_on_error=True)

    def test_validate_bar_none_value(self):
        """Test validate_bar with None value."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": None,
            "low": 99.0,
            "close": 103.0,
        }
        assert not StrategyBase.validate_bar(bar, ["datetime", "open", "high", "low", "close"])

    def test_validate_bar_none_value_raise(self):
        """Test validate_bar raises on None value when raise_on_error=True."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": None,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
        }
        with pytest.raises(ValueError, match="Field open is None"):
            StrategyBase.validate_bar(
                bar, ["datetime", "open", "high", "low", "close"], raise_on_error=True
            )

    def test_validate_bar_nan_value(self):
        """Test validate_bar with NaN value."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": float("nan"),
            "low": 99.0,
            "close": 103.0,
        }
        assert not StrategyBase.validate_bar(bar, ["datetime", "open", "high", "low", "close"])

    def test_validate_bar_nan_value_raise(self):
        """Test validate_bar raises on NaN value when raise_on_error=True."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": float("nan"),
            "low": 99.0,
            "close": 103.0,
        }
        with pytest.raises(ValueError, match="Field high is"):
            StrategyBase.validate_bar(
                bar, ["datetime", "open", "high", "low", "close"], raise_on_error=True
            )

    def test_validate_bar_inf_value(self):
        """Test validate_bar with Inf value."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": 105.0,
            "low": float("inf"),
            "close": 103.0,
        }
        assert not StrategyBase.validate_bar(bar, ["datetime", "open", "high", "low", "close"])

    def test_validate_bar_negative_value(self):
        """Test validate_bar with negative value."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": 105.0,
            "low": -99.0,
            "close": 103.0,
        }
        assert not StrategyBase.validate_bar(bar, ["datetime", "open", "high", "low", "close"])

    def test_validate_bar_negative_value_raise(self):
        """Test validate_bar raises on negative value when raise_on_error=True."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": 105.0,
            "low": -99.0,
            "close": 103.0,
        }
        with pytest.raises(ValueError, match="Field low"):
            StrategyBase.validate_bar(
                bar, ["datetime", "open", "high", "low", "close"], raise_on_error=True
            )

    def test_validate_bar_zero_value(self):
        """Test validate_bar with zero value."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": 105.0,
            "low": 0.0,
            "close": 103.0,
        }
        assert not StrategyBase.validate_bar(bar, ["datetime", "open", "high", "low", "close"])

    def test_validate_bar_non_numeric(self):
        """Test validate_bar with non-numeric value."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": "invalid",
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
        }
        assert not StrategyBase.validate_bar(bar, ["datetime", "open", "high", "low", "close"])

    def test_validate_bar_custom_fields(self):
        """Test validate_bar with custom fields."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "custom_field": "value",
            "atr": 2.5,
        }
        assert StrategyBase.validate_bar(bar, ["datetime", "custom_field", "atr"])
