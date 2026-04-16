"""Tests for SMA indicator."""

import pytest

from src.data.indicators.sma import SMA


class TestSMA:
    """Test SMA indicator."""

    def test_init(self):
        """Test SMA initialization."""
        sma = SMA(period=5, source_col="close")
        assert sma._period == 5
        assert sma._source_col == "close"
        assert not sma.is_ready
        assert sma.value is None

    def test_init_invalid_period(self):
        """Test SMA with invalid period."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            SMA(period=0)

    def test_update_until_ready(self):
        """Test SMA updates until ready."""
        sma = SMA(period=3, source_col="close")

        # First bar
        result = sma.update(close=10.0)
        assert result is None
        assert not sma.is_ready

        # Second bar
        result = sma.update(close=20.0)
        assert result is None
        assert not sma.is_ready

        # Third bar - now ready
        result = sma.update(close=30.0)
        assert result == 20.0  # (10 + 20 + 30) / 3
        assert sma.is_ready

    def test_update_rolling(self):
        """Test SMA rolling window."""
        sma = SMA(period=3, source_col="close")

        # Fill initial window
        sma.update(close=10.0)
        sma.update(close=20.0)
        sma.update(close=30.0)
        assert sma.value == 20.0

        # Add fourth bar - should drop first
        sma.update(close=40.0)
        assert sma.value == 30.0  # (20 + 30 + 40) / 3

        # Add fifth bar
        sma.update(close=50.0)
        assert sma.value == 40.0  # (30 + 40 + 50) / 3

    def test_update_missing_column(self):
        """Test SMA with missing source column."""
        sma = SMA(period=3, source_col="close")

        sma.update(close=10.0)
        sma.update(close=20.0)
        sma.update(close=30.0)

        # Missing column - should return current value
        result = sma.update(high=100.0)
        assert result == 20.0  # Unchanged

    def test_update_invalid_value(self):
        """Test SMA with invalid value."""
        sma = SMA(period=3, source_col="close")

        sma.update(close=10.0)
        sma.update(close=20.0)
        sma.update(close=30.0)

        # Invalid value - should return current value
        result = sma.update(close="invalid")
        assert result == 20.0  # Unchanged

    def test_reset(self):
        """Test SMA reset."""
        sma = SMA(period=3, source_col="close")

        sma.update(close=10.0)
        sma.update(close=20.0)
        sma.update(close=30.0)
        assert sma.is_ready

        sma.reset()
        assert not sma.is_ready
        assert sma.value is None
        assert len(sma._values) == 0

    def test_state_serialization(self):
        """Test SMA state save/load."""
        sma1 = SMA(period=3, source_col="close")
        sma1.update(close=10.0)
        sma1.update(close=20.0)
        sma1.update(close=30.0)

        # Save state
        state = sma1.save_state()
        assert state["class"] == "SMA"
        assert state["period"] == 3
        assert state["source_col"] == "close"
        assert state["values"] == [10.0, 20.0, 30.0]

        # Load into new instance
        sma2 = SMA(period=5, source_col="high")  # Different initial values
        sma2.load_state(state)

        assert sma2._period == 3
        assert sma2._source_col == "close"
        assert sma2.value == 20.0
        assert sma2.is_ready

    def test_state_class_mismatch(self):
        """Test SMA state with wrong class."""
        sma = SMA(period=3, source_col="close")
        state = {
            "class": "WrongClass",
            "period": 3,
            "source_col": "close",
            "values": [],
            "sum": 0.0,
            "count": 0,
            "current_value": None,
        }

        with pytest.raises(ValueError, match="State class"):
            sma.load_state(state)

    def test_custom_source_column(self):
        """Test SMA with custom source column."""
        sma = SMA(period=3, source_col="atr_14")

        sma.update(atr_14=5.0)
        sma.update(atr_14=10.0)
        result = sma.update(atr_14=15.0)

        assert result == 10.0  # (5 + 10 + 15) / 3
        assert sma.is_ready
