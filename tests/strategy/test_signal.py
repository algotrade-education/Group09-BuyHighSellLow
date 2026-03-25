"""Tests for TradeSignal and Signal enum."""

import pytest

from src.strategy.signal import Signal, TradeSignal


class TestSignal:
    """Test Signal enum."""

    def test_signal_values(self):
        """Test Signal enum has expected values."""
        assert Signal.LONG == "long"
        assert Signal.SHORT == "short"
        assert Signal.HOLD == "hold"
        assert Signal.EXIT == "exit"

    def test_signal_membership(self):
        """Test Signal enum membership."""
        assert Signal("long") is Signal.LONG
        assert Signal("short") is Signal.SHORT
        assert Signal("hold") is Signal.HOLD
        assert Signal("exit") is Signal.EXIT
        with pytest.raises(ValueError):
            Signal("invalid")


class TestTradeSignal:
    """Test TradeSignal dataclass."""

    def test_default_signal(self):
        """Test default TradeSignal is HOLD."""
        signal = TradeSignal()
        assert signal.signal == Signal.HOLD
        assert signal.entry_price == 0.0
        assert signal.stop_loss == 0.0
        assert signal.take_profit == 0.0
        assert signal.ord_type == "LIMIT"
        assert signal.reason == ""
        assert signal.metadata == {}

    def test_long_signal(self):
        """Test creating a LONG signal."""
        signal = TradeSignal(
            signal=Signal.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            ord_type="LIMIT",
            reason="Test long",
        )
        assert signal.signal == Signal.LONG
        assert signal.entry_price == 100.0
        assert signal.stop_loss == 95.0
        assert signal.take_profit == 110.0
        assert signal.ord_type == "LIMIT"
        assert signal.reason == "Test long"

    def test_short_signal(self):
        """Test creating a SHORT signal."""
        signal = TradeSignal(
            signal=Signal.SHORT,
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            ord_type="MARKET",
            reason="Test short",
        )
        assert signal.signal == Signal.SHORT
        assert signal.entry_price == 100.0
        assert signal.stop_loss == 105.0
        assert signal.take_profit == 90.0
        assert signal.ord_type == "MARKET"

    def test_invalid_ord_type(self):
        """Test that invalid ord_type raises ValueError."""
        with pytest.raises(ValueError, match="ord_type must be 'LIMIT' or 'MARKET'"):
            TradeSignal(signal=Signal.LONG, ord_type="INVALID")

    def test_negative_entry_price(self):
        """Test that negative entry_price raises ValueError."""
        with pytest.raises(ValueError, match="entry_price must be >= 0"):
            TradeSignal(signal=Signal.LONG, entry_price=-10.0)

    def test_negative_stop_loss(self):
        """Test that negative stop_loss raises ValueError."""
        with pytest.raises(ValueError, match="stop_loss must be >= 0"):
            TradeSignal(signal=Signal.LONG, stop_loss=-5.0)

    def test_negative_take_profit(self):
        """Test that negative take_profit raises ValueError."""
        with pytest.raises(ValueError, match="take_profit must be >= 0"):
            TradeSignal(signal=Signal.LONG, take_profit=-15.0)

    def test_metadata(self):
        """Test signal with metadata."""
        metadata = {"atr": 2.5, "range_size": 10.0}
        signal = TradeSignal(
            signal=Signal.LONG,
            entry_price=100.0,
            metadata=metadata,
        )
        assert signal.metadata == metadata
        assert signal.metadata["atr"] == 2.5

    # --- Property tests ---

    def test_is_entry_property(self):
        """Test is_entry property."""
        assert TradeSignal(signal=Signal.LONG).is_entry
        assert TradeSignal(signal=Signal.SHORT).is_entry
        assert not TradeSignal(signal=Signal.HOLD).is_entry
        assert not TradeSignal(signal=Signal.EXIT).is_entry

    def test_is_long_property(self):
        """Test is_long property."""
        assert TradeSignal(signal=Signal.LONG).is_long
        assert not TradeSignal(signal=Signal.SHORT).is_long
        assert not TradeSignal(signal=Signal.HOLD).is_long

    def test_is_short_property(self):
        """Test is_short property."""
        assert TradeSignal(signal=Signal.SHORT).is_short
        assert not TradeSignal(signal=Signal.LONG).is_short
        assert not TradeSignal(signal=Signal.HOLD).is_short

    def test_is_hold_property(self):
        """Test is_hold property."""
        assert TradeSignal(signal=Signal.HOLD).is_hold
        assert not TradeSignal(signal=Signal.LONG).is_hold
        assert not TradeSignal(signal=Signal.SHORT).is_hold

    def test_is_exit_property(self):
        """Test is_exit property."""
        assert TradeSignal(signal=Signal.EXIT).is_exit
        assert not TradeSignal(signal=Signal.LONG).is_exit
        assert not TradeSignal(signal=Signal.HOLD).is_exit

    # --- Repr tests ---

    def test_repr_hold(self):
        """Test __repr__ for HOLD signal."""
        signal = TradeSignal(signal=Signal.HOLD)
        assert repr(signal) == "TradeSignal(HOLD)"

    def test_repr_hold_with_reason(self):
        """Test __repr__ for HOLD signal with reason."""
        signal = TradeSignal(signal=Signal.HOLD, reason="Waiting for setup")
        assert repr(signal) == "TradeSignal(HOLD (Waiting for setup))"

    def test_repr_long(self):
        """Test __repr__ for LONG signal."""
        signal = TradeSignal(
            signal=Signal.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        assert "TradeSignal(LONG" in repr(signal)
        assert "entry=100.0" in repr(signal)
        assert "sl=95.00" in repr(signal)
        assert "tp=110.00" in repr(signal)

    def test_repr_short(self):
        """Test __repr__ for SHORT signal."""
        signal = TradeSignal(
            signal=Signal.SHORT,
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
        )
        assert "TradeSignal(SHORT" in repr(signal)
        assert "entry=100.0" in repr(signal)
