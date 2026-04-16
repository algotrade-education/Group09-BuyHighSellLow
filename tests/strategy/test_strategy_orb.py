"""Tests for ORBStrategy."""

from datetime import datetime

import pytest

from config.schemas.orb import ORBConfig
from src.strategy.base import PositionSnapshot
from src.strategy.orb import ORBStrategy


@pytest.fixture
def orb_config():
    """Create a basic ORB config for testing."""
    return ORBConfig(
        name="Test ORB",
        strategy={
            "resample_freq": "5min",
            "orb_minutes": 20,
            "atr_period": 14,
            "atr_tp_multiplier": 2.0,
            "atr_sl_multiplier": 1.5,
            "breakout_buffer": 0.1,
            "use_range_sl": True,
            "min_range_atr": 0.5,
            "max_range_atr": 3.0,
            "long_only": False,
            "use_volume_filter": False,
            "use_adx_filter": False,
            "require_close_confirmation": False,
            "max_trades_per_session": 2,
        },
        risk={
            "min_position_size": 1,
            "max_position_size": 10,
            "risk_per_trade_pct": 1.0,
            "max_daily_loss": 2.0,
            "entry_ord_type": "LIMIT",
        },
    )


@pytest.fixture
def strategy(orb_config):
    """Create ORBStrategy instance."""
    return ORBStrategy(orb_config)


class TestORBStrategyInit:
    """Test ORBStrategy initialization."""

    def test_init(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "ORBStrategy"
        assert strategy._p.orb_minutes == 20
        assert strategy._p.atr_period == 14
        assert strategy._range_high == 0.0
        assert strategy._range_low == float("inf")
        assert not strategy._range_formed
        assert strategy._trades_this_session == 0

    def test_build_registry(self):
        """Test build_registry creates correct indicators."""
        registry = ORBStrategy.build_registry(atr_period=14, adx_period=14, volume_ma_period=20)
        assert len(registry) == 3
        assert "atr_14" in registry.get_all_output_columns()
        assert "adx_14" in registry.get_all_output_columns()
        assert "volume_ma_20" in registry.get_all_output_columns()


class TestORBStrategyValidation:
    """Test bar validation."""

    def test_invalid_bar_missing_field(self, strategy):
        """Test signal generation with missing bar field."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": 105.0,
            # missing 'low' and 'close'
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold
        assert "Invalid bar" in signal.reason

    def test_invalid_bar_none_value(self, strategy):
        """Test signal generation with None value."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 30),
            "open": 100.0,
            "high": None,
            "low": 99.0,
            "close": 103.0,
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold


class TestORBStrategySessionHandling:
    """Test session handling."""

    def test_outside_trading_hours(self, strategy):
        """Test signal generation outside trading hours."""
        bar = {
            "datetime": datetime(2024, 1, 1, 8, 0),  # Before market open
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "atr_14": 2.0,
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold
        assert "Outside trading session" in signal.reason

    def test_session_reset_on_new_day(self, strategy):
        """Test session state resets on new day."""
        # Day 1 morning
        bar1 = {
            "datetime": datetime(2024, 1, 1, 9, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar1)
        assert strategy._range_high == 105.0
        assert strategy._range_low == 99.0

        # Day 2 morning - should reset
        bar2 = {
            "datetime": datetime(2024, 1, 2, 9, 0),
            "open": 110.0,
            "high": 115.0,
            "low": 109.0,
            "close": 113.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar2)
        assert strategy._range_high == 115.0
        assert strategy._range_low == 109.0

    def test_session_reset_on_new_session(self, strategy):
        """Test session state resets on new session."""
        # Morning session
        bar1 = {
            "datetime": datetime(2024, 1, 1, 9, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar1)
        assert strategy._range_high == 105.0

        # Afternoon session - should reset
        bar2 = {
            "datetime": datetime(2024, 1, 1, 13, 0),
            "open": 110.0,
            "high": 115.0,
            "low": 109.0,
            "close": 113.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar2)
        assert strategy._range_high == 115.0
        assert strategy._range_low == 109.0


class TestORBStrategyRangeFormation:
    """Test opening range formation."""

    def test_range_formation_first_bar(self, strategy):
        """Test range formation on first bar."""
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "atr_14": 2.0,
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold
        assert "Forming range" in signal.reason
        assert strategy._range_high == 105.0
        assert strategy._range_low == 99.0
        assert not strategy._range_formed

    def test_range_formation_multiple_bars(self, strategy):
        """Test range formation over multiple bars."""
        # Bar 1 (9:00)
        bar1 = {
            "datetime": datetime(2024, 1, 1, 9, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar1)
        assert strategy._range_high == 105.0
        assert strategy._range_low == 99.0

        # Bar 2 (9:05) - expands range
        bar2 = {
            "datetime": datetime(2024, 1, 1, 9, 5),
            "open": 103.0,
            "high": 107.0,
            "low": 98.0,
            "close": 106.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar2)
        assert strategy._range_high == 107.0
        assert strategy._range_low == 98.0

    def test_range_formed_after_orb_period(self, strategy):
        """Test range is marked as formed after ORB period."""
        # Formation period (9:00 - 9:20, 20 minutes)
        for minute in range(0, 20, 5):
            bar = {
                "datetime": datetime(2024, 1, 1, 9, minute),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "atr_14": 2.0,
            }
            signal = strategy.generate_signal(bar)
            assert signal.is_hold
            assert not strategy._range_formed

        # First bar after formation period (9:20)
        bar_after = {
            "datetime": datetime(2024, 1, 1, 9, 20),
            "open": 103.0,
            "high": 104.0,
            "low": 102.0,
            "close": 103.5,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar_after)
        assert strategy._range_formed


class TestORBStrategyBreakout:
    """Test breakout signal generation."""

    def setup_method(self):
        """Setup range before each test."""
        pass

    def _form_range(self, strategy, high=105.0, low=99.0):
        """Helper to form opening range."""
        for minute in range(0, 20, 5):
            bar = {
                "datetime": datetime(2024, 1, 1, 9, minute),
                "open": 100.0,
                "high": high,
                "low": low,
                "close": 102.0,
                "atr_14": 2.0,
            }
            strategy.generate_signal(bar)

    def test_long_breakout(self, strategy):
        """Test long breakout signal."""
        self._form_range(strategy, high=105.0, low=99.0)

        # Breakout bar
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 105.0,
            "high": 107.0,  # Breaks above range_high + buffer
            "low": 104.0,
            "close": 106.5,
            "atr_14": 2.0,
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_long
        assert signal.entry_price > 0
        assert signal.stop_loss == 99.0  # use_range_sl=True
        assert signal.take_profit > signal.entry_price
        assert signal.ord_type == "LIMIT"

    def test_short_breakout(self, strategy):
        """Test short breakout signal."""
        self._form_range(strategy, high=105.0, low=99.0)

        # Breakout bar
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 99.0,
            "high": 100.0,
            "low": 97.0,  # Breaks below range_low - buffer
            "close": 97.5,
            "atr_14": 2.0,
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_short
        assert signal.entry_price > 0
        assert signal.stop_loss == 105.0  # use_range_sl=True
        assert signal.take_profit < signal.entry_price

    def test_long_only_blocks_short(self, orb_config):
        """Test long_only blocks short signals."""
        orb_config.strategy.long_only = True
        strategy = ORBStrategy(orb_config)
        self._form_range(strategy, high=105.0, low=99.0)

        # Short breakout bar
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 99.0,
            "high": 100.0,
            "low": 97.0,
            "close": 97.5,
            "atr_14": 2.0,
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold  # Should not generate short signal


class TestORBStrategyFilters:
    """Test range and optional filters."""

    def _form_range(self, strategy, high=105.0, low=99.0):
        """Helper to form opening range."""
        for minute in range(0, 20, 5):
            bar = {
                "datetime": datetime(2024, 1, 1, 9, minute),
                "open": 100.0,
                "high": high,
                "low": low,
                "close": 102.0,
                "atr_14": 2.0,
            }
            strategy.generate_signal(bar)

    def test_range_too_narrow(self, orb_config):
        """Test range too narrow filter."""
        orb_config.strategy.min_range_atr = 2.0
        strategy = ORBStrategy(orb_config)
        self._form_range(strategy, high=101.0, low=100.0)  # Range = 1.0, ATR = 2.0

        bar = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 101.0,
            "high": 103.0,
            "low": 100.0,
            "close": 102.5,
            "atr_14": 2.0,
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold
        assert "Range too narrow" in signal.reason

    def test_range_too_wide(self, orb_config):
        """Test range too wide filter."""
        orb_config.strategy.max_range_atr = 2.0
        strategy = ORBStrategy(orb_config)
        self._form_range(strategy, high=110.0, low=100.0)  # Range = 10.0, ATR = 2.0

        bar = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 110.0,
            "high": 112.0,
            "low": 109.0,
            "close": 111.5,
            "atr_14": 2.0,
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold
        assert "Range too wide" in signal.reason

    def test_volume_filter(self, orb_config):
        """Test volume filter."""
        orb_config.strategy.use_volume_filter = True
        orb_config.strategy.volume_filter_threshold = 1.0
        strategy = ORBStrategy(orb_config)
        self._form_range(strategy, high=105.0, low=99.0)

        bar = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 105.0,
            "high": 107.0,
            "low": 104.0,
            "close": 106.5,
            "atr_14": 2.0,
            "volume": 500,
            "volume_ma_20": 1000,  # Volume < threshold * volume_ma
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold
        assert "Volume below threshold" in signal.reason

    def test_adx_filter(self, orb_config):
        """Test ADX filter."""
        orb_config.strategy.use_adx_filter = True
        orb_config.strategy.adx_min = 25.0
        strategy = ORBStrategy(orb_config)
        self._form_range(strategy, high=105.0, low=99.0)

        bar = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 105.0,
            "high": 107.0,
            "low": 104.0,
            "close": 106.5,
            "atr_14": 2.0,
            "adx_14": 20.0,  # ADX < adx_min
        }
        signal = strategy.generate_signal(bar)
        assert signal.is_hold
        assert "ADX too low" in signal.reason


class TestORBStrategyPositionManagement:
    """Test position management."""

    def _form_range(self, strategy):
        """Helper to form opening range."""
        for minute in range(0, 20, 5):
            bar = {
                "datetime": datetime(2024, 1, 1, 9, minute),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "atr_14": 2.0,
            }
            strategy.generate_signal(bar)

    def test_no_signal_when_position_open(self, strategy):
        """Test no new signal when position is already open."""
        self._form_range(strategy)

        # Create position
        position = PositionSnapshot(
            is_flat=False,
            is_long=True,
            is_short=False,
            quantity=10,
            entry_price=106.0,
            stop_loss=99.0,
            take_profit=112.0,
        )

        bar = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 106.0,
            "high": 108.0,
            "low": 105.0,
            "close": 107.5,
            "atr_14": 2.0,
        }
        signal = strategy.generate_signal(bar, position=position)
        assert signal.is_hold

    def test_max_trades_per_session(self, strategy):
        """Test max trades per session limit."""
        self._form_range(strategy)

        # First trade
        bar1 = {
            "datetime": datetime(2024, 1, 1, 9, 25),
            "open": 105.0,
            "high": 107.0,
            "low": 104.0,
            "close": 106.5,
            "atr_14": 2.0,
        }
        signal1 = strategy.generate_signal(bar1)
        assert signal1.is_long
        assert strategy._trades_this_session == 1

        # Second trade - need to form new range in same session
        # Move to 9:50 (after first range period)
        for minute in range(30, 50, 5):
            bar = {
                "datetime": datetime(2024, 1, 1, 9, minute),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "atr_14": 2.0,
            }
            strategy.generate_signal(bar)

        # Now at 9:50, range should be formed, try second breakout
        bar2 = {
            "datetime": datetime(2024, 1, 1, 9, 55),
            "open": 105.0,
            "high": 107.0,
            "low": 104.0,
            "close": 106.5,
            "atr_14": 2.0,
        }
        signal2 = strategy.generate_signal(bar2)
        assert signal2.is_long
        assert strategy._trades_this_session == 2

        # Third trade attempt - should be blocked
        # Move to 10:00 and form another range
        for minute in range(0, 20, 5):
            bar = {
                "datetime": datetime(2024, 1, 1, 10, minute),
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 102.0,
                "atr_14": 2.0,
            }
            strategy.generate_signal(bar)

        bar3 = {
            "datetime": datetime(2024, 1, 1, 10, 25),
            "open": 105.0,
            "high": 107.0,
            "low": 104.0,
            "close": 106.5,
            "atr_14": 2.0,
        }
        signal3 = strategy.generate_signal(bar3)
        assert signal3.is_hold
        assert "trade limit" in signal3.reason


class TestORBStrategyStateSerialization:
    """Test state save/load."""

    def test_save_state(self, strategy):
        """Test save_state."""
        # Set some state
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar)

        state = strategy.save_state()
        assert state["range_high"] == 105.0
        assert state["range_low"] == 99.0
        assert state["range_formed"] is False
        assert state["trades_this_session"] == 0

    def test_load_state(self, strategy, orb_config):
        """Test load_state."""
        state = {
            "current_date": "2024-01-01",
            "current_session": "morning",
            "last_bar_dt": "2024-01-01T09:00:00",
            "range_high": 105.0,
            "range_low": 99.0,
            "range_formed": True,
            "trades_this_session": 1,
        }

        strategy2 = ORBStrategy(orb_config)
        strategy2.load_state(state)
        assert strategy2._range_high == 105.0
        assert strategy2._range_low == 99.0
        assert strategy2._range_formed is True
        assert strategy2._trades_this_session == 1

    def test_reset(self, strategy):
        """Test reset clears all state."""
        # Set some state
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar)
        assert strategy._range_high == 105.0

        # Reset
        strategy.reset()
        assert strategy._range_high == 0.0
        assert strategy._range_low == float("inf")
        assert not strategy._range_formed
        assert strategy._trades_this_session == 0


class TestORBStrategyProperties:
    """Test read-only properties."""

    def test_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.range_high == 0.0
        assert strategy.range_low == float("inf")
        assert not strategy.range_formed
        assert strategy.trades_this_session == 0

        # Update state
        bar = {
            "datetime": datetime(2024, 1, 1, 9, 0),
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "close": 103.0,
            "atr_14": 2.0,
        }
        strategy.generate_signal(bar)

        assert strategy.range_high == 105.0
        assert strategy.range_low == 99.0
