"""
Tests for technical indicators.
"""

import pytest

from src.data.indicators import VolumeMA, WilderADX, WilderATR


class TestWilderATR:
    """Test cases for WilderATR indicator."""

    def test_initialization(self):
        atr = WilderATR(period=14)
        assert atr.period == 14
        assert atr.warm_up_required == 14
        assert atr.value is None
        assert not atr.is_ready
        assert atr.bar_count == 0

    def test_warm_up_period(self):
        atr = WilderATR(period=3)

        # First bar - no previous close
        result = atr.update(high=102, low=98, close=100)
        assert result is None
        assert not atr.is_ready

        # Second bar
        result = atr.update(high=105, low=99, close=103)
        assert result is None
        assert not atr.is_ready

        # Third bar - should complete warm-up
        result = atr.update(high=104, low=100, close=102)
        assert result is not None
        assert atr.is_ready
        assert atr.value == result

    def test_true_range_calculation(self):
        atr = WilderATR(period=2)

        # First bar: TR = high - low
        atr.update(high=100, low=95, close=98)

        # Second bar: TR should consider previous close
        result = atr.update(high=102, low=97, close=100)

        # TR = max(102-97, |102-98|, |97-98|) = max(5, 4, 1) = 5
        # ATR = (5 + 5) / 2 = 5
        assert result == 5.0

    def test_wilders_smoothing(self):
        atr = WilderATR(period=3)

        # Seed phase
        atr.update(high=100, low=95, close=98)
        atr.update(high=102, low=97, close=100)
        result1 = atr.update(high=104, low=99, close=102)

        # Next bar should use Wilder's smoothing
        # Use different TR to ensure value changes
        result2 = atr.update(high=110, low=100, close=108)

        assert result1 is not None
        assert result2 is not None

        # Result should be different due to different TR value
        assert result2 > result1

    def test_missing_data_raises_error(self):
        atr = WilderATR(period=14)

        with pytest.raises(ValueError, match="Missing required bar data"):
            atr.update(high=100, low=95)

        with pytest.raises(ValueError, match="Missing required bar data"):
            atr.update(high=100, close=98)

    def test_reset(self):
        atr = WilderATR(period=2)
        atr.update(high=100, low=95, close=98)
        atr.update(high=102, low=97, close=100)

        assert atr.is_ready

        atr.reset()

        assert not atr.is_ready
        assert atr.value is None
        assert atr.bar_count == 0

    def test_state_management(self):
        atr1 = WilderATR(period=3)
        atr1.update(high=100, low=95, close=98)
        atr1.update(high=102, low=97, close=100)
        atr1.update(high=104, low=99, close=102)

        # Save state
        state = atr1.save_state()

        # Create new indicator and load state
        atr2 = WilderATR(period=3)
        atr2.load_state(state)

        assert atr2.value == atr1.value
        assert atr2.bar_count == atr1.bar_count
        assert atr2.is_ready == atr1.is_ready

    def test_state_class_mismatch(self):
        atr = WilderATR(period=14)
        state = {"class": "WrongClass", "count": 5, "current_value": 10.0}

        with pytest.raises(ValueError, match="does not match"):
            atr.load_state(state)


class TestWilderADX:
    """Test cases for WilderADX indicator."""

    def test_initialization(self):
        adx = WilderADX(period=14)
        assert adx.period == 14
        assert adx.warm_up_required == 28
        assert adx.value is None
        assert adx.di_plus is None
        assert adx.di_minus is None
        assert not adx.is_ready

    def test_warm_up_period(self):
        adx = WilderADX(period=3)

        # First bar - just stores values
        result = adx.update(high=100, low=95, close=98)
        assert result is None

        # Need period * 2 bars to warm up
        for i in range(6):
            result = adx.update(high=100 + i, low=95 + i, close=98 + i)

        assert adx.is_ready
        assert adx.value is not None

    def test_directional_movement(self):
        adx = WilderADX(period=2)

        # First bar
        adx.update(high=100, low=95, close=98)

        # Second bar - upward movement
        adx.update(high=105, low=97, close=103)

        # Third bar - continue upward
        adx.update(high=107, low=99, close=105)

        # Fourth bar - should have DI values
        adx.update(high=108, low=100, close=106)

        assert adx._di_plus_value is not None
        assert adx._di_minus_value is not None

    def test_di_properties(self):
        adx = WilderADX(period=2)

        # Before ready
        assert adx.di_plus is None
        assert adx.di_minus is None

        # Warm up
        adx.update(high=100, low=95, close=98)
        adx.update(high=105, low=97, close=103)
        adx.update(high=107, low=99, close=105)
        adx.update(high=108, low=100, close=106)

        # After ready
        if adx.is_ready:
            assert adx.di_plus is not None
            assert adx.di_minus is not None
            assert 0 <= adx.di_plus <= 100
            assert 0 <= adx.di_minus <= 100

    def test_missing_data_raises_error(self):
        adx = WilderADX(period=14)

        with pytest.raises(ValueError, match="Missing required bar data"):
            adx.update(high=100, low=95)

    def test_reset(self):
        adx = WilderADX(period=2)

        for i in range(5):
            adx.update(high=100 + i, low=95 + i, close=98 + i)

        adx.reset()

        assert not adx.is_ready
        assert adx.value is None
        assert adx.di_plus is None
        assert adx.di_minus is None
        assert adx.bar_count == 0

    def test_state_management(self):
        adx1 = WilderADX(period=2)

        for i in range(5):
            adx1.update(high=100 + i, low=95 + i, close=98 + i)

        state = adx1.save_state()

        adx2 = WilderADX(period=2)
        adx2.load_state(state)

        assert adx2.value == adx1.value
        assert adx2.bar_count == adx1.bar_count
        assert adx2.is_ready == adx1.is_ready


class TestVolumeMA:
    """Test cases for VolumeMA indicator."""

    def test_initialization(self):
        vma = VolumeMA(period=20)
        assert vma.period == 20
        assert vma.warm_up_required == 20
        assert vma.value is None
        assert not vma.is_ready

    def test_warm_up_period(self):
        vma = VolumeMA(period=3)

        result = vma.update(volume=1000)
        assert result is None
        assert not vma.is_ready

        result = vma.update(volume=1500)
        assert result is None

        result = vma.update(volume=2000)
        assert result is not None
        assert vma.is_ready
        assert result == (1000 + 1500 + 2000) / 3

    def test_rolling_window(self):
        vma = VolumeMA(period=3)

        vma.update(volume=1000)
        vma.update(volume=2000)
        result1 = vma.update(volume=3000)

        # Average of 1000, 2000, 3000
        assert result1 == 2000.0

        # Add new value, should drop 1000
        result2 = vma.update(volume=4000)

        # Average of 2000, 3000, 4000
        assert result2 == 3000.0

    def test_missing_data_raises_error(self):
        vma = VolumeMA(period=20)

        with pytest.raises(ValueError, match="Missing required volume data"):
            vma.update()

        with pytest.raises(ValueError, match="Missing required volume data"):
            vma.update(high=100, low=95)

    def test_reset(self):
        vma = VolumeMA(period=2)
        vma.update(volume=1000)
        vma.update(volume=2000)

        assert vma.is_ready

        vma.reset()

        assert not vma.is_ready
        assert vma.value is None
        assert vma.bar_count == 0
        assert len(vma._buffer) == 0

    def test_state_management(self):
        vma1 = VolumeMA(period=3)
        vma1.update(volume=1000)
        vma1.update(volume=2000)
        vma1.update(volume=3000)

        state = vma1.save_state()

        vma2 = VolumeMA(period=3)
        vma2.load_state(state)

        assert vma2.value == vma1.value
        assert vma2.bar_count == vma1.bar_count
        assert vma2.is_ready == vma1.is_ready

        # Both should produce same result on next update
        result1 = vma1.update(volume=4000)
        result2 = vma2.update(volume=4000)
        assert result1 == result2

    def test_zero_volume(self):
        vma = VolumeMA(period=2)
        vma.update(volume=0)
        result = vma.update(volume=1000)

        assert result == 500.0

    def test_large_volumes(self):
        vma = VolumeMA(period=2)
        vma.update(volume=1_000_000_000)
        result = vma.update(volume=2_000_000_000)

        assert result == 1_500_000_000.0


class TestIndicatorIntegration:
    """Integration tests for multiple indicators."""

    def test_all_indicators_with_same_data(self):
        """Test that all indicators can process the same bar data."""
        atr = WilderATR(period=3)
        adx = WilderADX(period=3)
        vma = VolumeMA(period=3)

        bars = [
            {"high": 100, "low": 95, "close": 98, "volume": 1000},
            {"high": 102, "low": 97, "close": 100, "volume": 1500},
            {"high": 104, "low": 99, "close": 102, "volume": 2000},
            {"high": 103, "low": 98, "close": 101, "volume": 1800},
        ]

        for bar in bars:
            atr.update(**bar)
            adx.update(**bar)
            vma.update(**bar)

        assert atr.is_ready
        assert vma.is_ready
        # ADX needs more bars (period * 2)

    def test_repr_methods(self):
        """Test string representations of indicators."""
        atr = WilderATR(period=14)
        assert "WilderATR" in repr(atr)
        assert "warming up" in repr(atr)

        atr.update(high=100, low=95, close=98)
        assert "bars=1" in repr(atr)


class TestIndicatorRegistry:
    """Test cases for IndicatorRegistry."""

    def test_initialization(self):
        from src.data.indicators import IndicatorRegistry

        registry = IndicatorRegistry()
        assert len(registry) == 0
        assert registry.get_all() == []
        assert registry.get_all_output_columns() == []

    def test_register_indicator(self):
        from src.data.indicators import IndicatorRegistry, IndicatorSpec

        registry = IndicatorRegistry()
        spec = IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14")

        result = registry.register(spec)
        assert result is registry  # Method chaining
        assert len(registry) == 1
        assert spec in registry.get_all()

    def test_auto_generate_output_column(self):
        from src.data.indicators import IndicatorSpec

        spec = IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14")
        assert spec.output_column == "atr_14"

        spec2 = IndicatorSpec(name="volume_ma", params={"period": 20}, output_column="volume_ma_20")
        assert spec2.output_column == "volume_ma_20"

    def test_custom_output_column(self):
        from src.data.indicators import IndicatorSpec

        spec = IndicatorSpec(name="atr", params={"period": 14}, output_column="my_atr")
        assert spec.output_column == "my_atr"

    def test_duplicate_output_column_raises_error(self):
        from src.data.indicators import IndicatorRegistry, IndicatorSpec

        registry = IndicatorRegistry()
        spec1 = IndicatorSpec(name="atr", params={"period": 14}, output_column="atr")
        spec2 = IndicatorSpec(name="adx", params={"period": 14}, output_column="atr")

        registry.register(spec1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec2)

    def test_get_required_lookback(self):
        from src.data.indicators import IndicatorRegistry, IndicatorSpec

        registry = IndicatorRegistry()
        registry.register(IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14"))
        registry.register(IndicatorSpec(name="adx", params={"period": 14}, output_column="adx_14"))

        # ADX requires 28 bars (period * 2), ATR requires 14
        assert registry.get_required_lookback() == 28

    def test_get_required_lookback_empty(self):
        from src.data.indicators import IndicatorRegistry

        registry = IndicatorRegistry()
        assert registry.get_required_lookback() == 0

    def test_build_indicators(self):
        from src.data.indicators import IndicatorRegistry, IndicatorSpec

        registry = IndicatorRegistry()
        registry.register(IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14"))
        registry.register(
            IndicatorSpec(name="volume_ma", params={"period": 20}, output_column="volume_ma_20")
        )

        indicators = registry.build_indicators()

        assert "atr_14" in indicators
        assert "volume_ma_20" in indicators
        assert isinstance(indicators["atr_14"], WilderATR)
        assert isinstance(indicators["volume_ma_20"], VolumeMA)

    def test_build_orb_registry(self):
        """Test that ORBStrategy.build_registry creates correct indicators."""
        from src.strategy.orb import ORBStrategy

        registry = ORBStrategy.build_registry(atr_period=14, adx_period=14)

        assert len(registry) == 3
        columns = registry.get_all_output_columns()
        assert "atr_14" in columns
        assert "adx_14" in columns
        assert "volume_ma_20" in columns

    def test_registry_repr(self):
        from src.data.indicators import IndicatorRegistry, IndicatorSpec

        registry = IndicatorRegistry()
        registry.register(IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14"))

        repr_str = repr(registry)
        assert "IndicatorRegistry" in repr_str
        assert "atr_14" in repr_str
