"""Tests for DataPipeline."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.indicators import IndicatorRegistry, IndicatorSpec, WilderATR
from src.data.pipeline import DataPipeline


@pytest.fixture
def sample_df():
    """Sample OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 09:00", periods=30, freq="5min"),
            "open": range(100, 130),
            "high": range(101, 131),
            "low": range(99, 129),
            "close": range(100, 130),
            "volume": [1000 + i * 100 for i in range(30)],
        }
    )


@pytest.fixture
def registry_with_indicators():
    """Registry with ATR and VolumeMA."""
    registry = IndicatorRegistry()
    registry.register(IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14"))
    registry.register(
        IndicatorSpec(name="volume_ma", params={"period": 20}, output_column="volume_ma_20")
    )
    return registry


class TestDataPipelineInit:
    """Test DataPipeline initialization."""

    def test_init_defaults(self, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators)
        assert pipeline._registry == registry_with_indicators
        assert pipeline._cache_dir == Path("data/cache")
        assert pipeline._use_cache is True

    def test_init_custom_params(self, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators, cache_dir="custom/cache", use_cache=False)
        assert pipeline._cache_dir == Path("custom/cache")
        assert pipeline._use_cache is False


class TestDataPipelineRun:
    """Test DataPipeline.run method."""

    def test_run_empty_dataframe(self, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators, use_cache=False)
        df = pd.DataFrame()

        result = pipeline.run(df)

        assert result.empty

    def test_run_empty_registry(self, sample_df):
        empty_registry = IndicatorRegistry()
        pipeline = DataPipeline(empty_registry, use_cache=False)

        result = pipeline.run(sample_df)

        assert len(result) == 30
        assert "atr_14" not in result.columns

    def test_run_adds_indicator_columns(self, sample_df, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators, use_cache=False)

        result = pipeline.run(sample_df)

        assert "atr_14" in result.columns
        assert "volume_ma_20" in result.columns
        assert len(result) == 30

    def test_run_indicator_warm_up(self, sample_df, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators, use_cache=False)

        result = pipeline.run(sample_df)

        # First 14 bars should have NaN for ATR (warm-up period)
        assert result["atr_14"].iloc[:13].isna().all()
        assert result["atr_14"].iloc[14:].notna().any()

    def test_run_with_cache(self, sample_df, registry_with_indicators, tmp_path):
        pipeline = DataPipeline(registry_with_indicators, cache_dir=str(tmp_path), use_cache=True)

        # First run - compute and cache
        result1 = pipeline.run(sample_df)

        # Second run - should load from cache
        result2 = pipeline.run(sample_df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_get_required_lookback(self, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators)

        lookback = pipeline.get_required_lookback()

        # ATR needs 14, VolumeMA needs 20
        assert lookback == 20


class TestDataPipelineCompute:
    """Test DataPipeline._compute method."""

    def test_compute_single_indicator(self, sample_df):
        registry = IndicatorRegistry()
        registry.register(
            IndicatorSpec(name="volume_ma", params={"period": 3}, output_column="vma_3")
        )
        pipeline = DataPipeline(registry, use_cache=False)

        result = pipeline._compute(sample_df)

        assert "vma_3" in result.columns
        # After warm-up, should have valid values
        assert result["vma_3"].iloc[2:].notna().any()

    def test_compute_preserves_original_columns(self, sample_df, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators, use_cache=False)

        result = pipeline._compute(sample_df)

        for col in ["datetime", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_feed_indicator_missing_inputs(self, sample_df):
        registry = IndicatorRegistry()
        registry.register(IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14"))
        pipeline = DataPipeline(registry, use_cache=False)

        # Create bar without required inputs
        bar = {"datetime": "2024-01-01", "open": 100}
        indicator = WilderATR(period=14)

        result = pipeline._feed_indicator(indicator, "atr_14", bar)

        assert result is None


class TestDataPipelineCache:
    """Test DataPipeline cache functionality."""

    def test_build_cache_key_consistency(self, sample_df, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators)

        key1 = pipeline._build_cache_key(sample_df)
        key2 = pipeline._build_cache_key(sample_df)

        assert key1 == key2

    def test_build_cache_key_different_data(self, sample_df, registry_with_indicators):
        pipeline = DataPipeline(registry_with_indicators)

        key1 = pipeline._build_cache_key(sample_df)

        # Modify data
        sample_df_modified = sample_df.copy()
        sample_df_modified["close"] = sample_df_modified["close"] + 10

        key2 = pipeline._build_cache_key(sample_df_modified)

        # Keys should be same (we only check shape/datetime, not values)
        assert key1 == key2

    def test_build_cache_key_different_registry(self, sample_df):
        registry1 = IndicatorRegistry()
        registry1.register(IndicatorSpec(name="atr", params={"period": 14}, output_column="atr_14"))

        registry2 = IndicatorRegistry()
        registry2.register(IndicatorSpec(name="atr", params={"period": 20}, output_column="atr_20"))

        pipeline1 = DataPipeline(registry1)
        pipeline2 = DataPipeline(registry2)

        key1 = pipeline1._build_cache_key(sample_df)
        key2 = pipeline2._build_cache_key(sample_df)

        assert key1 != key2

    def test_clear_cache(self, sample_df, registry_with_indicators, tmp_path):
        pipeline = DataPipeline(registry_with_indicators, cache_dir=str(tmp_path))

        # Create cache
        pipeline.run(sample_df)

        count = pipeline.clear_cache()

        assert count == 1
        assert len(list(tmp_path.glob("*.parquet"))) == 0

    def test_clear_cache_empty_dir(self, registry_with_indicators, tmp_path):
        pipeline = DataPipeline(registry_with_indicators, cache_dir=str(tmp_path / "nonexistent"))

        count = pipeline.clear_cache()

        assert count == 0

    def test_load_cache_missing_columns(self, sample_df, registry_with_indicators, tmp_path):
        pipeline = DataPipeline(registry_with_indicators, cache_dir=str(tmp_path))

        # Create cache with indicators
        _ = pipeline.run(sample_df)

        # Modify registry to expect different columns
        registry_with_indicators.register(
            IndicatorSpec(name="adx", params={"period": 14}, output_column="adx_14")
        )
        pipeline2 = DataPipeline(registry_with_indicators, cache_dir=str(tmp_path))

        # Should recompute because cache is missing adx_14 column
        result2 = pipeline2.run(sample_df)

        assert "adx_14" in result2.columns
