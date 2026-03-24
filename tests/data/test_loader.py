"""Tests for DataLoader with monthly cache."""

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data.loader import DataLoader


@pytest.fixture
def mock_data_service():
    """Mock DataServiceBase."""
    service = Mock()
    service.fetch_ohlcv = Mock(return_value=pd.DataFrame())
    return service


@pytest.fixture
def sample_ohlcv():
    """Sample valid 1min OHLCV data (loader now returns 1min bars)."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 09:00", periods=100, freq="1min"),
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        }
    )


class TestDataLoaderInit:
    """Test DataLoader initialization."""

    def test_init_defaults(self, mock_data_service):
        loader = DataLoader(mock_data_service)
        assert loader._svc == mock_data_service
        assert loader._cache_root == Path("data/cache")
        assert loader._chunk_size == 30
        assert loader._max_age_days == 7

    def test_init_custom_params(self, mock_data_service):
        loader = DataLoader(
            mock_data_service,
            cache_dir="custom/cache",
            chunk_size_days=60,
            cache_max_age_days=14,
        )
        assert loader._cache_root == Path("custom/cache")
        assert loader._chunk_size == 60
        assert loader._max_age_days == 14


class TestDataLoaderLoad:
    """Test DataLoader.load method with monthly cache."""

    def test_load_from_db_success(self, mock_data_service, sample_ohlcv, tmp_path):
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path), chunk_size_days=30)

        result = loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)

        assert len(result) == 100  # 1min bars
        assert list(result.columns) == ["datetime", "open", "high", "low", "close", "volume"]
        mock_data_service.fetch_ohlcv.assert_called()

        # Check monthly cache was created
        cache_path = tmp_path / "VN30F1M" / "1min" / "2024_01.parquet"
        assert cache_path.exists()

    def test_load_empty_data_raises_error(self, mock_data_service, tmp_path):
        mock_data_service.fetch_ohlcv.return_value = pd.DataFrame()
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))

        with pytest.raises(ValueError, match="No data found"):
            loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)

    def test_load_with_cache_hit(self, mock_data_service, sample_ohlcv, tmp_path):
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        # First load - should fetch from DB and cache
        result1 = loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=True)
        assert len(result1) == 100

        # Second load - should hit cache (past month)
        mock_data_service.fetch_ohlcv.reset_mock()
        result2 = loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=True)

        assert len(result2) == 100
        mock_data_service.fetch_ohlcv.assert_not_called()

    def test_load_bypass_cache(self, mock_data_service, sample_ohlcv, tmp_path):
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        # First load with cache
        loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=True)

        # Second load with use_cache=False
        mock_data_service.fetch_ohlcv.reset_mock()
        result = loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)

        assert len(result) == 100
        mock_data_service.fetch_ohlcv.assert_called()

    def test_load_multiple_months(self, mock_data_service, sample_ohlcv, tmp_path):
        """Test loading data spanning multiple months."""
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))

        result = loader.load("VN30F1M", "2024-01-01", "2024-03-31", use_cache=False)

        # Should call fetch_ohlcv 3 times (once per month)
        assert mock_data_service.fetch_ohlcv.call_count == 3
        assert len(result) > 0


class TestDataLoaderTickCSV:
    """Test DataLoader.load_tick_csv method."""

    def test_load_tick_csv_single_file(self, mock_data_service, tmp_path):
        """Test loading and aggregating tick CSV to 1min bars."""
        # Create tick CSV
        tick_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01 09:00", periods=100, freq="10s"),
                "price": [1000.0 + i * 0.1 for i in range(100)],
                "quantity": [100 + i for i in range(100)],
            }
        )
        csv_path = tmp_path / "ticks_2024_01.csv"
        tick_data.to_csv(csv_path, index=False)

        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path / "cache"))

        result = loader.load_tick_csv(
            path_pattern=str(csv_path),
            symbol="VN30F1M",
            cache_result=True,
        )

        # Should aggregate to 1min bars
        assert len(result) > 0
        assert len(result) < len(tick_data)  # Aggregated
        assert list(result.columns) == ["datetime", "open", "high", "low", "close", "volume"]

    def test_load_tick_csv_file_not_found(self, mock_data_service):
        loader = DataLoader(mock_data_service)

        with pytest.raises(FileNotFoundError):
            loader.load_tick_csv("nonexistent_*.csv", symbol="VN30F1M")


class TestDataLoaderCache:
    """Test DataLoader monthly cache functionality."""

    def test_list_cached_months(self, mock_data_service, sample_ohlcv, tmp_path):
        """Test listing cached months."""
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        # Cache 3 months
        loader.load("VN30F1M", "2024-01-01", "2024-03-31", use_cache=False)

        cached = loader.list_cached_months("VN30F1M")
        assert len(cached) == 3
        assert "2024_01" in cached
        assert "2024_02" in cached
        assert "2024_03" in cached

    def test_invalidate_cache_all_symbols(self, mock_data_service, sample_ohlcv, tmp_path):
        """Test invalidating all symbols."""
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        # Create cache for 2 symbols
        loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)
        loader.load("VN30F2M", "2024-01-01", "2024-01-31", use_cache=False)

        count = loader.invalidate_cache()
        assert count == 2

    def test_invalidate_cache_specific_symbol(self, mock_data_service, sample_ohlcv, tmp_path):
        """Test invalidating specific symbol."""
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        loader.load("VN30F1M", "2024-01-01", "2024-02-28", use_cache=False)

        count = loader.invalidate_cache(symbol="VN30F1M")
        assert count == 2  # 2 months

    def test_invalidate_cache_specific_month(self, mock_data_service, sample_ohlcv, tmp_path):
        """Test invalidating specific month."""
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        loader.load("VN30F1M", "2024-01-01", "2024-02-28", use_cache=False)

        count = loader.invalidate_cache(symbol="VN30F1M", month_key="2024_01")
        assert count == 1

        # February should still be cached
        manifest = loader._get_manifest("VN30F1M")
        assert manifest.is_cached("2024_02")
        assert not manifest.is_cached("2024_01")


class TestDataLoaderChunking:
    """Test DataLoader delegates chunking config to data service."""

    def test_chunk_size_passed_to_service(self, mock_data_service, sample_ohlcv, tmp_path):
        """Test that chunk_size_days is passed to fetch_ohlcv."""
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path), chunk_size_days=60)

        loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)

        # Check that chunk_size_days was passed
        call_args = mock_data_service.fetch_ohlcv.call_args
        assert call_args[1]["chunk_size_days"] == 60

    def test_deduplication_across_months(self, mock_data_service, tmp_path):
        """Test that overlapping data across months is deduplicated."""
        # Create data with some overlap
        data1 = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=50, freq="1h"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            }
        )
        data2 = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-31 20:00", periods=50, freq="1h"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000.0,
            }
        )

        mock_data_service.fetch_ohlcv.side_effect = [data1, data2]
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))

        result = loader.load("VN30F1M", "2024-01-01", "2024-02-28", use_cache=False)

        # Should deduplicate overlapping timestamps
        assert result["datetime"].is_unique
