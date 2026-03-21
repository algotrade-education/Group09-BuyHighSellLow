"""Tests for DataLoader."""

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
    """Sample valid OHLCV data."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 09:00", periods=10, freq="5min"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
    )


class TestDataLoaderInit:
    """Test DataLoader initialization."""

    def test_init_defaults(self, mock_data_service):
        loader = DataLoader(mock_data_service)
        assert loader._svc == mock_data_service
        assert loader._cache_dir == Path("data/cache")
        assert loader._chunk_size == 30

    def test_init_custom_params(self, mock_data_service):
        loader = DataLoader(mock_data_service, cache_dir="custom/cache", chunk_size_days=60)
        assert loader._cache_dir == Path("custom/cache")
        assert loader._chunk_size == 60


class TestDataLoaderLoad:
    """Test DataLoader.load method."""

    def test_load_from_db_success(self, mock_data_service, sample_ohlcv, tmp_path):
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path), chunk_size_days=30)

        result = loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)

        assert len(result) == 10
        assert list(result.columns) == ["datetime", "open", "high", "low", "close", "volume"]
        mock_data_service.fetch_ohlcv.assert_called()

    def test_load_empty_data_raises_error(self, mock_data_service, tmp_path):
        mock_data_service.fetch_ohlcv.return_value = pd.DataFrame()
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))

        with pytest.raises(ValueError, match="No data available"):
            loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)

    def test_load_db_error_raises_runtime_error(self, mock_data_service, tmp_path):
        mock_data_service.fetch_ohlcv.side_effect = Exception("DB connection failed")
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))

        with pytest.raises(RuntimeError, match="DB fetch failed"):
            loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)

    def test_load_with_cache_hit(self, mock_data_service, sample_ohlcv, tmp_path):
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        # First load - should fetch from DB and cache
        result1 = loader.load("VN30F1M", "2024-01-01", "2024-01-31", freq="5min")
        assert len(result1) == 10

        # Second load - should hit cache
        mock_data_service.fetch_ohlcv.reset_mock()
        result2 = loader.load("VN30F1M", "2024-01-01", "2024-01-31", freq="5min")

        assert len(result2) == 10
        mock_data_service.fetch_ohlcv.assert_not_called()

    def test_load_bypass_cache(self, mock_data_service, sample_ohlcv, tmp_path):
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        # First load with cache
        loader.load("VN30F1M", "2024-01-01", "2024-01-31")

        # Second load with use_cache=False
        mock_data_service.fetch_ohlcv.reset_mock()
        result = loader.load("VN30F1M", "2024-01-01", "2024-01-31", use_cache=False)

        assert len(result) == 10
        mock_data_service.fetch_ohlcv.assert_called()


class TestDataLoaderCSV:
    """Test DataLoader.load_csv method."""

    def test_load_csv_single_file(self, mock_data_service, sample_ohlcv, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_ohlcv.to_csv(csv_path, index=False)

        loader = DataLoader(mock_data_service)
        result = loader.load_csv(str(csv_path))

        assert len(result) == 10
        assert "datetime" in result.columns

    def test_load_csv_file_not_found(self, mock_data_service):
        loader = DataLoader(mock_data_service)

        with pytest.raises(FileNotFoundError):
            loader.load_csv("nonexistent.csv")

    def test_load_csv_with_duplicates(self, mock_data_service, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "datetime": ["2024-01-01 09:00", "2024-01-01 09:05", "2024-01-01 09:05"],
                "open": [100.0, 101.0, 101.5],
                "high": [101.0, 102.0, 102.5],
                "low": [99.0, 100.0, 100.5],
                "close": [100.5, 101.5, 101.8],
                "volume": [1000, 1100, 1150],
            }
        )
        df.to_csv(csv_path, index=False)

        loader = DataLoader(mock_data_service)
        result = loader.load_csv(str(csv_path))

        # Should keep last duplicate
        assert len(result) == 2
        assert result.iloc[1]["open"] == 101.5

    def test_load_csv_no_validation(self, mock_data_service, tmp_path):
        csv_path = tmp_path / "bad_data.csv"
        df = pd.DataFrame(
            {
                "datetime": ["2024-01-01 09:00"],
                "open": [-100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )
        df.to_csv(csv_path, index=False)

        loader = DataLoader(mock_data_service)
        result = loader.load_csv(str(csv_path), validate=False)

        assert len(result) == 1


class TestDataLoaderCache:
    """Test DataLoader cache functionality."""

    def test_cache_key_generation(self, mock_data_service):
        loader = DataLoader(mock_data_service)

        key1 = loader._build_cache_key("VN30F1M", "2024-01-01", "2024-01-31", "5min")
        key2 = loader._build_cache_key("VN30F1M", "2024-01-01", "2024-01-31", "5min")
        key3 = loader._build_cache_key("VN30F1M", "2024-01-01", "2024-01-31", "15min")

        assert key1 == key2
        assert key1 != key3

    def test_invalidate_cache_all(self, mock_data_service, sample_ohlcv, tmp_path):
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        # Create some cache files
        loader.load("VN30F1M", "2024-01-01", "2024-01-31")
        loader.load("VN30F2M", "2024-01-01", "2024-01-31")

        count = loader.invalidate_cache()
        assert count == 2

    def test_invalidate_cache_empty_dir(self, mock_data_service, tmp_path):
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path / "nonexistent"))
        count = loader.invalidate_cache()
        assert count == 0

    def test_corrupt_cache_refetches(self, mock_data_service, sample_ohlcv, tmp_path):
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path))
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv

        # Create cache
        loader.load("VN30F1M", "2024-01-01", "2024-01-31")

        # Corrupt cache file
        cache_files = list(tmp_path.glob("*.parquet"))
        cache_files[0].write_text("corrupted data")

        # Should refetch from DB
        mock_data_service.fetch_ohlcv.reset_mock()
        result = loader.load("VN30F1M", "2024-01-01", "2024-01-31")

        assert len(result) == 10
        mock_data_service.fetch_ohlcv.assert_called()


class TestDataLoaderChunking:
    """Test DataLoader delegates chunking config to data service."""

    def test_fetch_from_db_passes_chunk_size_to_data_service(self, mock_data_service, sample_ohlcv):
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv
        loader = DataLoader(mock_data_service, chunk_size_days=60)

        result = loader._fetch_from_db("VN30F1M", "2024-01-01", "2024-01-31")

        assert len(result) == 10
        mock_data_service.fetch_ohlcv.assert_called_once_with(
            contract_name="VN30F1M",
            from_date="2024-01-01",
            to_date="2024-01-31",
            chunk_size_days=60,
        )

    def test_fetch_from_db_single_service_call(self, mock_data_service, sample_ohlcv):
        mock_data_service.fetch_ohlcv.return_value = sample_ohlcv
        loader = DataLoader(mock_data_service, chunk_size_days=15)

        loader._fetch_from_db("VN30F1M", "2024-01-01", "2024-02-15")

        assert mock_data_service.fetch_ohlcv.call_count == 1

    def test_fetch_chunks_deduplication(self, mock_data_service, tmp_path):
        # Create overlapping chunks
        chunk1 = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=5, freq="1D"),
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        mock_data_service.fetch_ohlcv.return_value = chunk1
        loader = DataLoader(mock_data_service, cache_dir=str(tmp_path), chunk_size_days=3)

        result = loader._fetch_from_db("VN30F1M", "2024-01-01", "2024-01-10")

        # Should deduplicate overlapping dates
        assert result["datetime"].is_unique
