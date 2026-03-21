"""
Test suite for DataServiceBase abstract class.
"""

from datetime import datetime

import pandas as pd
import pytest

from src.database.base import DataServiceBase


class ConcreteDataService(DataServiceBase):
    """Concrete implementation for testing."""

    def __init__(self) -> None:
        self.matched_data = pd.DataFrame()
        self.matched_range_data = pd.DataFrame()
        self.last_matched_data = pd.DataFrame()
        self.close_data = pd.DataFrame()
        self.bid_ask_data = pd.DataFrame()

    def get_matched_data(self, contract_name: str, from_date: str, to_date: str) -> pd.DataFrame:
        return self.matched_data

    def get_matched_data_in_range(
        self, contract_name: str, from_datetime: datetime, to_datetime: datetime
    ) -> pd.DataFrame:
        return self.matched_range_data

    def get_last_matched_before(
        self, contract_name: str, before_datetime: datetime
    ) -> pd.DataFrame:
        return self.last_matched_data

    def get_close_data(self, contract_name: str, from_date: str, to_date: str) -> pd.DataFrame:
        return self.close_data

    def get_bid_ask_data(self, contract_name: str, from_date: str, to_date: str) -> pd.DataFrame:
        return self.bid_ask_data


class TestDataServiceBase:
    """Test DataServiceBase abstract methods and concrete implementations."""

    @pytest.fixture
    def service(self) -> ConcreteDataService:
        """Create concrete service instance."""
        return ConcreteDataService()

    def test_fetch_ohlcv_empty_chunks(self, service: ConcreteDataService) -> None:
        """Test fetch_ohlcv with no data."""
        service.matched_data = pd.DataFrame()

        df = service.fetch_ohlcv("VN30F1M", "2024-01-01", "2024-01-31")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_fetch_ohlcv_with_data(self, service: ConcreteDataService) -> None:
        """Test fetch_ohlcv aggregates tick data to OHLCV."""
        service.matched_data = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=5, freq="1min"),
                "tickersymbol": ["VN30F2401"] * 5,
                "price": [1200.0, 1201.0, 1199.0, 1202.0, 1203.0],
                "quantity": [100.0, 150.0, 120.0, 180.0, 200.0],
            }
        )

        df = service.fetch_ohlcv("VN30F1M", "2024-01-01", "2024-01-31", chunk_size_days=30)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_fetch_bucket_bar_empty(self, service: ConcreteDataService) -> None:
        """Test fetch_bucket_bar with no data."""
        service.matched_range_data = pd.DataFrame()

        df = service.fetch_bucket_bar(
            "VN30F1M",
            datetime(2024, 1, 1, 9, 0),
            datetime(2024, 1, 1, 9, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_fetch_bucket_bar_with_data(self, service: ConcreteDataService) -> None:
        """Test fetch_bucket_bar creates single OHLCV bar."""
        service.matched_range_data = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=3, freq="10s"),
                "tickersymbol": ["VN30F2401"] * 3,
                "price": [1200.0, 1205.0, 1198.0],
                "quantity": [100.0, 150.0, 120.0],
            }
        )
        service.last_matched_data = pd.DataFrame(
            {
                "datetime": [datetime(2024, 1, 1, 8, 59)],
                "tickersymbol": ["VN30F2401"],
                "price": [1195.0],
                "quantity": [50.0],
            }
        )

        df = service.fetch_bucket_bar(
            "VN30F1M",
            datetime(2024, 1, 1, 9, 0),
            datetime(2024, 1, 1, 9, 1),
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["open"].iloc[0] == 1200.0
        assert df["high"].iloc[0] == 1205.0
        assert df["low"].iloc[0] == 1198.0
        assert df["close"].iloc[0] == 1198.0
        assert "volume" in df.columns
        assert "rows" in df.columns

    def test_close(self, service: ConcreteDataService) -> None:
        """Test close method exists and can be called."""
        # Base class close() is a no-op, just verify it doesn't raise
        service.close()

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        service = ConcreteDataService()

        with service as svc:
            assert svc is service

        # Context manager should work without errors

    def test_aggregate_to_ohlcv_empty(self, service: ConcreteDataService) -> None:
        """Test _aggregate_to_ohlcv with empty DataFrame."""
        df = service._aggregate_to_ohlcv(pd.DataFrame())
        assert df.empty

    def test_aggregate_to_ohlcv_with_ticks(self, service: ConcreteDataService) -> None:
        """Test _aggregate_to_ohlcv aggregates correctly."""
        ticks = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=10, freq="30s"),
                "price": [1200, 1201, 1199, 1202, 1203, 1200, 1198, 1201, 1204, 1205],
                "quantity": [100, 150, 120, 180, 200, 110, 130, 160, 190, 210],
            }
        )

        df = service._aggregate_to_ohlcv(ticks)

        assert not df.empty
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_calculate_volume_no_quantity_column(self, service: ConcreteDataService) -> None:
        """Test _calculate_volume when quantity column is missing."""
        ticks = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=3, freq="1min"),
                "price": [1200.0, 1201.0, 1202.0],
            }
        )

        volume = service._calculate_volume(ticks, "VN30F1M", datetime(2024, 1, 1, 9, 0))

        assert volume == 0.0

    def test_calculate_volume_with_previous_quantity(self, service: ConcreteDataService) -> None:
        """Test _calculate_volume with previous quantity."""
        service.last_matched_data = pd.DataFrame(
            {
                "datetime": [datetime(2024, 1, 1, 8, 59)],
                "quantity": [50.0],
            }
        )

        ticks = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=3, freq="1min"),
                "price": [1200.0, 1201.0, 1202.0],
                "quantity": [100.0, 150.0, 170.0],
            }
        )

        volume = service._calculate_volume(ticks, "VN30F1M", datetime(2024, 1, 1, 9, 0))

        assert volume > 0

    def test_calculate_volume_without_previous_quantity(self, service: ConcreteDataService) -> None:
        """Test _calculate_volume without previous quantity."""
        service.last_matched_data = pd.DataFrame()

        ticks = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01 09:00", periods=3, freq="1min"),
                "price": [1200.0, 1201.0, 1202.0],
                "quantity": [100.0, 150.0, 170.0],
            }
        )

        volume = service._calculate_volume(ticks, "VN30F1M", datetime(2024, 1, 1, 9, 0))

        assert volume >= 0
