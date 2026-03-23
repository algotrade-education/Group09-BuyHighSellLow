"""
Test suite for PostgresDataService class.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.database.data_service import PostgresDataService, create_data_service, get_data_service
from src.database.query import (
    BID_ASK_QUERY,
    CLOSE_QUERY,
    MATCHED_LAST_BEFORE_QUERY,
    MATCHED_QUERY,
    MATCHED_RANGE_QUERY,
)


class TestPostgresDataService:
    """Test PostgresDataService data fetching methods."""

    @pytest.fixture
    def mock_connection(self) -> MagicMock:
        """Create mock DatabaseConnection."""
        conn = MagicMock()
        conn.execute.return_value = []
        return conn

    @pytest.fixture
    def data_service(self, mock_connection: MagicMock) -> PostgresDataService:
        """Create PostgresDataService with mock connection."""
        return PostgresDataService(mock_connection)

    def test_init(self, mock_connection: MagicMock) -> None:
        """Test initialization."""
        svc = PostgresDataService(mock_connection)
        assert svc._conn == mock_connection

    def test_get_matched_data_success(
        self, data_service: PostgresDataService, mock_connection: MagicMock
    ) -> None:
        """Test get_matched_data returns DataFrame."""
        mock_connection.execute.return_value = [
            (datetime(2024, 1, 1, 9, 0), "VN30F2401", 1200.0, 100.0),
            (datetime(2024, 1, 1, 9, 1), "VN30F2401", 1201.0, 150.0),
        ]

        df = data_service.get_matched_data("VN30F1M", "2024-01-01", "2024-01-31")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["datetime", "tickersymbol", "price", "quantity"]
        mock_connection.execute.assert_called_once_with(
            MATCHED_QUERY, ("VN30F1M", "2024-01-01", "2024-01-31")
        )

    def test_get_matched_data_empty(
        self, data_service: PostgresDataService, mock_connection: MagicMock
    ) -> None:
        """Test get_matched_data returns empty DataFrame when no data."""
        mock_connection.execute.return_value = []

        df = data_service.get_matched_data("VN30F1M", "2024-01-01", "2024-01-31")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_matched_data_timeout(
        self, data_service: PostgresDataService, mock_connection: MagicMock
    ) -> None:
        """Test get_matched_data handles timeout gracefully."""
        mock_connection.execute.side_effect = TimeoutError("Query timeout")

        df = data_service.get_matched_data("VN30F1M", "2024-01-01", "2024-01-31")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_matched_data_exception(
        self, data_service: PostgresDataService, mock_connection: MagicMock
    ) -> None:
        """Test get_matched_data handles exceptions gracefully."""
        mock_connection.execute.side_effect = Exception("Database error")

        df = data_service.get_matched_data("VN30F1M", "2024-01-01", "2024-01-31")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_matched_data_in_range(
        self, data_service: PostgresDataService, mock_connection: MagicMock
    ) -> None:
        """Test get_matched_data_in_range with datetime parameters."""
        mock_connection.execute.return_value = [
            (datetime(2024, 1, 1, 9, 30), "VN30F2401", 1200.0, 100.0),
        ]

        from_dt = datetime(2024, 1, 1, 9, 0)
        to_dt = datetime(2024, 1, 1, 10, 0)
        df = data_service.get_matched_data_in_range("VN30F1M", from_dt, to_dt)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        mock_connection.execute.assert_called_once_with(
            MATCHED_RANGE_QUERY, ("VN30F1M", from_dt, to_dt)
        )

    def test_get_last_matched_before(
        self, data_service: PostgresDataService, mock_connection: MagicMock
    ) -> None:
        """Test get_last_matched_before returns last tick."""
        mock_connection.execute.return_value = [
            (datetime(2024, 1, 1, 8, 59), "VN30F2401", 1199.0, 50.0),
        ]

        before_dt = datetime(2024, 1, 1, 9, 0)
        df = data_service.get_last_matched_before("VN30F1M", before_dt)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        mock_connection.execute.assert_called_once_with(
            MATCHED_LAST_BEFORE_QUERY, ("VN30F1M", before_dt)
        )

    def test_get_close_data(
        self, data_service: PostgresDataService, mock_connection: MagicMock
    ) -> None:
        """Test get_close_data returns daily close prices."""
        mock_connection.execute.return_value = [
            (datetime(2024, 1, 1), "VN30F2401", 1200.0),
            (datetime(2024, 1, 2), "VN30F2401", 1205.0),
        ]

        df = data_service.get_close_data("VN30F1M", "2024-01-01", "2024-01-31")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["datetime", "tickersymbol", "close"]
        mock_connection.execute.assert_called_once_with(
            CLOSE_QUERY, ("VN30F1M", "2024-01-01", "2024-01-31")
        )

    def test_get_bid_ask_data(
        self, data_service: PostgresDataService, mock_connection: MagicMock
    ) -> None:
        """Test get_bid_ask_data returns bid/ask spread."""
        mock_connection.execute.return_value = [
            (datetime(2024, 1, 1, 9, 0), "VN30F2401", 1199.0, 1201.0, 2.0),
        ]

        df = data_service.get_bid_ask_data("VN30F1M", "2024-01-01", "2024-01-31")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == ["datetime", "tickersymbol", "best-bid", "best-ask", "spread"]
        mock_connection.execute.assert_called_once_with(
            BID_ASK_QUERY, ("VN30F1M", "2024-01-01", "2024-01-31")
        )

    def test_close(self, data_service: PostgresDataService, mock_connection: MagicMock) -> None:
        """Test close delegates to connection."""
        data_service.close()
        mock_connection.close.assert_called_once()


class TestDataServiceFactory:
    """Test factory and singleton functions."""

    @patch("src.database.data_service.get_secrets")
    @patch("src.database.data_service.DatabaseConnection")
    def test_create_data_service(
        self, mock_db_conn_class: MagicMock, mock_get_secrets: MagicMock
    ) -> None:
        """Test create_data_service creates new instance."""
        mock_secrets = MagicMock()
        mock_secrets.db.to_psycopg2_kwargs.return_value = {"host": "localhost"}
        mock_get_secrets.return_value = mock_secrets

        mock_conn_instance = MagicMock()
        mock_db_conn_class.return_value = mock_conn_instance

        svc = create_data_service(max_retries=5, retry_delay=2.0, statement_timeout_ms=10000)

        assert isinstance(svc, PostgresDataService)
        mock_db_conn_class.assert_called_once_with(
            connect_kwargs={"host": "localhost"},
            max_retries=5,
            retry_delay=2.0,
            statement_timeout_ms=10000,
        )
        mock_conn_instance.connect.assert_not_called()  # Connection should not establish until first query

    @patch("src.database.data_service.create_data_service")
    def test_get_data_service_singleton(self, mock_create: MagicMock) -> None:
        """Test get_data_service returns singleton."""
        # Clear cache
        get_data_service.cache_clear()

        mock_svc = MagicMock()
        mock_create.return_value = mock_svc

        svc1 = get_data_service()
        svc2 = get_data_service()

        assert svc1 is svc2
        mock_create.assert_called_once()

    @patch("src.database.data_service.get_secrets")
    @patch("src.database.data_service.DatabaseConnection")
    def test_context_manager(
        self, mock_db_conn_class: MagicMock, mock_get_secrets: MagicMock
    ) -> None:
        """Test data service as context manager."""
        mock_secrets = MagicMock()
        mock_secrets.db.to_psycopg2_kwargs.return_value = {"host": "localhost"}
        mock_get_secrets.return_value = mock_secrets

        mock_conn_instance = MagicMock()
        mock_db_conn_class.return_value = mock_conn_instance

        with create_data_service() as svc:
            assert isinstance(svc, PostgresDataService)

        mock_conn_instance.close.assert_called_once()
