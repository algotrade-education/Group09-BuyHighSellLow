"""
Integration tests for database module.
These tests verify the interaction between components.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.database import create_data_service, get_data_service
from src.database.connection import DatabaseConnection


class TestDatabaseIntegration:
    """Integration tests for database components."""

    @patch("src.database.data_service.get_secrets")
    @patch("src.database.connection.psycopg2.connect")
    def test_create_and_query_data_service(
        self, mock_connect: MagicMock, mock_get_secrets: MagicMock
    ) -> None:
        """Test creating data service and executing query."""
        # Setup mocks
        mock_secrets = MagicMock()
        mock_secrets.db.to_psycopg2_kwargs.return_value = {"host": "localhost"}
        mock_get_secrets.return_value = mock_secrets

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter(
            [(datetime(2024, 1, 1, 9, 0), "VN30F2401", 1200.0, 100.0)]
        )
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Create service and query
        svc = create_data_service()
        df = svc.get_matched_data("VN30F1M", "2024-01-01", "2024-01-31")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        mock_connect.assert_called_once()

    @patch("src.database.data_service.get_secrets")
    @patch("src.database.connection.psycopg2.connect")
    def test_singleton_pattern(self, mock_connect: MagicMock, mock_get_secrets: MagicMock) -> None:
        """Test get_data_service returns same instance."""
        get_data_service.cache_clear()

        mock_secrets = MagicMock()
        mock_secrets.db.to_psycopg2_kwargs.return_value = {"host": "localhost"}
        mock_get_secrets.return_value = mock_secrets

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        svc1 = get_data_service()
        svc2 = get_data_service()

        assert svc1 is svc2
        assert mock_connect.call_count == 1

    @patch("src.database.connection.psycopg2.connect")
    def test_connection_retry_and_recovery(self, mock_connect: MagicMock) -> None:
        """Test connection retry mechanism."""
        from psycopg2 import OperationalError

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Fail twice, then succeed
        mock_connect.side_effect = [
            OperationalError("Connection refused"),
            OperationalError("Connection refused"),
            mock_conn,
        ]

        conn = DatabaseConnection(
            connect_kwargs={"host": "localhost"},
            max_retries=3,
            retry_delay=0.01,
        )
        conn.connect()

        assert conn._conn is not None
        assert mock_connect.call_count == 3

    @patch("src.database.data_service.get_secrets")
    @patch("src.database.connection.psycopg2.connect")
    def test_full_workflow_with_context_manager(
        self, mock_connect: MagicMock, mock_get_secrets: MagicMock
    ) -> None:
        """Test complete workflow using context manager."""
        mock_secrets = MagicMock()
        mock_secrets.db.to_psycopg2_kwargs.return_value = {"host": "localhost"}
        mock_get_secrets.return_value = mock_secrets

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter(
            [
                (datetime(2024, 1, 1, 9, 0), "VN30F2401", 1200.0, 100.0),
                (datetime(2024, 1, 1, 9, 1), "VN30F2401", 1201.0, 150.0),
            ]
        )
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with create_data_service() as svc:
            df = svc.get_matched_data("VN30F1M", "2024-01-01", "2024-01-31")
            assert len(df) == 2

        mock_conn.close.assert_called_once()
