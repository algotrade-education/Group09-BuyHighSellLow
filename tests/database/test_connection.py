"""
Test suite for DatabaseConnection class.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import psycopg2
import pytest
from psycopg2 import OperationalError

from src.database.connection import DatabaseConnection


class TestDatabaseConnection:
    """Test DatabaseConnection lifecycle and error handling."""

    @pytest.fixture
    def mock_connect_kwargs(self) -> dict[str, Any]:
        """Mock connection parameters."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
        }

    @pytest.fixture
    def db_conn(self, mock_connect_kwargs: dict[str, Any]) -> DatabaseConnection:
        """Create DatabaseConnection instance."""
        return DatabaseConnection(
            connect_kwargs=mock_connect_kwargs,
            max_retries=3,
            retry_delay=0.1,
            statement_timeout_ms=5000,
        )

    def test_init(self, mock_connect_kwargs: dict[str, Any]) -> None:
        """Test initialization with parameters."""
        conn = DatabaseConnection(
            connect_kwargs=mock_connect_kwargs,
            max_retries=5,
            retry_delay=2.0,
            statement_timeout_ms=10000,
        )
        assert conn._kwargs == mock_connect_kwargs
        assert conn._max_retries == 5
        assert conn._retry_delay == 2.0
        assert conn._timeout_ms == 10000
        assert conn._conn is None

    @patch("src.database.connection.psycopg2.connect")
    def test_connect_success(self, mock_connect: MagicMock, db_conn: DatabaseConnection) -> None:
        """Test successful connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db_conn.connect()

        mock_connect.assert_called_once()
        mock_conn.set_isolation_level.assert_called_once()
        mock_cursor.execute.assert_called_once()
        assert db_conn._conn is not None

    @patch("src.database.connection.psycopg2.connect")
    def test_connect_retry_on_operational_error(
        self, mock_connect: MagicMock, db_conn: DatabaseConnection
    ) -> None:
        """Test connection retry on OperationalError."""
        mock_connect.side_effect = [
            OperationalError("Connection refused"),
            OperationalError("Connection refused"),
            MagicMock(),
        ]

        db_conn.connect()

        assert mock_connect.call_count == 3

    @patch("src.database.connection.psycopg2.connect")
    def test_connect_fails_after_max_retries(
        self, mock_connect: MagicMock, db_conn: DatabaseConnection
    ) -> None:
        """Test connection failure after max retries."""
        mock_connect.side_effect = OperationalError("Connection refused")

        with pytest.raises(ConnectionError, match="Could not connect"):
            db_conn.connect()

        assert mock_connect.call_count == 3

    @patch("src.database.connection.psycopg2.connect")
    def test_connect_unexpected_error(
        self, mock_connect: MagicMock, db_conn: DatabaseConnection
    ) -> None:
        """Test unexpected error during connection."""
        mock_connect.side_effect = ValueError("Unexpected error")

        with pytest.raises(ValueError):
            db_conn.connect()

    @patch("src.database.connection.psycopg2.connect")
    def test_ensure_connected_when_not_connected(
        self, mock_connect: MagicMock, db_conn: DatabaseConnection
    ) -> None:
        """Test ensure_connected creates connection if none exists."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db_conn.ensure_connnected()

        mock_connect.assert_called_once()
        assert db_conn._conn is not None

    @patch("src.database.connection.psycopg2.connect")
    def test_ensure_connected_when_connection_lost(
        self, mock_connect: MagicMock, db_conn: DatabaseConnection
    ) -> None:
        """Test ensure_connected reconnects if connection is lost."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        # First connection succeeds
        mock_connect.return_value = mock_conn
        db_conn.connect()

        # Health check fails
        mock_cursor.execute.side_effect = OperationalError("Connection lost")

        # Second connection succeeds
        mock_conn2 = MagicMock()
        mock_cursor2 = MagicMock()
        mock_conn2.cursor.return_value.__enter__.return_value = mock_cursor2
        mock_connect.return_value = mock_conn2

        db_conn.ensure_connnected()

        assert mock_connect.call_count == 2

    @patch("src.database.connection.psycopg2.connect")
    def test_execute_success(self, mock_connect: MagicMock, db_conn: DatabaseConnection) -> None:
        """Test successful query execution."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter([("row1",), ("row2",)])
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db_conn.connect()
        results = db_conn.execute("SELECT * FROM test", ())

        assert results == [("row1",), ("row2",)]
        mock_cursor.execute.assert_called()

    @patch("src.database.connection.psycopg2.connect")
    def test_execute_without_connection(
        self, mock_connect: MagicMock, db_conn: DatabaseConnection
    ) -> None:
        """Test execute auto-connects if not connected."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter([])
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        results = db_conn.execute("SELECT 1", ())

        mock_connect.assert_called_once()
        assert results == []

    @patch("src.database.connection.psycopg2.connect")
    def test_execute_query_timeout(self, mock_connect: MagicMock) -> None:
        """Test query timeout handling."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter([])
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Create and connect (with no timeout set to avoid issues)
        db_conn = DatabaseConnection(
            connect_kwargs={"host": "localhost"},
            max_retries=3,
            retry_delay=0.1,
            statement_timeout_ms=0,  # Disable timeout for connect
        )
        db_conn.connect()

        # Mock ensure_connected to skip health check, then make execute raise timeout
        with patch.object(db_conn, "ensure_connnected"):
            mock_cursor.execute.side_effect = psycopg2.extensions.QueryCanceledError("timeout")

            with pytest.raises(TimeoutError, match="timed out"):
                db_conn.execute("SELECT * FROM large_table", ())

    @patch("src.database.connection.psycopg2.connect")
    def test_close(self, mock_connect: MagicMock, db_conn: DatabaseConnection) -> None:
        """Test connection close."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        db_conn.connect()
        db_conn.close()

        mock_conn.close.assert_called_once()
        assert db_conn._conn is None

    def test_close_without_connection(self, db_conn: DatabaseConnection) -> None:
        """Test close when no connection exists."""
        db_conn.close()  # Should not raise

    @patch("src.database.connection.psycopg2.connect")
    def test_context_manager(
        self, mock_connect: MagicMock, mock_connect_kwargs: dict[str, Any]
    ) -> None:
        """Test context manager protocol."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        with DatabaseConnection(mock_connect_kwargs) as conn:
            assert conn._conn is not None

        mock_conn.close.assert_called_once()

    @patch("src.database.connection.psycopg2.connect")
    def test_safe_close_handles_exception(
        self, mock_connect: MagicMock, db_conn: DatabaseConnection
    ) -> None:
        """Test _safe_close handles exceptions gracefully."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.close.side_effect = Exception("Close error")
        mock_connect.return_value = mock_conn

        db_conn.connect()
        db_conn.close()  # Should not raise

        assert db_conn._conn is None
