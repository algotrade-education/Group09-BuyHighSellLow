"""
Pytest fixtures for database tests.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def sample_tick_data() -> pd.DataFrame:
    """Sample tick data for testing."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01 09:00", periods=10, freq="1min"),
            "tickersymbol": ["VN30F2401"] * 10,
            "price": [
                1200.0,
                1201.0,
                1199.0,
                1202.0,
                1203.0,
                1200.0,
                1198.0,
                1201.0,
                1204.0,
                1205.0,
            ],
            "quantity": [100.0, 150.0, 170.0, 200.0, 250.0, 270.0, 290.0, 320.0, 350.0, 380.0],
        }
    )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Sample OHLCV data for testing."""
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=5, freq="1D"),
            "open": [1200.0, 1205.0, 1210.0, 1208.0, 1215.0],
            "high": [1210.0, 1215.0, 1220.0, 1218.0, 1225.0],
            "low": [1195.0, 1200.0, 1205.0, 1203.0, 1210.0],
            "close": [1205.0, 1210.0, 1208.0, 1215.0, 1220.0],
            "volume": [1000.0, 1200.0, 1100.0, 1300.0, 1400.0],
        }
    )


@pytest.fixture
def mock_db_connection() -> MagicMock:
    """Mock DatabaseConnection for testing."""
    conn = MagicMock()
    conn.execute.return_value = []
    return conn


@pytest.fixture
def mock_psycopg2_connection() -> MagicMock:
    """Mock psycopg2 connection object."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__iter__.return_value = iter([])
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn
