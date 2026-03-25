"""
PostgreSQL implementation of DataServiceBase.
Use DatabaseConnection for lifecycle management, query execution, error handling.
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache

import pandas as pd

from config.secrets import get_secrets
from src.database.base import DataServiceBase
from src.database.connection import DatabaseConnection
from src.database.query import (
    BID_ASK_QUERY,
    CLOSE_QUERY,
    MATCHED_LAST_BEFORE_QUERY,
    MATCHED_QUERY,
    MATCHED_RANGE_QUERY,
)

logger = logging.getLogger(__name__)


class PostgresDataService(DataServiceBase):
    """
    PostgreSQL implementation of DataServiceBase.

    Usage:
        # Recommended: Use factory or singleton for connection management
        svc = get_data_service()
        df = svc.get_matched_data("VN30F1M", "2024-01-01", "2024-12-31")

        # Context manager - auto-close connection
        with get_data_service() as svc:
            df = svc.fetch_ohlcv("VN30F1M", "2024-01-01", "2024-12-31")
    """

    def __init__(self, connection: DatabaseConnection) -> None:
        """
        Args:
            connection: DatabaseConnection instance.
                        Caller is responsible for lifecycle management (connect, close).
        """
        self._conn = connection

    # --- Tick data ---

    def get_matched_data(
        self,
        contract_name: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """Tick-by-tick matched trade data by date range."""
        return self._query_to_df(
            query=MATCHED_QUERY,
            params=(contract_name, from_date, to_date),
            columns=["datetime", "tickersymbol", "price", "quantity"],
            label="matched",
        )

    def get_matched_data_in_range(
        self,
        contract_name: str,
        from_datetime: datetime,
        to_datetime: datetime,
    ) -> pd.DataFrame:
        """Tick data for exact datetime range [from, to)."""
        return self._query_to_df(
            query=MATCHED_RANGE_QUERY,
            params=(contract_name, from_datetime, to_datetime),
            columns=["datetime", "tickersymbol", "price", "quantity"],
            label="matched_range",
        )

    def get_last_matched_before(
        self,
        contract_name: str,
        before_datetime: datetime,
    ) -> pd.DataFrame:
        """Get the last matched trade before a specific datetime."""
        return self._query_to_df(
            query=MATCHED_LAST_BEFORE_QUERY,
            params=(contract_name, before_datetime),
            columns=["datetime", "tickersymbol", "price", "quantity"],
            label="last_matched_before",
        )

    # --- Reference data ---

    def get_close_data(
        self,
        contract_name: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """Daily close price."""
        return self._query_to_df(
            query=CLOSE_QUERY,
            params=(contract_name, from_date, to_date),
            columns=["datetime", "tickersymbol", "close"],
            label="close",
        )

    def get_bid_ask_data(
        self,
        contract_name: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """Best bid/ask spread data."""
        return self._query_to_df(
            query=BID_ASK_QUERY,
            params=(contract_name, from_date, to_date),
            columns=["datetime", "tickersymbol", "best-bid", "best-ask", "spread"],
            label="bid_ask",
        )

    # --- Lifecycle ---

    def close(self) -> None:
        self._conn.close()

    # --- Private ---

    def _query_to_df(
        self,
        query: str,
        params: tuple,
        columns: list[str],
        label: str,
    ) -> pd.DataFrame:
        """
        Execute query, wrap results in DataFrame.
        Return empty DataFrame if query fails - do not raise.
        Caller decides how to handle empty case.
        """
        try:
            rows = self._conn.execute(query, params)
            return pd.DataFrame(rows, columns=columns)
        except TimeoutError:
            logger.error("Query timeout when fetching '%s'.", label)
            return pd.DataFrame()
        except Exception as e:
            logger.error("Query failed when fetching '%s': %s", label, e)
            return pd.DataFrame()


# --- Factory ---


def create_data_service(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    statement_timeout_ms: int = 30_000,
) -> PostgresDataService:
    """
    Factory function - create new PostgresDataService instance with its own DatabaseConnection.

    DO NOT use Singleton - each call creates a new instance with a new connection.
    Use when you need multiple connections (e.g. parallel optimization).

    Usage:
        svc = create_data_service()
        df = svc.fetch_ohlcv("VN30F1M", "2024-01-01", "2024-12-31")
        svc.close()

        # Hoặc context manager
        with create_data_service() as svc:
            df = svc.fetch_ohlcv(...)
    """
    secrets = get_secrets()
    conn = DatabaseConnection(
        connect_kwargs=secrets.db.to_psycopg2_kwargs(),
        max_retries=max_retries,
        retry_delay=retry_delay,
        statement_timeout_ms=statement_timeout_ms,
    )

    # Lazy connection: conn.connect() will be called automatically by ensure_connected() upon first query
    return PostgresDataService(conn)


@lru_cache(maxsize=1)
def get_data_service() -> PostgresDataService:
    """
    Lazy singleton - use for single-thread contexts.
    (backtest, single paper trading instance).

    DO NOT use in parallel contexts -
    each worker should call create_data_service() to get its own instance.

    Raises:
        ConnectionError: If the connection could not be established after retries.
    """
    return create_data_service()
