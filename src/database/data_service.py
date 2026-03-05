"""
This module provides a DataService class that interacts with a database to perform CRUD operations.
It uses psycopg2 to connect to a PostgreSQL database with comprehensive error handling.
"""

import logging
import time
from typing import Callable, List, Tuple

import pandas as pd
import psycopg2
from psycopg2 import (
    InterfaceError,
    OperationalError,
)
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from config.config import DB_CONFIG
from src.database.query import BID_ASK_QUERY, CLOSE_QUERY, MATCHED_QUERY

logger = logging.getLogger(__name__)


class DataService:
    """
    A service class to perform CRUD operations on a PostgreSQL database.
    """

    connection = None
    connection_string = None

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,  # Initial delay in seconds, doubles each retry
    ):
        """
        Initializes the DataService instance by establishing a database connection.

        Args:
            max_retries: Maximum number of connection retry attempts
            retry_delay: Initial delay between retries (seconds), doubles each retry
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Validate DB config
        self._validate_config()

        self.connection_string = (
            f"host={DB_CONFIG['host']} "
            f"port={DB_CONFIG['port']} "
            f"user={DB_CONFIG['user']} "
            f"password={DB_CONFIG['password']} "
            f"dbname={DB_CONFIG['database']} "
            f"connect_timeout=10"
        )
        # self._connect()  # Lazy connection on first query

    def _validate_config(self) -> None:
        """Validate database configuration."""
        required_keys = ["host", "port", "user", "password", "database"]
        missing_keys = [key for key in required_keys if not DB_CONFIG.get(key)]

        if missing_keys:
            raise ValueError(
                f"Missing database configuration keys: {', '.join(missing_keys)}"
            )

    def _connect(self) -> None:
        """
        Establishes a connection to the database.
        """
        for attempt in range(self.max_retries):
            try:
                self.connection = psycopg2.connect(
                    dsn=self.connection_string,
                )
                # Set isolation level
                self.connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

                logger.info("Database connection established (attempt %s)", attempt + 1)
                return

            except (OperationalError, InterfaceError) as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        "Connection attempt %s failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Failed to connect after %s attempts", self.max_retries
                    )

            except Exception as e:
                logger.error("Unexpected error during connection: %s", e)
                break

        # If we get here, connection failed
        raise ConnectionError(
            "Could not establish database connection after multiple attempts"
        )

    def _ensure_connection(self) -> None:
        """
        Ensures that the database connection is active. If not, reconnects.
        """
        if self.connection is None:
            logger.info("Connection is None, attempting to connect...")
            self._connect()
            return

        try:
            # Test if connection is alive by executing a simple query
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        except (OperationalError, InterfaceError) as e:
            # Connection is dead, reconnect
            logger.warning("Connection lost: %s. Reconnecting...", e)
            try:
                self.connection.close()
            except Exception:
                pass  # Ignore errors during close

            self._connect()

    def _execute_query(
        self,
        query: str,
        params: Tuple,
    ) -> List[Tuple]:
        """
        Execute a query and return raw rows.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List[Tuple]: Query results as a list of tuples
        """
        self._ensure_connection()
        if self.connection is None:
            raise ConnectionError("No active database connection")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return list(cursor)

        except psycopg2.extensions.QueryCanceledError as e:
            raise TimeoutError(f"Query timed out: {e}")

        except Exception as e:
            raise RuntimeError(f"Database query failed: {e}")

    def _query_to_df(
        self,
        query: str,
        params: Tuple,
        columns: List[str],
        label: str,
    ) -> pd.DataFrame:
        """
        Execute a query and return results as a DataFrame.

        Args:
            query:   SQL query string.
            params:  Query parameters.
            columns: Column names for the resulting DataFrame.
            label:   Human-readable name used in error log messages.

        Returns:
            pd.DataFrame with *columns*, empty on any error.
        """
        try:
            rows = self._execute_query(query, params)
            return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            logger.error("Unexpected error fetching %s data: %s", label, e)
            return pd.DataFrame()

    def get_matched_data(
        self,
        from_date: str,
        to_date: str,
        contract_name: str,
    ) -> pd.DataFrame:
        """Retrieve matched (trade) tick data for *contract_name*."""
        return self._query_to_df(
            MATCHED_QUERY,
            (contract_name, from_date, to_date),
            ["datetime", "tickersymbol", "price", "quantity"],
            label="matched",
        )

    def get_bid_ask_data(
        self,
        from_date: str,
        to_date: str,
        contract_name: str,
    ) -> pd.DataFrame:
        """Retrieve best bid/ask spread data for *contract_name*."""
        return self._query_to_df(
            BID_ASK_QUERY,
            (contract_name, from_date, to_date),
            ["datetime", "tickersymbol", "best-bid", "best-ask", "spread"],
            label="bid-ask",
        )

    def get_close_data(
        self,
        from_date: str,
        to_date: str,
        contract_name: str,
    ) -> pd.DataFrame:
        """Retrieve daily close price data for *contract_name*."""
        return self._query_to_df(
            CLOSE_QUERY,
            (contract_name, from_date, to_date),
            ["date", "tickersymbol", "close"],
            label="close",
        )

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error("Error closing connection: %s", e)
            finally:
                self.connection = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Create a singleton instance of DataService
data_service = DataService()


def fetch_in_chunks(
    fetch_method: Callable,
    contract_name: str,
    start_date: str,
    end_date: str,
    chunk_size: int = 30,  # Number of days per chunk
) -> pd.DataFrame:
    """
    Fetch data in chunks to avoid database timeouts/errors with large datasets.

    Args:
        fetch_method (Callable): The method to fetch data (e.g., get_matched_data).
        contract_name (str): The type of contract to filter data.
        start_date (str): The start date for data retrieval.
        end_date (str): The end date for data retrieval.
        chunk_size (int): The number of days per chunk.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated results from all chunks.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    chunks = []
    current = start

    logger.info(
        "Fetching data in chunks (%s) from %s to %s...",
        chunk_size,
        start.date(),
        end.date(),
    )

    while current <= end:
        next_chunk = current + pd.Timedelta(days=chunk_size)  # days, not nanoseconds
        chunk_end = min(next_chunk, end)

        s_date = current.strftime("%Y-%m-%d")
        e_date = chunk_end.strftime("%Y-%m-%d")

        logger.info("Fetching chunk: %s to %s", s_date, e_date)

        try:
            df_chunk = fetch_method(
                from_date=s_date,
                to_date=e_date,
                contract_name=contract_name,
            )
            chunks.append(df_chunk)
        except Exception as e:
            logger.error("Skipping chunk due to error: %s", e)

        # Move to next day
        current = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def fetch_and_merge_data(
    contract_name: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch close and matched data and merge them.

    Args:
        contract_name (str): The type of contract to filter data.
        start_date (str): The start date for data retrieval.
        end_date (str): The end date for data retrieval.

    Returns:
        pd.DataFrame: A DataFrame containing the merged data.
    """
    # 1. Fetch Close Data
    logger.info("Loading close price data...")
    close_data = data_service.get_close_data(
        from_date=start_date,
        to_date=end_date,
        contract_name=contract_name,
    )
    if not close_data.empty:
        close_data["date"] = pd.to_datetime(close_data["date"]).dt.date
    logger.info("Close data loaded: %s rows.", len(close_data))

    # 2. Fetch Matched Data (Chunked)
    logger.info("Loading matched price data...")
    matched_data = fetch_in_chunks(
        data_service.get_matched_data,
        contract_name,
        start_date,
        end_date,
        chunk_size=30,  # Number of days per chunk
    )
    if not matched_data.empty:
        matched_data = matched_data.astype({"price": float})
        matched_data["datetime"] = pd.to_datetime(matched_data["datetime"])
    logger.info("Matched data loaded: %s rows.", len(matched_data))

    # 3. Merge Data
    logger.info("Merging data...")

    if matched_data.empty:
        logger.warning("No matched data found.")
        return pd.DataFrame()

    # Matched data is the primary source
    data = matched_data.sort_values(["datetime", "tickersymbol"])
    data["date"] = data["datetime"].dt.date

    # Merge with Close Data
    data = pd.merge(
        data,
        close_data,
        on=["date", "tickersymbol"],
        how="left",
        sort=True,
    )

    return data
