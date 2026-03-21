"""
Connection management for the PostgreSQL database.
This module provides functions to establish and manage connections to the database,
ensuring efficient and secure access to the data.
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import Any

import psycopg2
from psycopg2 import InterfaceError, OperationalError
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(__name__)

# Default statement timeout -- query will be killed if it runs longer than this (in milliseconds)
DEFAULT_STATEMENT_TIMEOUT_MS = 30_000  # 30 seconds


class DatabaseConnection:
    """
    Manages a single PostgreSQL connection

    Usage:
        conn = DatabaseConnection(secrets.db.to_psycopg2_kwarg())
        conn.connect()

        rows = conn.execute("SELECT 1", ())
        conn.close()
    """

    def __init__(
        self,
        connect_kwargs: dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
    ) -> None:
        """
        Args:
            connect_kwargs: Keyword arguments for establishing the database connection
            max_retries: Maximum number of times to retry a failed connection
            retry_delay: Initial delay between retries (seconds), double every try.
            statement_timeout_ms: TImeout for each query, 0 = no timeout.
        """
        self._kwargs = connect_kwargs
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._timeout_ms = statement_timeout_ms
        self._conn: psycopg2.extensions.connection | None = None

    # --- Public APIs ---

    def connect(self) -> None:
        """
        Establish a connection to the database.

        Raises:
            ConnectionError: If the connection could not be established after retries.
        """
        for attempt in range(self._max_retries):
            try:
                # Reset before each attempt
                self._conn = None

                self._conn = psycopg2.connect(**self._kwargs)
                self._conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

                # Set statement timeout for this session
                if self._timeout_ms > 0:
                    with self._conn.cursor() as cur:
                        cur.execute(f"SET statement_timeout = {self._timeout_ms}")

                logger.info("DB Connected (attempt %d)", attempt + 1)
                return

            except (OperationalError, InterfaceError) as e:
                # Reset connection on failure and retry
                self._conn = None

                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        "DB connection failed (attempt %d/%d): %s. Retrying in %.1f seconds...",
                        attempt + 1,
                        self._max_retries,
                        str(e),
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "DB connection failed after %d attempts: %s",
                        self._max_retries,
                        str(e),
                    )
                    raise ConnectionError(
                        f"Could not connect to the database after {self._max_retries} attempts."
                    ) from e

            except Exception as e:
                self._conn = None
                logger.error("Unexpected error during DB connection: %s", str(e))
                raise

    def ensure_connected(self) -> None:
        """
        Check connection health and reconnect if necessary.
        Call before executing queries to ensure the connection is alive.
        """
        if self._conn is None:
            logger.debug("No active DB connection. Attempting to connect...")
            self.connect()
            return

        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (OperationalError, InterfaceError) as e:
            logger.warning("DB connection lost: %s. Reconnecting...", str(e))
            self._safe_close()
            self.connect()

    def execute(self, query: str, params: tuple = ()) -> list[tuple]:
        """
        Execute a query and return the results.

        Args:
            query: SQL query to execute
            params: Parameters for the SQL query

        Returns:
            List of tuples containing the query results.

        Raises:
            ConnectionError: If the connection is not established.
            psycopg2.Error: If the query execution fails.
        """
        self.ensure_connected()

        if self._conn is None:
            raise ConnectionError("Not connected to the database.")

        try:
            with self._conn.cursor() as cur:
                cur.execute(query, params)
                return list(cur)

        except psycopg2.extensions.QueryCanceledError as e:
            raise TimeoutError("The database query timed out.") from e

        except Exception as e:
            logger.error("Error executing query: %s", str(e))
            raise

    def close(self) -> None:
        """Close the database connection."""
        self._safe_close()
        logger.info("DB Connection closed.")

    # --- Context Manager ---

    def __enter__(self) -> DatabaseConnection:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- Private Helpers ---

    def _safe_close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None
