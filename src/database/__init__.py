"""
Database module - PostgreSQL data service implementation.

This module provides interfaces and implementations for accessing market data
from PostgreSQL database, including tick data, OHLCV aggregation, and reference data.

Main Components:
    - DataServiceBase: Abstract interface for data services
    - PostgresDataService: PostgreSQL implementation
    - DatabaseConnection: Connection management with retry logic
    - Query constants: SQL queries for data fetching

Usage:
    # Singleton pattern (recommended for single-threaded contexts)
    from src.database import get_data_service

    svc = get_data_service()
    df = svc.get_matched_data("VN30F1M", "2024-01-01", "2024-12-31")

    # Factory pattern (for parallel contexts)
    from src.database import create_data_service

    with create_data_service() as svc:
        df = svc.fetch_ohlcv("VN30F1M", "2024-01-01", "2024-12-31")
"""

from src.database.base import DataServiceBase
from src.database.connection import DatabaseConnection
from src.database.data_service import PostgresDataService, create_data_service, get_data_service
from src.database.query import (
    BID_ASK_QUERY,
    CLOSE_QUERY,
    MATCHED_LAST_BEFORE_QUERY,
    MATCHED_QUERY,
    MATCHED_RANGE_QUERY,
)

__all__ = [
    # Base interface
    "DataServiceBase",
    # Connection management
    "DatabaseConnection",
    # Service implementation
    "PostgresDataService",
    "create_data_service",
    "get_data_service",
    # SQL queries
    "MATCHED_QUERY",
    "MATCHED_RANGE_QUERY",
    "MATCHED_LAST_BEFORE_QUERY",
    "CLOSE_QUERY",
    "BID_ASK_QUERY",
]
