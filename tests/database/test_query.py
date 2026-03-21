"""
Test suite for SQL query constants.
"""

from src.database.query import (
    BID_ASK_QUERY,
    CLOSE_QUERY,
    MATCHED_LAST_BEFORE_QUERY,
    MATCHED_QUERY,
    MATCHED_RANGE_QUERY,
)


class TestQueryConstants:
    """Test SQL query string constants."""

    def test_matched_query_exists(self) -> None:
        """Test MATCHED_QUERY is defined."""
        assert isinstance(MATCHED_QUERY, str)
        assert len(MATCHED_QUERY) > 0
        assert "SELECT" in MATCHED_QUERY
        assert "quote.matched" in MATCHED_QUERY
        assert "quote.futurecontractcode" in MATCHED_QUERY

    def test_matched_query_has_parameters(self) -> None:
        """Test MATCHED_QUERY has correct parameter placeholders."""
        assert MATCHED_QUERY.count("%s") == 3  # futurecode, start_date, end_date

    def test_matched_range_query_exists(self) -> None:
        """Test MATCHED_RANGE_QUERY is defined."""
        assert isinstance(MATCHED_RANGE_QUERY, str)
        assert "SELECT" in MATCHED_RANGE_QUERY
        assert "quote.matched" in MATCHED_RANGE_QUERY
        assert "m.datetime >=" in MATCHED_RANGE_QUERY
        assert "m.datetime <" in MATCHED_RANGE_QUERY

    def test_matched_range_query_has_parameters(self) -> None:
        """Test MATCHED_RANGE_QUERY has correct parameter placeholders."""
        assert MATCHED_RANGE_QUERY.count("%s") == 3  # futurecode, from_datetime, to_datetime

    def test_matched_last_before_query_exists(self) -> None:
        """Test MATCHED_LAST_BEFORE_QUERY is defined."""
        assert isinstance(MATCHED_LAST_BEFORE_QUERY, str)
        assert "SELECT" in MATCHED_LAST_BEFORE_QUERY
        assert "LIMIT 1" in MATCHED_LAST_BEFORE_QUERY
        assert "ORDER BY m.datetime DESC" in MATCHED_LAST_BEFORE_QUERY

    def test_matched_last_before_query_has_parameters(self) -> None:
        """Test MATCHED_LAST_BEFORE_QUERY has correct parameter placeholders."""
        assert MATCHED_LAST_BEFORE_QUERY.count("%s") == 2  # futurecode, before_datetime

    def test_close_query_exists(self) -> None:
        """Test CLOSE_QUERY is defined."""
        assert isinstance(CLOSE_QUERY, str)
        assert "SELECT" in CLOSE_QUERY
        assert "quote.close" in CLOSE_QUERY

    def test_close_query_has_parameters(self) -> None:
        """Test CLOSE_QUERY has correct parameter placeholders."""
        assert CLOSE_QUERY.count("%s") == 3  # futurecode, start_date, end_date

    def test_bid_ask_query_exists(self) -> None:
        """Test BID_ASK_QUERY is defined."""
        assert isinstance(BID_ASK_QUERY, str)
        assert "SELECT" in BID_ASK_QUERY
        assert "quote.bidprice" in BID_ASK_QUERY
        assert "quote.askprice" in BID_ASK_QUERY
        assert "b.depth = 1" in BID_ASK_QUERY

    def test_bid_ask_query_has_parameters(self) -> None:
        """Test BID_ASK_QUERY has correct parameter placeholders."""
        assert BID_ASK_QUERY.count("%s") == 3  # futurecode, start_date, end_date

    def test_all_queries_have_order_by(self) -> None:
        """Test all queries have ORDER BY clause for consistent results."""
        queries = [
            MATCHED_QUERY,
            MATCHED_RANGE_QUERY,
            MATCHED_LAST_BEFORE_QUERY,
            CLOSE_QUERY,
            BID_ASK_QUERY,
        ]
        for query in queries:
            assert "ORDER BY" in query, f"Query missing ORDER BY: {query[:50]}"
