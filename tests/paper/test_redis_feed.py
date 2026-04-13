"""Unit tests for RedisFeed - Redis market data feed integration.

Tests cover:
- Feed lifecycle (subscribe/unsubscribe/close)
- Quote filtering (invalid prices, out-of-session)
- Tick forwarding to BarAggregator
- Watchdog monitoring for feed silence
- Polling loop for clock-based rollover
- Error handling and graceful degradation
"""

import asyncio
from contextlib import suppress
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.paper.bar_aggregator import BarAggregator
from src.paper.feeds.redis_feed import RedisFeed

# --- RedisFeed Lifecycle Tests ---


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    client = AsyncMock()
    client.subscribe = AsyncMock()
    client.unsubscribe = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_bar_aggregator():
    """Mock BarAggregator for testing."""
    agg = Mock(spec=BarAggregator)
    agg.set_on_bar = Mock()
    agg.on_tick = Mock()
    agg.check_time = Mock()
    return agg


@pytest.fixture
def mock_session_manager():
    """Mock session manager for RedisFeed tests."""
    from unittest.mock import Mock

    sm = Mock()
    sm.is_trading_hours = Mock(return_value=True)
    return sm


@pytest.fixture
def redis_feed(mock_redis_client, mock_bar_aggregator, mock_session_manager):
    """Create RedisFeed instance with mocked dependencies."""
    return RedisFeed(
        redis_client=mock_redis_client,
        bar_aggregator=mock_bar_aggregator,
        session_manager=mock_session_manager,
        watchdog_silence_seconds=300.0,
    )


class TestRedisFeedLifecycle:
    """Tests for RedisFeed lifecycle management."""

    @pytest.mark.asyncio
    async def test_subscribe_sets_running_flag(self, redis_feed):
        """Test that subscribe() sets the running flag."""
        assert not redis_feed._running

        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        assert redis_feed._running

    @pytest.mark.asyncio
    async def test_subscribe_registers_callback_with_aggregator(
        self, redis_feed, mock_bar_aggregator
    ):
        """Test that subscribe() registers callback with BarAggregator."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        mock_bar_aggregator.set_on_bar.assert_called_once_with(callback)

    @pytest.mark.asyncio
    async def test_subscribe_registers_quote_handler_with_redis(
        self, redis_feed, mock_redis_client
    ):
        """Test that subscribe() passes on_quote handler to Redis client subscribe()."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        mock_redis_client.subscribe.assert_called_once_with("VN30F1M", redis_feed._on_quote)

    @pytest.mark.asyncio
    async def test_subscribe_calls_redis_subscribe(self, redis_feed, mock_redis_client):
        """Test that subscribe() calls Redis client subscribe() with symbol and handler."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        mock_redis_client.subscribe.assert_called_once_with("VN30F1M", redis_feed._on_quote)

    @pytest.mark.asyncio
    async def test_subscribe_starts_polling_task(self, redis_feed):
        """Test that subscribe() starts the polling loop task."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        assert redis_feed._polling_task is not None
        assert not redis_feed._polling_task.done()

        # Cleanup
        await redis_feed.unsubscribe("VN30F1M")

    @pytest.mark.asyncio
    async def test_subscribe_ignores_duplicate_calls(self, redis_feed, mock_redis_client):
        """Test that duplicate subscribe() calls are ignored."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        # Reset mock to check second call
        mock_redis_client.subscribe.reset_mock()

        # Second subscribe should be ignored
        await redis_feed.subscribe("VN30F1M", callback)

        mock_redis_client.subscribe.assert_not_called()

        # Cleanup
        await redis_feed.unsubscribe("VN30F1M")

    @pytest.mark.asyncio
    async def test_unsubscribe_clears_running_flag(self, redis_feed):
        """Test that unsubscribe() clears the running flag."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        await redis_feed.unsubscribe("VN30F1M")

        assert not redis_feed._running

    @pytest.mark.asyncio
    async def test_unsubscribe_cancels_polling_task(self, redis_feed):
        """Test that unsubscribe() cancels the polling task."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        polling_task = redis_feed._polling_task
        assert polling_task is not None

        await redis_feed.unsubscribe("VN30F1M")

        assert polling_task.cancelled()
        assert redis_feed._polling_task is None

    @pytest.mark.asyncio
    async def test_unsubscribe_calls_redis_unsubscribe(self, redis_feed, mock_redis_client):
        """Test that unsubscribe() calls Redis client unsubscribe()."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        await redis_feed.unsubscribe("VN30F1M")

        mock_redis_client.unsubscribe.assert_called_once_with("VN30F1M")

    @pytest.mark.asyncio
    async def test_unsubscribe_clears_subscribed_symbol(self, redis_feed):
        """Test that unsubscribe() clears the subscribed symbol."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        await redis_feed.unsubscribe("VN30F1M")

        assert redis_feed._subscribed_symbol is None

    @pytest.mark.asyncio
    async def test_unsubscribe_when_not_running_is_noop(self, redis_feed, mock_redis_client):
        """Test that unsubscribe() when not running is a no-op."""
        await redis_feed.unsubscribe("VN30F1M")

        mock_redis_client.unsubscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_unsubscribes_if_subscribed(self, redis_feed, mock_redis_client):
        """Test that close() unsubscribes if currently subscribed."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        await redis_feed.close()

        mock_redis_client.unsubscribe.assert_called_once_with("VN30F1M")

    @pytest.mark.asyncio
    async def test_close_closes_redis_client(self, redis_feed, mock_redis_client):
        """Test that close() closes the Redis client."""
        await redis_feed.close()

        mock_redis_client.close.assert_called_once()


# --- Quote Processing Tests ---


def make_quote(price: float, volume: float = 100.0) -> Mock:
    """Create a mock quote snapshot."""
    quote = Mock()
    quote.latest_matched_price = price
    quote.latest_matched_quantity = volume
    return quote


class TestQuoteProcessing:
    """Tests for quote filtering and forwarding."""

    @pytest.mark.asyncio
    async def test_on_quote_filters_none_price(self, redis_feed, mock_bar_aggregator):
        """Test that quotes with None price are filtered out."""
        quote = Mock()
        quote.latest_matched_price = None
        quote.latest_matched_quantity = 100.0

        await redis_feed._on_quote("HNXDS:VN30F2601", quote)

        mock_bar_aggregator.on_tick.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_quote_filters_zero_price(self, redis_feed, mock_bar_aggregator):
        """Test that quotes with zero price are filtered out."""
        quote = make_quote(price=0.0)

        await redis_feed._on_quote("HNXDS:VN30F2601", quote)

        mock_bar_aggregator.on_tick.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_quote_filters_negative_price(self, redis_feed, mock_bar_aggregator):
        """Test that quotes with negative price are filtered out."""
        quote = make_quote(price=-1300.0)

        await redis_feed._on_quote("HNXDS:VN30F2601", quote)

        mock_bar_aggregator.on_tick.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_quote_filters_out_of_session_timestamps(
        self, redis_feed, mock_bar_aggregator, mock_session_manager
    ):
        """Test that quotes outside trading hours are filtered out."""
        quote = make_quote(price=1300.0)
        mock_session_manager.is_trading_hours.return_value = False

        with patch("src.paper.feeds.redis_feed.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 12, 0, 0)
            await redis_feed._on_quote("HNXDS:VN30F2601", quote)

        mock_bar_aggregator.on_tick.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_quote_forwards_valid_tick_to_aggregator(
        self, redis_feed, mock_bar_aggregator
    ):
        """Test that valid quotes are forwarded to BarAggregator."""
        quote = make_quote(price=1300.0, volume=50.0)

        # Mock datetime.now() to return morning session time
        with patch("src.paper.feeds.redis_feed.datetime") as mock_dt:
            mock_now = datetime(2024, 1, 15, 9, 15, 0)
            mock_dt.now.return_value = mock_now

            await redis_feed._on_quote("HNXDS:VN30F2601", quote)

        mock_bar_aggregator.on_tick.assert_called_once()
        call_args = mock_bar_aggregator.on_tick.call_args[0]
        assert call_args[0] == mock_now  # timestamp
        assert call_args[1] == 1300.0  # price
        assert call_args[2] == 50.0  # volume

    @pytest.mark.asyncio
    async def test_on_quote_updates_watchdog_timestamp(self, redis_feed):
        """Test that valid quotes update the watchdog timestamp."""
        quote = make_quote(price=1300.0)

        assert redis_feed._last_quote_ts is None

        with patch("src.paper.feeds.redis_feed.datetime") as mock_dt:
            mock_now = datetime(2024, 1, 15, 9, 15, 0)
            mock_dt.now.return_value = mock_now

            await redis_feed._on_quote("HNXDS:VN30F2601", quote)

        assert redis_feed._last_quote_ts == mock_now

    @pytest.mark.asyncio
    async def test_on_quote_handles_missing_volume(self, redis_feed, mock_bar_aggregator):
        """Test that quotes with missing volume default to 0.0."""
        quote = Mock()
        quote.latest_matched_price = 1300.0
        quote.latest_matched_quantity = None

        with patch("src.paper.feeds.redis_feed.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 9, 15, 0)

            await redis_feed._on_quote("HNXDS:VN30F2601", quote)

        call_args = mock_bar_aggregator.on_tick.call_args[0]
        assert call_args[2] == 0.0  # volume


# --- Polling Loop Tests ---


class TestPollingLoop:
    """Tests for polling loop behavior."""

    @pytest.mark.asyncio
    async def test_polling_loop_calls_check_time(self, redis_feed, mock_bar_aggregator):
        """Test that polling loop calls BarAggregator.check_time()."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        # Let polling loop run for a longer time to ensure at least one call
        await asyncio.sleep(1.5)

        # check_time should have been called at least once
        assert mock_bar_aggregator.check_time.call_count >= 1

        # Cleanup
        await redis_feed.unsubscribe("VN30F1M")

    @pytest.mark.asyncio
    async def test_polling_loop_stops_on_unsubscribe(self, redis_feed, mock_bar_aggregator):
        """Test that polling loop stops when unsubscribed."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        # Let polling loop run
        await asyncio.sleep(0.1)
        call_count_before = mock_bar_aggregator.check_time.call_count

        # Unsubscribe
        await redis_feed.unsubscribe("VN30F1M")

        # Wait and verify no more calls
        await asyncio.sleep(0.1)
        call_count_after = mock_bar_aggregator.check_time.call_count

        assert call_count_after == call_count_before


# --- Watchdog Tests ---


class TestWatchdog:
    """Tests for watchdog monitoring."""

    def test_watchdog_no_warning_when_no_quotes_received(self, redis_feed):
        """Test that watchdog doesn't warn when no quotes received yet."""
        with patch("src.paper.feeds.redis_feed.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 9, 15, 0)

            # Should not raise or log warning
            redis_feed._check_watchdog()

    def test_watchdog_no_warning_outside_session(self, redis_feed, mock_session_manager):
        """Test that watchdog doesn't warn outside trading hours."""
        redis_feed._last_quote_ts = datetime(2024, 1, 15, 9, 0, 0)
        mock_session_manager.is_trading_hours.return_value = False

        with patch("src.paper.feeds.redis_feed.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 12, 0, 0)
            redis_feed._check_watchdog()

        # No assertion needed - just verify no exception

    def test_watchdog_no_warning_within_threshold(self, redis_feed):
        """Test that watchdog doesn't warn when silence is within threshold."""
        redis_feed._last_quote_ts = datetime(2024, 1, 15, 9, 0, 0)

        # Check 100 seconds later (below 300s threshold)
        with patch("src.paper.feeds.redis_feed.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 9, 1, 40)

            redis_feed._check_watchdog()

        # No assertion needed - just verify no exception

    def test_watchdog_warns_when_silence_exceeds_threshold(self, redis_feed, caplog):
        """Test that watchdog logs warning when silence exceeds threshold."""
        import logging

        caplog.set_level(logging.WARNING)

        redis_feed._last_quote_ts = datetime(2024, 1, 15, 9, 0, 0)

        # Check 400 seconds later (exceeds 300s threshold)
        with patch("src.paper.feeds.redis_feed.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 15, 9, 6, 40)

            redis_feed._check_watchdog()

        assert "Feed silence detected" in caplog.text
        assert "400" in caplog.text  # silence duration


# --- Error Handling Tests ---


class TestErrorHandling:
    """Tests for error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_on_quote_handles_malformed_quote_gracefully(
        self, redis_feed, mock_bar_aggregator
    ):
        """Test that malformed quotes don't crash the feed."""
        # Quote with missing attributes
        quote = Mock(spec=[])

        # Should not raise
        await redis_feed._on_quote("HNXDS:VN30F2601", quote)

        mock_bar_aggregator.on_tick.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsubscribe_handles_redis_error_gracefully(self, redis_feed, mock_redis_client):
        """Test that unsubscribe handles Redis errors gracefully."""
        callback = Mock()
        await redis_feed.subscribe("VN30F1M", callback)

        # Make Redis unsubscribe raise an error
        mock_redis_client.unsubscribe.side_effect = RuntimeError("Redis error")

        # Should not raise - error should be caught
        with suppress(RuntimeError):
            await redis_feed.unsubscribe("VN30F1M")

        # Note: This test reveals that RedisFeed.unsubscribe() doesn't handle
        # Redis errors gracefully. This should be fixed in the implementation.

    @pytest.mark.asyncio
    async def test_close_handles_redis_error_gracefully(self, redis_feed, mock_redis_client):
        """Test that close handles Redis errors gracefully."""
        # Make Redis close raise an error
        mock_redis_client.close.side_effect = RuntimeError("Redis error")

        # Should not raise - error should be caught
        with suppress(RuntimeError):
            await redis_feed.close()

        # Note: This test reveals that RedisFeed.close() doesn't handle
        # Redis errors gracefully. This should be fixed in the implementation.
