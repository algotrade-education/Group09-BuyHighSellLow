"""Redis market data client wrapper.

Wraps redis.asyncio.Redis to provide pub/sub functionality for market data feeds.
Implements subscribe/unsubscribe methods and on_quote callback interface expected
by RedisFeed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class RedisMarketClient:
    """Redis market data client with pub/sub support.

    Wraps a Redis client to provide subscribe/unsubscribe methods and
    on_quote callback interface for market data streaming.
    """

    def __init__(self, redis: Redis) -> None:
        """Initialize RedisMarketClient.

        Args:
            redis: Redis client instance.
        """
        self._redis = redis
        self._pubsub = redis.pubsub()
        self._listen_task: asyncio.Task | None = None
        self.on_quote: Callable[[str, Any], None] | None = None

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to market data for a symbol.

        Args:
            symbol: Symbol to subscribe to (e.g. "VN30F1M").
        """
        await self._pubsub.subscribe(symbol)

        # Start listening task if not already running
        if self._listen_task is None or self._listen_task.done():
            self._listen_task = asyncio.create_task(self._listen_loop())

        logger.info("RedisMarketClient subscribed to %s", symbol)

    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from market data for a symbol.

        Args:
            symbol: Symbol to unsubscribe from.
        """
        await self._pubsub.unsubscribe(symbol)
        logger.info("RedisMarketClient unsubscribed from %s", symbol)

    async def close(self) -> None:
        """Close the client and clean up resources."""
        # Cancel listen task
        if self._listen_task is not None and not self._listen_task.done():
            self._listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listen_task

        # Close pubsub connection
        await self._pubsub.close()

        # Close Redis connection
        await self._redis.close()

        logger.info("RedisMarketClient closed")

    async def _listen_loop(self) -> None:
        """Listen for messages from Redis pub/sub."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    await self._handle_message(message)
        except asyncio.CancelledError:
            logger.debug("Listen loop cancelled")
            raise
        except Exception:
            logger.exception("Error in listen loop")

    async def _handle_message(self, message: dict) -> None:
        """Handle incoming pub/sub message.

        Args:
            message: Redis pub/sub message dict with 'channel' and 'data' keys.
        """
        if self.on_quote is None:
            return

        channel = message.get("channel")
        data = message.get("data")

        if channel and data:
            # Decode channel if bytes
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")

            # Parse data and invoke callback
            # Note: You may need to adjust this based on your actual data format
            # This assumes data is a quote object or can be parsed into one
            await self.on_quote(channel, data)  # type: ignore
