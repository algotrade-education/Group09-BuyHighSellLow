"""Feed base abstract interface for paper trading system.

This module defines the abstract interface that all feed implementations must follow.
Feeds are responsible for providing market data bars to the engine, either from live
sources (Redis) or historical replay (sim mode).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable


class FeedBase(ABC):
    """Abstract base class for market data feeds.

    All feed implementations (RedisFeed, SimFeed) must inherit from this class
    and implement the subscribe(), unsubscribe(), and close() methods.

    This interface allows swapping between Redis feed and sim feed without
    changing engine code.
    """

    @abstractmethod
    async def subscribe(self, symbol: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to market data for a symbol.

        Args:
            symbol: The symbol to subscribe to (e.g. "VN30F1M")
            callback: Callback function to invoke when a new bar is ready.
                     The callback receives a bar dict with keys:
                     - datetime: datetime object (bucket start time)
                     - open, high, low, close, volume: float
                     - indicator columns (e.g. atr_14): float

        The feed should begin emitting bars via the callback after subscription.
        """
        pass

    @abstractmethod
    async def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from market data for a symbol.

        Args:
            symbol: The symbol to unsubscribe from

        After unsubscribing, the callback should no longer be invoked for this symbol.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the feed and clean up resources.

        This method should gracefully shut down the feed, close any connections,
        unsubscribe from all symbols, and ensure all background tasks are cancelled.
        """
        pass

    async def wait_for_completion(self) -> None:
        """Wait until the feed has finished emitting all bars.

        Default implementation returns immediately (live feeds never "complete").
        Override in finite feeds (e.g. SimFeed) to await replay completion.
        """
        return
