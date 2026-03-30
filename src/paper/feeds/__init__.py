"""Paper feed adapters.

Exports feed interface and concrete live/simulation feed implementations.
"""

from src.paper.feeds.base import FeedBase
from src.paper.feeds.redis_feed import RedisFeed
from src.paper.feeds.sim_feed import SimFeed

__all__ = ["FeedBase", "RedisFeed", "SimFeed"]
