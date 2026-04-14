"""Bootstrap module for paper trading system.

Responsible for constructing feed and broker client instances based on
operating mode (LIVE, DRY-RUN, SIM).

Key responsibilities:
- Validate environment configuration
- Construct appropriate feed type (RedisFeed or SimFeed)
- Construct broker client with resolved SenderCompID
- Raise BootstrapError on misconfiguration
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from config.secrets import get_secrets
from src.paper.bar_aggregator import BarAggregator
from src.paper.feeds.base import FeedBase
from src.paper.feeds.redis_feed import RedisFeed
from src.paper.feeds.sim_feed import SimFeed

if TYPE_CHECKING:
    import pandas as pd
    from paperbroker.client import PaperBrokerClient

logger = logging.getLogger(__name__)


class BootstrapError(Exception):
    """Bootstrap configuration or initialization error."""


def build_clients(
    *,
    dry_run: bool,
    sim: bool,
    bar_aggregator: BarAggregator | None = None,
    session_manager: Any | None = None,
    sim_df: pd.DataFrame | None = None,
    sim_pipeline: Any | None = None,
    atr_period: int = 14,
    sim_speed: float = 0.0,
) -> tuple[FeedBase, PaperBrokerClient | None]:
    """Build feed and broker client instances based on operating mode.

    Operating modes:
    - LIVE: dry_run=False, sim=False → RedisFeed + PaperBrokerClient
    - DRY-RUN: dry_run=True, sim=False → RedisFeed + None (no FIX sends)
    - SIM: sim=True → SimFeed + None (no external connections)

    Args:
        dry_run: If True, orders are logged but not sent via FIX.
        sim: If True, use SimFeed for historical replay (no Redis/FIX).
        bar_aggregator: BarAggregator instance (required for RedisFeed).
        session_manager: SessionManager for trading hours validation (required for RedisFeed).
        sim_df: Historical DataFrame (required for SimFeed).
        sim_pipeline: Optional pipeline for indicator computation (SimFeed).
        atr_period: ATR period for warmup calculation (SimFeed).
        sim_speed: Seconds to sleep between bars (SimFeed, 0=max throughput).

    Returns:
        Tuple of (feed, broker_client).
        - feed: RedisFeed or SimFeed instance
        - broker_client: PaperBrokerClient instance (LIVE mode only) or None

    Raises:
        SystemExit: If MARKET_REDIS_HOST is not set in LIVE/DRY-RUN mode.
    """
    # SIM mode: no external connections required
    if sim:
        logger.info("Bootstrap: SIM mode - constructing SimFeed")
        if sim_df is None:
            raise BootstrapError("sim_df is required for SIM mode")

        feed: FeedBase = SimFeed(
            df=sim_df,
            pipeline=sim_pipeline,
            atr_period=atr_period,
            speed=sim_speed,
        )
        return feed, None

    # LIVE/DRY-RUN mode: validate Redis configuration
    secrets = get_secrets()

    # Check MARKET_REDIS_HOST is set
    redis_host = secrets.redis.host
    if not redis_host or redis_host == "localhost":  # noqa: SIM102
        # Check if explicitly set in environment
        if "MARKET_REDIS_HOST" not in os.environ:
            raise BootstrapError(
                "MARKET_REDIS_HOST environment variable is not set. "
                "This is required for LIVE and DRY-RUN modes. "
                "Please set MARKET_REDIS_HOST in your .env file or environment."
            )

    # Construct Redis client
    logger.info("Bootstrap: Connecting to Redis at %s:%d", redis_host, secrets.redis.port)

    try:
        from paperbroker.market_data import RedisMarketDataClient

        redis_client = RedisMarketDataClient(
            host=redis_host,
            port=secrets.redis.port,
            password=secrets.redis.password.get_secret_value() if secrets.redis.password else None,
            merge_updates=True,
        )
    except ImportError as exc:
        logger.error(
            "Bootstrap: paperbroker.market_data not available. Ensure paperbroker package is installed."
        )
        raise RuntimeError("paperbroker.market_data not available") from exc

    # Construct RedisFeed
    if bar_aggregator is None:
        raise BootstrapError("bar_aggregator is required for RedisFeed")
    if session_manager is None:
        raise BootstrapError("session_manager is required for RedisFeed")

    redis_feed: FeedBase = RedisFeed(
        redis_client=redis_client,
        bar_aggregator=bar_aggregator,
        session_manager=session_manager,
        watchdog_silence_seconds=300.0,
    )

    # Construct broker client (LIVE mode only)
    broker_client = None
    if not dry_run:
        logger.info("Bootstrap: LIVE mode - constructing PaperBrokerClient")
        broker_client = build_broker_client()
    else:
        logger.info("Bootstrap: DRY-RUN mode - orders will be logged only")

    return redis_feed, broker_client


def build_broker_client() -> PaperBrokerClient:
    """Construct PaperBrokerClient with resolved credentials.

    Uses get_broker_credentials() from config.secrets for clean credential
    resolution with multiple strategies (API, environment variables).

    Returns:
        Connected and logged-on PaperBrokerClient instance.

    Raises:
        SystemExit: If credentials cannot be resolved or client construction fails.
    """
    try:
        # Import broker client (lazy import)
        from paperbroker.client import PaperBrokerClient
    except ImportError as exc:
        logger.error(
            "Bootstrap: PaperBrokerClient not available. Ensure paperbroker package is installed."
        )
        raise RuntimeError("PaperBrokerClient not available") from exc

    try:
        # Get resolved credentials from config.secrets
        from config.secrets import get_broker_credentials

        creds = get_broker_credentials(enable_api_resolution=True)

        # Construct client using credentials
        client = PaperBrokerClient(**creds.to_client_kwargs(), console=False)

        logger.info("Bootstrap: Starting FIX session...")
        client.connect()

        timeout = int(os.getenv("PAPER_FIX_LOGON_TIMEOUT", "60"))
        if not client.wait_until_logged_on(timeout=timeout):
            error = client.last_logon_error() or "no reason returned"
            raise BootstrapError(f"FIX logon failed: {error}")

        logger.info(
            "Bootstrap: PaperBrokerClient constructed with SenderCompID=%s",
            creds.sender_comp_id,
        )
        return client

    except ValueError as exc:
        raise BootstrapError(f"Failed to resolve credentials: {exc}") from exc
    except Exception as exc:
        raise BootstrapError(f"Failed to construct PaperBrokerClient: {exc}") from exc
