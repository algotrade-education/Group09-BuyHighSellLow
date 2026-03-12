"""Bootstrap helpers for paper trading runners."""

import os
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from src.paper.connect import resolve_fix_sender_comp_id
from src.paper_trade.warmup_cache import load_with_cache
from src.utils.cli_helpers import load_sample_data, prepare_backtest_dataset

if TYPE_CHECKING:
    import pandas as pd
    from paperbroker.client import PaperBrokerClient
    from paperbroker.market_data import RedisMarketDataClient


def build_clients(
    *,
    dry_run: bool,
    logger: Any,
) -> Tuple["PaperBrokerClient", Optional["RedisMarketDataClient"]]:
    """Build PaperBrokerClient + RedisMarketDataClient from environment variables."""
    from paperbroker.client import PaperBrokerClient

    username = os.getenv("PAPER_USERNAME", "BL01")
    password = os.getenv("PAPER_PASSWORD", "123")
    rest_url = os.getenv("PAPER_REST_BASE_URL", "http://localhost:9090")
    host = os.getenv("SOCKET_CONNECT_HOST", "localhost")
    port = int(os.getenv("SOCKET_CONNECT_PORT", "5001"))
    env_sender = os.getenv("SENDER_COMP_ID", "cross-FIX")
    target = os.getenv("TARGET_COMP_ID", "SERVER")
    sub_account = os.getenv("PAPER_ACCOUNT_ID_D1", "D1")

    resolved_sender = resolve_fix_sender_comp_id(rest_url, username, password)
    sender = resolved_sender or env_sender

    client = PaperBrokerClient(
        default_sub_account=sub_account,
        username=username,
        password=password,
        rest_base_url=rest_url,
        socket_connect_host=host,
        socket_connect_port=port,
        sender_comp_id=sender,
        target_comp_id=target,
        console=False,
    )

    redis_host = os.getenv("MARKET_REDIS_HOST")
    redis_port = int(os.getenv("MARKET_REDIS_PORT", "6379"))
    redis_pw = os.getenv("MARKET_REDIS_PASSWORD")

    if not redis_host and not dry_run:
        logger.error(
            "MARKET_REDIS_HOST is not set. "
            "Add it to your .env or use --sim to run without Redis."
        )
        sys.exit(1)

    redis_client = None
    if redis_host:
        from paperbroker.market_data import RedisMarketDataClient

        redis_client = RedisMarketDataClient(
            host=redis_host,
            port=redis_port,
            password=redis_pw,
            merge_updates=True,
        )

    return client, redis_client


def prepare_sim_replay_data(
    *,
    sample: str,
    symbol: str,
    strategy_params: Dict[str, Any],
    resample_freq: str,
    logger: Any,
) -> "pd.DataFrame":
    """Load and preprocess historical data for sim replay mode."""
    logger.info("Sim mode: loading %s data for %s…", sample, symbol)
    raw = load_sample_data(sample=sample, contract=symbol.split(":")[-1])
    sim_df, _ = prepare_backtest_dataset(raw, strategy_params, resample_freq)
    logger.info("Sim data ready: %d bars.", len(sim_df))
    return sim_df


def prepare_live_warmup_data(
    *,
    symbol: str,
    strategy_params: Dict[str, Any],
    resample_freq: str,
    logger: Any,
    warmup_days: int = 5,
) -> Tuple[Optional["pd.DataFrame"], Optional[Dict[str, Any]]]:
    """Fetch and preprocess recent DB history used for live warmup.

    Past days are served from a local Parquet cache
    (``data/cache/warmup/{db_symbol}/{YYYY-MM-DD}.parquet``) so only today's
    data requires a DB round-trip on repeated starts.
    """
    historical_df: Optional["pd.DataFrame"] = None
    incomplete_bar: Optional[Dict[str, Any]] = None

    contract = symbol.split(":")[-1]
    db_symbol = "VN30F1M" if contract.startswith("VN30F") else contract

    logger.info(
        "Fetching warmup history for %s (last %d days, past days cached)…",
        db_symbol,
        warmup_days,
    )
    raw = load_with_cache(db_symbol, n_days=warmup_days)

    if not raw.empty:
        historical_df, incomplete_bar = prepare_backtest_dataset(
            raw, strategy_params, resample_freq
        )
        logger.info(
            "Warmup ready: %d bars for %s.",
            len(historical_df),
            db_symbol,
        )
    else:
        logger.warning(
            "No recent history found for warmup using symbol %s! Strategy starts cold.",
            db_symbol,
        )

    return historical_df, incomplete_bar
