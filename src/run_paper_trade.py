"""
Unified Paper Trading Entry Point.

This script wires any implemented strategy (ORB, KSB, VWAP) to the
`PaperTrader` engine. It supports three distinct modes of operation:

1. LIVE:     Connects to Redis (market data) and FIX (order execution).
2. DRY-RUN:  Connects to Redis/FIX but logs orders instead of sending them.
3. SIM:      Replays historical bars from disk (no external dependencies).

Usage:
    # LIVE ORB Trading
    python -m src.run_paper_trade --strategy orb --symbol HNXDS:VN30F2601

    # SIM VWAP Replay
    python -m src.run_paper_trade --strategy vwap --sim --sample is

Arguments:
    --strategy:  Strategy key (orb, ksb, vwap).
    --config:    Path to specific JSON config (overrides defaults).
    --symbol:    The market symbol to trade (HNXDS format).
    --dry-run:   Disable order submission.
    --sim:       Use historical data replay instead of live market feed.

Data Flow:
1. Load strategy and risk configurations.
2. Build connection clients (Redis, FIX/REST).
3. (Optional) Warmup: Fetch recent history from DB for indicator stability.
4. Start PaperTrader event loop.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from config.config import DEFAULT_INITIAL_CAPITAL
from src.paper.engine import PaperTrader
from src.strategy.base import Strategy
from src.utils.cli_helpers import (
    build_ksb_strategy,
    build_orb_strategy,
    build_vwap_strategy,
    load_ksb_config_context,
    load_orb_config_context,
    load_sample_data,
    load_vwap_config_context,
    prepare_backtest_dataset,
)
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/paper_trade.log")


def _build_clients(dry_run: bool):
    """Build PaperBrokerClient + RedisMarketDataClient from .env."""
    # --- FIX / REST client ---
    from paperbroker.client import PaperBrokerClient
    from src.paper.connect import resolve_fix_sender_comp_id

    username = os.getenv("PAPER_USERNAME", "BL01")
    password = os.getenv("PAPER_PASSWORD", "123")
    rest_url = os.getenv("PAPER_REST_BASE_URL", "http://localhost:9090")
    host = os.getenv("SOCKET_CONNECT_HOST", "localhost")
    port = int(os.getenv("SOCKET_CONNECT_PORT", "5001"))
    env_sender = os.getenv("SENDER_COMP_ID", "cross-FIX")
    target = os.getenv("TARGET_COMP_ID", "SERVER")
    sub_account = os.getenv("PAPER_ACCOUNT_ID_D1", "D1")

    # Resolve the correct fixAccountID from the broker REST API first.
    # The server validates SenderCompID against this UUID - using the .env value directly
    # causes an immediate FIX logout before wait_until_logged_on() can succeed.
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

    # --- Redis market data client ---
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
            merge_updates=True,  # Always show full snapshots (merged mode)
        )

    return client, redis_client


_CONFIG_LOADERS = {
    "orb": load_orb_config_context,
    "ksb": load_ksb_config_context,
    "vwap": load_vwap_config_context,
}

_STRATEGY_BUILDERS = {
    "orb": build_orb_strategy,
    "ksb": build_ksb_strategy,
    "vwap": build_vwap_strategy,
}

_DEFAULT_CONFIGS = {
    "orb": "config/strategy_params/orb_default.json",
    "ksb": "config/strategy_params/ksb_default.json",
    "vwap": "config/strategy_params/vwap_default.json",
}


async def main(args: argparse.Namespace) -> None:
    """
    Orchestrate the PaperTrader setup and execution loop.

    Handles configuration loading, historical data warmup,
    and asynchronous engine lifecycle.
    """
    load_dotenv()

    strat_key = args.strategy
    config_path = args.config or _DEFAULT_CONFIGS[strat_key]

    config, strategy_params, resample_freq = _CONFIG_LOADERS[strat_key](config_path)
    strategy: Strategy = _STRATEGY_BUILDERS[strat_key](strategy_params)

    sim_df = None

    if args.sim:
        # ── Sim mode: load + preprocess historical data ──────────────────
        logger.info("Sim mode: loading %s data for %s…", args.sample, args.symbol)
        raw = load_sample_data(sample=args.sample, contract=args.symbol.split(":")[-1])
        sim_df, _ = prepare_backtest_dataset(raw, strategy_params, resample_freq)
        logger.info("Sim data ready: %d bars.", len(sim_df))

        trader = PaperTrader(
            strategy=strategy,
            symbol=args.symbol,
            config=config,
            client=None,
            redis_client=None,
            bar_freq=resample_freq,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            dry_run=True,  # sim is always dry-run
        )
    else:
        # ── Live mode: build real clients ─────────────────────────────────
        client, redis_client = _build_clients(dry_run=args.dry_run)

        # Pre-load historical data for indicator warmup
        historical_df = None
        from datetime import datetime, timedelta
        from src.database.data_service import fetch_and_merge_data

        logger.info("Fetching recent history for indicator warmup...")
        # Fetch last 5 days to be safe (covering weekends and holidays)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        # Database stores continuous futures data under VN30F1M, not the specific contract code (e.g., VN30F2601)
        contract = args.symbol.split(":")[-1]
        db_symbol = "VN30F1M" if contract.startswith("VN30F") else contract

        raw = fetch_and_merge_data(
            contract_name=db_symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        if not raw.empty:
            historical_df, incomplete_bar = prepare_backtest_dataset(
                raw, strategy_params, resample_freq
            )
            logger.info(
                "Fetched %d historical bars for warmup using symbol %s.",
                len(historical_df),
                db_symbol,
            )
        else:
            logger.warning(
                "No recent history found for warmup using symbol %s! Strategy starts cold.",
                db_symbol,
            )

        trader = PaperTrader(
            strategy=strategy,
            symbol=args.symbol,
            config=config,
            client=client,
            redis_client=redis_client,
            bar_freq=resample_freq,
            initial_capital=DEFAULT_INITIAL_CAPITAL,
            dry_run=args.dry_run,
        )

    logger.info(
        "Starting PaperTrader | symbol=%s | freq=%s | mode=%s",
        args.symbol,
        resample_freq,
        "SIM" if args.sim else ("DRY-RUN" if args.dry_run else "LIVE"),
    )

    try:
        if args.sim:
            await trader.start(sim_df=sim_df)
        else:
            await trader.start(
                historical_df=historical_df,
                incomplete_bar=incomplete_bar if not args.dry_run else None,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted - stopping…")
        await trader.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run paper trading (live, dry-run, or sim) for any strategy."
    )
    parser.add_argument(
        "--strategy",
        choices=["orb", "ksb", "vwap"],
        default="orb",
        help="Strategy to run: orb, ksb, or vwap (default: orb).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to strategy config file (uses strategy default if omitted).",
    )
    parser.add_argument(
        "--symbol",
        default="HNXDS:VN30F2601",
        help="Full market symbol (e.g. HNXDS:VN30F2601).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect to FIX/Redis but log orders without sending them.",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Replay historical data (no Redis or FIX needed).",
    )
    parser.add_argument(
        "--sample",
        choices=["is", "os"],
        default="is",
        help="Historical sample to replay in --sim mode (is=in-sample, os=out-of-sample).",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
