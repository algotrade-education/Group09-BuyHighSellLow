"""
Paper trading CLI runner — wires ORB strategy to the live PaperTrader engine.

Usage:
    # Live mode (requires Redis + PaperBroker running):
    python -m src.run_paper_trade --config config/strategy_params/orb_default.json \\
        --symbol HNXDS:VN30F2601

    # Dry-run (FIX/Redis connected but no real orders sent):
    python -m src.run_paper_trade --config config/strategy_params/orb_default.json \\
        --symbol HNXDS:VN30F2601 --dry-run

    # Sim mode (replay historical in-sample data, no Redis needed):
    python -m src.run_paper_trade --config config/strategy_params/orb_default.json \\
        --symbol HNXDS:VN30F2601 --sim --sample is
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from config.config import DEFAULT_INITIAL_CAPITAL
from src.data.preprocessor import Preprocessor
from src.paper.engine import PaperTrader
from src.run_data_loader import load_data
from src.strategy.ORB import OpeningRangeBreakout
from src.utils.config_loader import load_config
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
    # The server validates SenderCompID against this UUID — using the .env value directly
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


async def main(args: argparse.Namespace) -> None:
    load_dotenv()

    # Load config
    config = load_config(args.config)
    strategy_params = config.get("strategy", {})
    resample_freq = strategy_params.get("resample_freq", "5min")

    # Build strategy (exclude non-constructor keys)
    strategy_kwargs = {k: v for k, v in strategy_params.items() if k != "resample_freq"}
    strategy = OpeningRangeBreakout(**strategy_kwargs)

    sim_df = None

    if args.sim:
        # ── Sim mode: load + preprocess historical data ──────────────────
        logger.info("Sim mode: loading %s data for %s…", args.sample, args.symbol)
        raw = load_data(sample=args.sample, contract=args.symbol.split(":")[-1])
        preprocessor = Preprocessor(atr_period=strategy_params.get("atr_period", 14))
        sim_df = preprocessor.prepare_for_backtest(raw, resample_freq=resample_freq)
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
        await trader.start(sim_df=sim_df)
    except KeyboardInterrupt:
        logger.info("Interrupted — stopping…")
        await trader.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ORB paper trading (live, dry-run, or sim)."
    )
    parser.add_argument(
        "--config",
        default="config/strategy_params/orb_default.json",
        help="Path to strategy config file.",
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
