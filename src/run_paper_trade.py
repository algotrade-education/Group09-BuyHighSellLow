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

from dotenv import load_dotenv

from config.config import DEFAULT_INITIAL_CAPITAL
from src.paper.engine import PaperTrader
from src.paper_trade.bootstrap import (
    build_clients,
    prepare_live_warmup_data,
    prepare_sim_replay_data,
)
from src.strategy.base import Strategy
from src.utils.cli_helpers import (
    build_ksb_strategy,
    build_orb_strategy,
    build_vwap_strategy,
    load_ksb_config_context,
    load_orb_config_context,
    load_vwap_config_context,
)
from src.utils.logger import setup_logging

logger = setup_logging(__name__, log_file="logs/paper_trade.log")


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
    historical_df = None
    incomplete_bar = None

    if args.sim:
        # ── Sim mode: load + preprocess historical data ──────────────────
        sim_df = prepare_sim_replay_data(
            sample=args.sample,
            symbol=args.symbol,
            strategy_params=strategy_params,
            resample_freq=resample_freq,
            logger=logger,
        )

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
        client, redis_client = build_clients(dry_run=args.dry_run, logger=logger)

        # Pre-load historical data for indicator warmup
        historical_df, incomplete_bar = prepare_live_warmup_data(
            symbol=args.symbol,
            strategy_params=strategy_params,
            resample_freq=resample_freq,
            logger=logger,
            warmup_days=7,
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
