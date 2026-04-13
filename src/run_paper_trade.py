"""Entry point for paper trading system.

Supports three operating modes:
- LIVE: Connect to Redis (market data) + FIX (order execution)
- DRY-RUN: Connect to Redis/FIX but only log orders (no actual sends)
- SIM: Replay historical data (no Redis or FIX required)

Usage:
    # Live mode
    python -m src.run_paper_trade --strategy orb --symbol VN30F1M

    # Dry-run mode (market data live, orders logged only)
    python -m src.run_paper_trade --strategy orb --symbol VN30F1M --dry-run

    # Sim mode (historical replay)
    python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim --sample 100

    # Sim mode with date range
    python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim \\
        --sim-start 2024-01-01 --sim-end 2024-03-31

    # Custom config file
    python -m src.run_paper_trade --strategy orb --symbol VN30F1M --config my_config.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from config.constants import DEFAULT_INITIAL_CAPITAL, DEFAULT_SYMBOL
from src.utils.frequency import parse_frequency_to_minutes
from src.utils.logger import setup_logging

logger = setup_logging(
    name="run_paper_trade",
    log_file="logs/paper_trade.log",
    capture_all_loggers=True,
    console_level=logging.DEBUG,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Paper Trading System - LIVE / DRY-RUN / SIM modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live mode (full production)
  python -m src.run_paper_trade --strategy orb --symbol VN30F1M

  # Dry-run mode (market data live, orders logged only)
  python -m src.run_paper_trade --strategy orb --symbol VN30F1M --dry-run

  # Sim mode (historical replay, no external connections)
  python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim --sample 100

  # Sim mode with date range
  python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim \\
      --sim-start 2024-01-01 --sim-end 2024-03-31

  # Custom config file
  python -m src.run_paper_trade --strategy orb --symbol VN30F1M --config my_config.json
        """,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="orb",
        required=True,
        help="Strategy name (e.g. 'orb'). Must match a config file in config/strategy_params/",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help=f"Trading symbol (default: {DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect to Redis for market data but only log orders (no FIX sends)",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Replay historical data (no Redis or FIX required)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of bars to use in sim mode (default: all available)",
    )
    parser.add_argument(
        "--sim-start",
        type=str,
        default=None,
        help="Sim mode start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sim-end",
        type=str,
        default=None,
        help="Sim mode end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom strategy config JSON (default: config/strategy_params/{strategy}_default.json)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help=f"Initial capital in VND (default: {DEFAULT_INITIAL_CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Bar frequency override (e.g. '5min', '15min'). Default: use resample_freq from config",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments, exit with error message on failure."""
    if args.dry_run and args.sim:
        logger.error("Cannot specify both --dry-run and --sim. Choose one operating mode.")
        sys.exit(1)
    if args.config and not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    if args.sample is not None and args.sample <= 0:
        logger.error("--sample must be a positive integer, got: %d", args.sample)
        sys.exit(1)
    if args.capital <= 0:
        logger.error("--capital must be positive, got: %.2f", args.capital)
        sys.exit(1)
    if args.freq is not None:
        from src.utils.frequency import validate_frequency

        if not validate_frequency(args.freq):
            logger.error("Invalid --freq value: %s", args.freq)
            sys.exit(1)


def determine_mode(args: argparse.Namespace) -> str:
    """Return operating mode string: 'LIVE', 'DRY-RUN', or 'SIM'."""
    if args.sim:
        return "SIM"
    if args.dry_run:
        return "DRY-RUN"
    return "LIVE"


def _log_startup_info(args: argparse.Namespace, mode: str) -> None:
    """Log startup banner with key configuration values."""
    logger.info("=" * 60)
    logger.info("Paper Trading System - Starting")
    logger.info("=" * 60)
    logger.info("Mode    : %s", mode)
    logger.info("Strategy: %s", args.strategy)
    logger.info("Symbol  : %s", args.symbol)
    logger.info("Capital : %s VND", f"{args.capital:,.0f}")
    config_path = args.config or f"config/strategy_params/{args.strategy}_default.json"
    logger.info("Config  : %s", config_path)
    if mode == "SIM":
        if args.sim_start or args.sim_end:
            logger.info(
                "Sim range: %s → %s",
                args.sim_start or "earliest",
                args.sim_end or "latest",
            )
        logger.info("Sample  : %s", args.sample or "all bars")
    logger.info("=" * 60)


def _load_strategy_config(args: argparse.Namespace) -> Any:
    """Load and return the strategy config, exiting on missing file."""
    from config.schemas.orb import ORBConfig

    path = (
        Path(args.config)
        if args.config
        else Path(f"config/strategy_params/{args.strategy}_default.json")
    )
    if not path.exists():
        logger.error("Strategy config not found: %s", path)
        sys.exit(1)
    logger.info("Loading strategy config from %s", path)
    return ORBConfig.from_json(str(path))


# ---------------------------------------------------------------------------
# Engine runner
# ---------------------------------------------------------------------------


async def run_engine(args: argparse.Namespace, mode: str) -> None:
    """Run the paper trading engine with graceful shutdown.

    Args:
        args: Parsed command-line arguments.
        mode: Operating mode ('LIVE', 'DRY-RUN', or 'SIM').
    """
    from src.paper.builder import (
        build_engine,
        build_handler_configs,
        build_handlers,
        build_position_sizer,
        build_risk_manager,
        build_session_manager,
        build_strategy,
        build_tracker,
        prepare_live,
        prepare_sim,
    )

    strategy_config = _load_strategy_config(args)

    freq_str = args.freq or strategy_config.strategy.resample_freq
    freq_minutes = parse_frequency_to_minutes(freq_str)
    logger.info("Bar frequency: %d min (%s)", freq_minutes, freq_str)

    # Shared objects
    tracker = build_tracker(args.capital)
    session_manager = build_session_manager()
    strategy = build_strategy(strategy_config)
    risk_manager = build_risk_manager(strategy_config, args.capital)
    position_sizer = build_position_sizer(strategy_config)
    risk_handler_config, signal_handler_config = build_handler_configs(freq_minutes)

    engine = None
    historical_df = None
    sim_df = None

    try:
        if mode == "SIM":
            feed, order_manager, reconciler, sim_df = prepare_sim(
                args=args,
                strategy_config=strategy_config,
                strategy=strategy,
                tracker=tracker,
            )
        else:
            feed, order_manager, reconciler, historical_df = prepare_live(
                args=args,
                mode=mode,
                strategy_config=strategy_config,
                tracker=tracker,
                session_manager=session_manager,
                freq_minutes=freq_minutes,
            )

        bar_handler, risk_handler, signal_handler = build_handlers(
            tracker=tracker,
            order_manager=order_manager,
            session_manager=session_manager,
            risk_manager=risk_manager,
            strategy=strategy,
            position_sizer=position_sizer,
            risk_handler_config=risk_handler_config,
            signal_handler_config=signal_handler_config,
        )

        engine = build_engine(
            feed=feed,
            tracker=tracker,
            reconciler=reconciler,
            order_manager=order_manager,
            session_manager=session_manager,
            strategy=strategy,
            bar_handler=bar_handler,
            risk_handler=risk_handler,
            signal_handler=signal_handler,
            symbol=args.symbol,
        )

        shutdown_event = asyncio.Event()

        def _on_signal(sig: int, frame: object) -> None:
            logger.info("Received signal %s, shutting down...", sig)
            shutdown_event.set()

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        logger.info("Starting engine in %s mode...", mode)
        await engine.start(historical_df=historical_df, sim_df=sim_df)

        if mode != "SIM":
            await shutdown_event.wait()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Fatal error in engine")
        sys.exit(1)
    finally:
        if engine is not None:
            logger.info("Stopping engine...")
            await engine.stop()
        logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the paper trading system."""
    args = parse_args()
    setup_logging(args.log_level)
    validate_args(args)
    mode = determine_mode(args)
    _log_startup_info(args, mode)
    try:
        asyncio.run(run_engine(args, mode))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
