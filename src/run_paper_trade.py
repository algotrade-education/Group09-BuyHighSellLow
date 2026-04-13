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

    # Sim mode with date range and sample limit
    python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim \\
        --sim-start 2024-01-01 --sim-end 2024-03-31 --sample 500

    # Custom config file
    python -m src.run_paper_trade --strategy orb --symbol VN30F1M --config my_config.json
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import DEFAULT_INITIAL_CAPITAL, DEFAULT_SYMBOL
from src.utils.frequency import format_minutes_to_frequency, parse_frequency_to_minutes
from src.utils.logger import setup_logging

logger = setup_logging(
    name="run_paper_trade",
    log_file="logs/paper_trade.log",
    capture_all_loggers=True,
    # console_level=logging.DEBUG
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
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

  # Sim mode with date range (replay specific period)
  python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim \\
      --sim-start 2024-01-01 --sim-end 2024-03-31

  # Sim mode with date range and sample limit (first 500 bars in range)
  python -m src.run_paper_trade --strategy orb --symbol VN30F1M --sim \\
      --sim-start 2024-01-01 --sim-end 2024-03-31 --sample 500

  # Custom config file
  python -m src.run_paper_trade --strategy orb --symbol VN30F1M --config my_config.json
        """,
    )

    # Required arguments
    parser.add_argument(
        "--strategy",
        type=str,
        default="orb",
        required=True,
        help="Strategy name (e.g., 'orb', 'breakout'). Must match a config file in config/strategy_params/",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=DEFAULT_SYMBOL,
        help=f"Trading symbol (default: {DEFAULT_SYMBOL})",
    )

    # Operating mode flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: connect to Redis for market data but only log orders (no FIX sends)",
    )

    parser.add_argument(
        "--sim",
        action="store_true",
        help="Sim mode: replay historical data (no Redis or FIX required)",
    )

    # Sim mode options
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of bars to sample for sim mode (default: use all available data)",
    )
    parser.add_argument(
        "--sim-start",
        type=str,
        default=None,
        help="Sim mode: start date for replay (YYYY-MM-DD format, e.g., 2024-01-01)",
    )
    parser.add_argument(
        "--sim-end",
        type=str,
        default=None,
        help="Sim mode: end date for replay (YYYY-MM-DD format, e.g., 2024-12-31)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom strategy config JSON file (default: config/strategy_params/{strategy}_default.json)",
    )

    # Additional options
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_INITIAL_CAPITAL,
        help=f"Initial capital (default: {DEFAULT_INITIAL_CAPITAL:,.0f} VND)",
    )

    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Bar frequency in minutes (default: use resample_freq from strategy config)",
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
    """Validate parsed arguments.

    Args:
        args: Parsed arguments namespace.

    Raises:
        SystemExit: If validation fails.
    """
    # Validate mode flags
    if args.dry_run and args.sim:
        logger.error("Cannot specify both --dry-run and --sim. Choose one operating mode.")
        sys.exit(1)

    # Validate config file exists if specified
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)

    # Validate sample is positive if specified
    if args.sample is not None and args.sample <= 0:
        logger.error("--sample must be a positive integer, got: %d", args.sample)
        sys.exit(1)

    # Validate capital is positive
    if args.capital <= 0:
        logger.error("--capital must be positive, got: %.2f", args.capital)
        sys.exit(1)

    # Validate freq if specified
    if args.freq is not None:
        from src.utils.frequency import validate_frequency

        if not validate_frequency(args.freq):
            logger.error(
                "--freq must be a positive integer or format like '5min', '1H', got: %s", args.freq
            )
            sys.exit(1)


def determine_mode(args: argparse.Namespace) -> str:
    """Determine operating mode from arguments.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Operating mode string: "LIVE", "DRY-RUN", or "SIM".
    """
    if args.sim:
        return "SIM"
    elif args.dry_run:
        return "DRY-RUN"
    else:
        return "LIVE"


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    return Path(f"config/strategy_params/{args.strategy}_default.json")


def _load_strategy_config(args: argparse.Namespace) -> Any:
    from config.schemas.orb import ORBConfig

    config_path = _resolve_config_path(args)
    if not config_path.exists():
        logger.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    logger.info("Loading strategy config from %s", config_path)
    return ORBConfig.from_json(str(config_path))


def _build_shared_runtime(
    args: argparse.Namespace,
    strategy_config: Any,
    freq_minutes: int,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    from config.constants import VN30F_COMMISSION_RATE, VN30F_CONTRACT_MULTIPLIER
    from config.schemas.paper import get_paper_engine_config
    from src.engine.account.sizer import PercentRiskSizer
    from src.engine.session.vn30_session import VN30Session
    from src.paper.account.tracker import Tracker
    from src.paper.handlers.risk_handler import RiskHandlerConfig
    from src.paper.handlers.signal_handler import SignalHandlerConfig
    from src.paper.risk_manager import RiskManager
    from src.strategy.orb import ORBStrategy

    # Load paper engine config from environment
    engine_config = get_paper_engine_config()

    tracker = Tracker(
        initial_capital=args.capital,
        commission_rate=VN30F_COMMISSION_RATE,
        contract_multiplier=VN30F_CONTRACT_MULTIPLIER,
    )
    session_manager = VN30Session()
    risk_manager = RiskManager(
        use_trailing_stop=strategy_config.risk.use_trailing_stop,
        trailing_atr_multiplier=strategy_config.risk.trailing_atr_multiplier,
        max_daily_loss_fraction=strategy_config.risk.max_daily_loss,
        initial_capital=args.capital,
        max_loss_per_trade_fraction=0.0,
    )
    strategy = ORBStrategy(config=strategy_config)

    risk_handler_config = RiskHandlerConfig(
        force_flat_on_session_close=engine_config.force_flat_on_session_close,
        force_flat_preclose_seconds=engine_config.force_flat_preclose_seconds,
        force_flat_on_last_candle=engine_config.force_flat_on_last_candle,
        defer_exit_outside_session=engine_config.defer_exit_outside_session,
        freq_minutes=freq_minutes,
    )
    position_sizer = PercentRiskSizer(
        risk_per_trade_pct=strategy_config.risk.risk_per_trade_pct,
        max_size=strategy_config.risk.max_position_size,
    )
    signal_handler_config = SignalHandlerConfig(
        entry_cutoff_seconds=engine_config.entry_cutoff_seconds,
        allow_late_entry=engine_config.allow_late_entry,
        contract_multiplier=VN30F_CONTRACT_MULTIPLIER,
    )

    return (
        tracker,
        session_manager,
        risk_manager,
        strategy,
        risk_handler_config,
        position_sizer,
        signal_handler_config,
    )


def _create_bar_aggregator(
    freq_minutes: int,
    atr_period: int,
    session_manager: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict]:
    """Create BarAggregator with runtime config and fallback provider.

    Args:
        freq_minutes: Bar frequency in minutes.
        atr_period: ATR period for warmup.
        session_manager: Session manager instance.
        args: Command-line arguments.

    Returns:
        Tuple of (bar_aggregator, runtime_config).
    """
    from config.schemas.paper import get_paper_bar_config
    from src.database.data_service import get_data_service
    from src.paper.bar_aggregator import BarAggregator
    from src.paper.bar_fallback import create_fallback_provider

    bar_config = get_paper_bar_config()
    runtime_config = {
        "stale_trade_seconds": bar_config.stale_seconds,
        "min_live_updates": bar_config.min_updates,
        "preclose_fetch_seconds": bar_config.preclose_fetch_seconds,
        "debug_quotes": bar_config.debug_quotes,
    }
    logger.info(
        "Bar runtime config: stale=%ss min_updates=%d preclose_fetch=%ss debug_quotes=%s",
        bar_config.stale_seconds,
        bar_config.min_updates,
        bar_config.preclose_fetch_seconds,
        bar_config.debug_quotes,
    )

    fallback_provider = None
    if bar_config.enable_db_bar_fallback:
        data_service = get_data_service()
        fallback_provider = create_fallback_provider(
            data_service=data_service,
            symbol=args.symbol,
            freq_minutes=freq_minutes,
            enabled=True,
        )
        logger.info("DB bar fallback enabled")
    else:
        logger.info("DB bar fallback disabled")

    bar_aggregator = BarAggregator(
        freq_minutes=freq_minutes,
        atr_period=atr_period,
        fallback_bar_provider=fallback_provider,
        runtime_config=runtime_config,
        session_manager=session_manager,
    )

    return bar_aggregator, runtime_config


def _create_handlers(
    tracker: Any,
    order_manager: Any,
    session_manager: Any,
    risk_manager: Any,
    strategy: Any,
    position_sizer: Any,
    risk_handler_config: Any,
    signal_handler_config: Any,
) -> tuple[Any, Any, Any]:
    """Create handler instances for the engine pipeline.

    The deferred exit callback is NOT wired here — it requires the engine
    instance which doesn't exist yet. Call risk_handler.set_deferred_exit_callback()
    after engine construction.

    Args:
        tracker: Account tracker.
        order_manager: Order manager.
        session_manager: Session manager.
        risk_manager: Risk manager.
        strategy: Trading strategy.
        position_sizer: Position sizer.
        risk_handler_config: Risk handler configuration.
        signal_handler_config: Signal handler configuration.

    Returns:
        Tuple of (bar_handler, risk_handler, signal_handler).
    """
    from src.paper.handlers.bar_handler import BarHandler
    from src.paper.handlers.risk_handler import RiskHandler
    from src.paper.handlers.signal_handler import SignalHandler

    bar_handler = BarHandler(
        tracker=tracker,
        order_manager=order_manager,
        session_manager=session_manager,
    )
    risk_handler = RiskHandler(
        tracker=tracker,
        order_manager=order_manager,
        risk_manager=risk_manager,
        session_manager=session_manager,
        config=risk_handler_config,
    )
    signal_handler = SignalHandler(
        strategy=strategy,
        tracker=tracker,
        order_manager=order_manager,
        risk_manager=risk_manager,
        session_manager=session_manager,
        position_sizer=position_sizer,
        config=signal_handler_config,
    )

    return bar_handler, risk_handler, signal_handler


def _load_sim_raw_data(args: argparse.Namespace, data_service: Any) -> pd.DataFrame:
    from src.paper.warmup_cache import load_with_cache

    if args.sim_start or args.sim_end:
        from src.data.loader import DataLoader

        start_date_str = args.sim_start if args.sim_start else "2020-01-01"
        end_date_str = args.sim_end if args.sim_end else pd.Timestamp.now().strftime("%Y-%m-%d")

        logger.info("Loading data for date range: %s to %s", start_date_str, end_date_str)

        try:
            loader = DataLoader(data_service, cache_dir="data/cache")
            return loader.load(
                symbol=args.symbol,
                start=start_date_str,
                end=end_date_str,
                use_cache=True,
            )
        except Exception as e:
            logger.error("Failed to load data for specified date range: %s", e)
            sys.exit(1)

    return load_with_cache(
        data_service=data_service,
        db_symbol=args.symbol,
        n_days=30,
    )


def _prepare_sim_mode_runtime(
    args: argparse.Namespace,
    strategy_config: Any,
    strategy: Any,
    tracker: Any,
    atr_period: int,
) -> tuple[Any, Any, Any, pd.DataFrame | None]:
    from src.data.pipeline import DataPipeline
    from src.data.preprocessor import DataPreprocessor
    from src.database.data_service import get_data_service
    from src.paper.account.reconciler import Reconciler
    from src.paper.bootstrap import build_clients
    from src.paper.execution.order_manager import OrderManager

    logger.info("SIM mode: Loading historical data...")
    data_service = get_data_service()
    raw_df = _load_sim_raw_data(args, data_service)

    if raw_df.empty:
        logger.error("No historical data available for symbol %s", args.symbol)
        sys.exit(1)

    logger.info("Loaded %d bars (1-minute) from database", len(raw_df))
    data_start = raw_df["datetime"].iloc[0]
    data_end = raw_df["datetime"].iloc[-1]
    logger.info("Data range: %s to %s", data_start, data_end)

    target_freq = strategy_config.strategy.resample_freq
    logger.info("Resampling to %s bars for strategy...", target_freq)

    preprocessor = DataPreprocessor()
    processed_df = preprocessor.prepare(raw_df, freq=target_freq)

    logger.info("After preprocessing: %d bars (%s)", len(processed_df), target_freq)

    logger.info("Computing indicators...")
    registry = strategy.build_registry(
        atr_period=strategy_config.strategy.atr_period,
        adx_period=strategy_config.strategy.adx_period,
        volume_ma_period=strategy_config.strategy.volume_ma_period,
    )
    pipeline = DataPipeline(registry=registry, cache_dir="data/cache", use_cache=True)
    processed_df = pipeline.run(processed_df)
    warmup_bars = pipeline.get_required_lookback()
    logger.info("Indicators computed (warmup = %d bars)", warmup_bars)

    sim_df = processed_df.head(args.sample) if args.sample else processed_df

    if args.sample:
        logger.info(
            "Sample limit applied: using first %d of %d bars",
            len(sim_df),
            len(processed_df),
        )

    if not sim_df.empty:
        sim_start = sim_df["datetime"].iloc[0]
        sim_end = sim_df["datetime"].iloc[-1]
        logger.info("Simulation period: %s to %s (%d bars)", sim_start, sim_end, len(sim_df))

    feed, _ = build_clients(
        dry_run=False,
        sim=True,
        sim_df=sim_df,
        atr_period=atr_period,
        sim_speed=0.0,
    )
    order_manager = OrderManager(
        client=None,
        tracker=tracker,
        symbol=args.symbol,
        dry_run=True,
    )
    reconciler = Reconciler(
        client=None,
        tracker=tracker,
        order_manager=order_manager,
        symbol=args.symbol,
    )
    return feed, order_manager, reconciler, sim_df


def _prepare_live_or_dry_runtime(
    args: argparse.Namespace,
    mode: str,
    tracker: Any,
    session_manager: Any,
    freq_minutes: int,
    atr_period: int,
) -> tuple[Any, Any, Any, pd.DataFrame | None]:
    from src.database.data_service import get_data_service
    from src.paper.account.reconciler import Reconciler
    from src.paper.bootstrap import build_clients
    from src.paper.execution.order_manager import OrderManager
    from src.paper.warmup_cache import load_with_cache
    from src.paper.warmup_seed import extract_incomplete_bar

    logger.info("%s mode: Loading warmup data...", mode)
    data_service = get_data_service()
    raw_df = load_with_cache(
        data_service=data_service,
        db_symbol=args.symbol,
        n_days=5,
        convert_to_ohlcv=True,
        ohlcv_freq=format_minutes_to_frequency(freq_minutes),
    )

    # Extract incomplete bar for seeding
    incomplete_bar = extract_incomplete_bar(raw_df)

    if not raw_df.empty:
        logger.info("Loaded %d bars for warmup", len(raw_df))
        if incomplete_bar:
            logger.info(
                "Extracted incomplete bar: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
                incomplete_bar["datetime"],
                incomplete_bar["open"],
                incomplete_bar["high"],
                incomplete_bar["low"],
                incomplete_bar["close"],
                incomplete_bar["volume"],
            )
    else:
        logger.warning("No historical data available for warmup")

    # Create bar aggregator with config
    bar_aggregator, _ = _create_bar_aggregator(freq_minutes, atr_period, session_manager, args)

    # Preload history and seed incomplete bar
    if not raw_df.empty:
        logger.info("Preloading %d bars into BarAggregator for warmup", len(raw_df))
        bar_aggregator.preload_history(raw_df)

        if incomplete_bar is not None:
            logger.info("Seeding incomplete bar into BarAggregator...")
            bar_aggregator.seed_current_live_bar(incomplete_bar, validate_bucket=True)

    # Build clients and managers
    feed, broker_client = build_clients(
        dry_run=(mode == "DRY-RUN"),
        sim=False,
        bar_aggregator=bar_aggregator,
        session_manager=session_manager,
    )
    order_manager = OrderManager(
        client=broker_client,
        tracker=tracker,
        symbol=args.symbol,
        dry_run=(mode == "DRY-RUN"),
    )
    reconciler = Reconciler(
        client=broker_client,
        tracker=tracker,
        order_manager=order_manager,
        symbol=args.symbol,
    )

    return feed, order_manager, reconciler, raw_df if not raw_df.empty else None


def _build_engine(
    args: argparse.Namespace,
    feed: Any,
    order_manager: Any,
    tracker: Any,
    reconciler: Any,
    session_manager: Any,
    risk_manager: Any,
    strategy: Any,
    risk_handler_config: Any,
    position_sizer: Any,
    signal_handler_config: Any,
) -> Any:
    from config.schemas.paper import get_paper_engine_config
    from src.paper.engine import PaperEngine
    from src.paper.stats import SessionStats

    # Load paper engine config from environment
    engine_config = get_paper_engine_config()
    logger.info(
        "Paper engine config: close_on_shutdown=%s force_hard_exit=%s "
        "force_flat_on_session_close=%s force_flat_preclose=%ss force_flat_on_last_candle=%s",
        engine_config.close_on_shutdown,
        engine_config.force_hard_exit,
        engine_config.force_flat_on_session_close,
        engine_config.force_flat_preclose_seconds,
        engine_config.force_flat_on_last_candle,
    )

    # Create handlers (no deferred exit callback yet - engine not constructed)
    bar_handler, risk_handler, signal_handler = _create_handlers(
        tracker=tracker,
        order_manager=order_manager,
        session_manager=session_manager,
        risk_manager=risk_manager,
        strategy=strategy,
        position_sizer=position_sizer,
        risk_handler_config=risk_handler_config,
        signal_handler_config=signal_handler_config,
    )

    # Create stats
    stats = SessionStats(
        tracker=tracker,
        benchmark_equity=None,
    )

    # Create engine
    engine = PaperEngine(
        feed=feed,
        bar_handler=bar_handler,
        risk_handler=risk_handler,
        signal_handler=signal_handler,
        tracker=tracker,
        reconciler=reconciler,
        order_manager=order_manager,
        stats=stats,
        session_manager=session_manager,
        strategy=strategy,
        symbol=args.symbol,
        close_on_shutdown=engine_config.close_on_shutdown,
        force_hard_exit=engine_config.force_hard_exit,
        output_dir="results/paper",
    )

    # Wire deferred exit callback now that engine exists - avoids closure over None
    risk_handler.set_deferred_exit_callback(
        lambda reason: setattr(engine, "_deferred_exit_reason", reason)
    )

    return engine


async def run_engine(args: argparse.Namespace, mode: str) -> None:
    """Run the paper trading engine with graceful shutdown.

    Args:
        args: Parsed command-line arguments.
        mode: Operating mode (LIVE, DRY-RUN, or SIM).
    """
    strategy_config = _load_strategy_config(args)

    # Use resample_freq from config, allow CLI override
    if args.freq is not None:
        freq_str = args.freq
        logger.info("Using frequency from CLI: %s", freq_str)
    else:
        freq_str = strategy_config.strategy.resample_freq
        logger.info("Using frequency from config: %s", freq_str)

    # Parse frequency string to minutes using centralized utility
    freq_minutes = parse_frequency_to_minutes(freq_str)
    logger.info("Bar frequency: %d minutes", freq_minutes)
    atr_period = strategy_config.strategy.atr_period  # Use ATR period from config

    (
        tracker,
        session_manager,
        risk_manager,
        strategy,
        risk_handler_config,
        position_sizer,
        signal_handler_config,
    ) = _build_shared_runtime(args, strategy_config, freq_minutes)

    engine = None
    historical_df = None
    sim_df = None

    try:
        if mode == "SIM":
            feed, order_manager, reconciler, sim_df = _prepare_sim_mode_runtime(
                args=args,
                strategy_config=strategy_config,
                strategy=strategy,
                tracker=tracker,
                atr_period=atr_period,
            )

        else:
            feed, order_manager, reconciler, historical_df = _prepare_live_or_dry_runtime(
                args=args,
                mode=mode,
                tracker=tracker,
                session_manager=session_manager,
                freq_minutes=freq_minutes,
                atr_period=atr_period,
            )

        engine = _build_engine(
            args=args,
            feed=feed,
            order_manager=order_manager,
            tracker=tracker,
            reconciler=reconciler,
            session_manager=session_manager,
            risk_manager=risk_manager,
            strategy=strategy,
            risk_handler_config=risk_handler_config,
            position_sizer=position_sizer,
            signal_handler_config=signal_handler_config,
        )

        # Setup graceful shutdown
        shutdown_event = asyncio.Event()

        def signal_handler_func(sig: int, frame: object) -> None:
            logger.info("Received signal %s, initiating graceful shutdown...", sig)
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler_func)
        signal.signal(signal.SIGTERM, signal_handler_func)

        logger.info("Starting engine...")
        await engine.start(
            historical_df=historical_df,
            sim_df=sim_df,
        )

        # SIM replay is complete when engine.start() returns - no need to wait
        # for a signal. LIVE/DRY-RUN must wait for SIGINT/SIGTERM.
        if mode != "SIM":
            await shutdown_event.wait()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    except Exception:
        logger.exception("Fatal error in engine")
        sys.exit(1)
    finally:
        if engine is not None:
            logger.info("Calling engine.stop()...")
            await engine.stop()
        logger.info("Shutdown complete")


def main() -> None:
    """Main entry point for paper trading system."""
    # Parse and validate arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Validate arguments
    validate_args(args)

    # Determine operating mode
    mode = determine_mode(args)

    # Run engine with asyncio
    try:
        asyncio.run(run_engine(args, mode))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
