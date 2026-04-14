"""
Paper trading object graph builder.

Each function builds one specific thing. Higher-level functions (prepare_sim,
prepare_live) assemble the pieces for a given mode and return what run_paper_trade
needs to start the engine.

Nothing here touches CLI args or logging setup — that stays in run_paper_trade.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.data.pipeline import DataPipeline
    from src.engine.account.sizer import PositionSizer
    from src.engine.session.base import SessionManager
    from src.paper.account.reconciler import Reconciler
    from src.paper.account.tracker import Tracker
    from src.paper.bar_aggregator import BarAggregator
    from src.paper.engine import PaperEngine
    from src.paper.execution.order_manager import OrderManager
    from src.paper.feeds.base import FeedBase
    from src.paper.handlers.bar_handler import BarHandler
    from src.paper.handlers.risk_handler import RiskHandler, RiskHandlerConfig
    from src.paper.handlers.signal_handler import SignalHandler, SignalHandlerConfig
    from src.paper.risk_manager import RiskManager
    from src.strategy.orb import ORBStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level builders — one thing each
# ---------------------------------------------------------------------------


def build_tracker(capital: float) -> Tracker:
    """Build the account Tracker with VN30F commission and contract multiplier constants.

    Args:
        capital: Initial capital in VND.
    """
    from config.constants import VN30F_COMMISSION_RATE, VN30F_CONTRACT_MULTIPLIER
    from src.paper.account.tracker import Tracker

    return Tracker(
        initial_capital=capital,
        commission_rate=VN30F_COMMISSION_RATE,
        contract_multiplier=VN30F_CONTRACT_MULTIPLIER,
    )


def build_session_manager() -> SessionManager:
    """Build the VN30 session manager for trading hours validation."""
    from src.engine.session.vn30_session import VN30Session

    return VN30Session()


def build_strategy(strategy_config: Any) -> ORBStrategy:
    """Build the ORB strategy from a loaded config.

    Args:
        strategy_config: Validated ORBConfig instance.
    """
    from src.strategy.orb import ORBStrategy

    return ORBStrategy(config=strategy_config)


def build_risk_manager(strategy_config: Any, capital: float) -> RiskManager:
    """Build the paper trading risk manager.

    Reads trailing stop, ATR multiplier, and daily loss limit from strategy config.

    Args:
        strategy_config: Validated ORBConfig instance.
        capital: Initial capital used to calculate absolute daily loss limit.
    """
    from src.paper.risk_manager import RiskManager

    return RiskManager(
        use_trailing_stop=strategy_config.risk.use_trailing_stop,
        trailing_atr_multiplier=strategy_config.risk.trailing_atr_multiplier,
        max_daily_loss_fraction=strategy_config.risk.max_daily_loss,
        initial_capital=capital,
        max_loss_per_trade_fraction=0.0,
    )


def build_position_sizer(strategy_config: Any) -> PositionSizer:
    """Build the percent-risk position sizer.

    Args:
        strategy_config: Validated ORBConfig instance.
    """
    from src.engine.account.sizer import PercentRiskSizer

    return PercentRiskSizer(
        risk_per_trade_pct=strategy_config.risk.risk_per_trade_pct,
        max_size=strategy_config.risk.max_position_size,
    )


def build_indicator_pipeline(strategy_config: Any, use_cache: bool = False) -> DataPipeline:
    """Build the indicator pipeline from the strategy's registry.

    Registers ATR, ADX, and volume MA indicators as defined in strategy config.

    Args:
        strategy_config: Validated ORBConfig instance.
        use_cache: Whether to cache computed indicator DataFrames to disk.
                   Use True for SIM (avoids recomputing on reruns), False for live.
    """
    from src.data.pipeline import DataPipeline
    from src.strategy.orb import ORBStrategy

    registry = ORBStrategy.build_registry(
        atr_period=strategy_config.strategy.atr_period,
        adx_period=strategy_config.strategy.adx_period,
        volume_ma_period=strategy_config.strategy.volume_ma_period,
    )
    return DataPipeline(registry=registry, cache_dir="data/cache", use_cache=use_cache)


def build_bar_aggregator(
    freq_minutes: int,
    atr_period: int,
    session_manager: SessionManager,
    symbol: str,
    pipeline: DataPipeline | None = None,
) -> BarAggregator:
    """Build the BarAggregator for live tick-to-bar conversion.

    Reads bar quality config from environment (stale threshold, min updates, etc.)
    and optionally wires a DB fallback provider for sparse-tick recovery.

    Args:
        freq_minutes: Bar frequency in minutes (e.g. 1 for 1-min bars).
        atr_period: ATR period, used to determine the indicator warmup length.
        session_manager: Session manager for trading hours validation.
        symbol: Trading symbol, used by the DB fallback provider.
        pipeline: Indicator pipeline to enrich each emitted bar. Pass None to
                  emit raw OHLCV bars without indicators (not recommended for live).
    """
    from config.schemas.paper import get_paper_bar_config
    from src.database.data_service import get_data_service
    from src.paper.bar_aggregator import BarAggregator
    from src.paper.bar_fallback import create_fallback_provider

    bar_config = get_paper_bar_config()
    logger.info(
        "Bar config: stale=%ss min_updates=%d preclose_fetch=%ss debug_quotes=%s",
        bar_config.stale_seconds,
        bar_config.min_updates,
        bar_config.preclose_fetch_seconds,
        bar_config.debug_quotes,
    )

    fallback_provider = None
    if bar_config.enable_db_bar_fallback:
        fallback_provider = create_fallback_provider(
            data_service=get_data_service(),
            symbol=symbol,
            freq_minutes=freq_minutes,
            enabled=True,
        )
        logger.info("DB bar fallback enabled")
    else:
        logger.info("DB bar fallback disabled")

    return BarAggregator(
        freq_minutes=freq_minutes,
        atr_period=atr_period,
        fallback_bar_provider=fallback_provider,
        session_manager=session_manager,
        runtime_config={
            "pipeline": pipeline,
            "stale_trade_seconds": bar_config.stale_seconds,
            "min_live_updates": bar_config.min_updates,
            "preclose_fetch_seconds": bar_config.preclose_fetch_seconds,
            "debug_quotes": bar_config.debug_quotes,
        },
    )


def build_order_manager(
    tracker: Tracker,
    symbol: str,
    dry_run: bool,
    broker_client: Any = None,
) -> OrderManager:
    """Build the order manager for submitting entries and exits.

    Args:
        tracker: Account tracker, updated on each fill.
        symbol: Trading symbol for order tagging.
        dry_run: If True, orders are logged but not sent via FIX.
        broker_client: Live broker client. Pass None for SIM or dry-run.
    """
    from src.paper.execution.order_manager import OrderManager

    return OrderManager(
        client=broker_client,
        tracker=tracker,
        symbol=symbol,
        dry_run=dry_run,
    )


def build_reconciler(
    tracker: Tracker,
    order_manager: OrderManager,
    symbol: str,
    broker_client: Any = None,
) -> Reconciler:
    """Build the broker state reconciler.

    Reconciles position, cash, and open orders from the broker on startup
    so the tracker reflects actual broker state before the first bar.

    Args:
        tracker: Account tracker to reconcile into.
        order_manager: Order manager for cancelling stale orders.
        symbol: Trading symbol.
        broker_client: Live broker client. Pass None for SIM or dry-run.
    """
    from src.paper.account.reconciler import Reconciler

    return Reconciler(
        client=broker_client,
        tracker=tracker,
        order_manager=order_manager,
        symbol=symbol,
    )


def build_handler_configs(
    freq_minutes: int,
) -> tuple[RiskHandlerConfig, SignalHandlerConfig]:
    """Build RiskHandler and SignalHandler configs from environment variables.

    Both configs are driven entirely by paper engine env vars (PAPER_* keys),
    not by strategy config. This keeps runtime toggles separate from strategy params.

    Args:
        freq_minutes: Bar frequency in minutes, needed by RiskHandlerConfig for
                      pre-close timing calculations.
    """
    from config.constants import VN30F_CONTRACT_MULTIPLIER
    from config.schemas.paper import get_paper_engine_config
    from src.paper.handlers.risk_handler import RiskHandlerConfig
    from src.paper.handlers.signal_handler import SignalHandlerConfig

    engine_config = get_paper_engine_config()

    risk_cfg = RiskHandlerConfig(
        force_flat_on_session_close=engine_config.force_flat_on_session_close,
        force_flat_preclose_seconds=engine_config.force_flat_preclose_seconds,
        force_flat_on_last_candle=engine_config.force_flat_on_last_candle,
        defer_exit_outside_session=engine_config.defer_exit_outside_session,
        freq_minutes=freq_minutes,
    )
    signal_cfg = SignalHandlerConfig(
        entry_cutoff_seconds=engine_config.entry_cutoff_seconds,
        allow_late_entry=engine_config.allow_late_entry,
        contract_multiplier=VN30F_CONTRACT_MULTIPLIER,
    )
    return risk_cfg, signal_cfg


def build_handlers(
    tracker: Tracker,
    order_manager: OrderManager,
    session_manager: SessionManager,
    risk_manager: RiskManager,
    strategy: ORBStrategy,
    position_sizer: PositionSizer,
    risk_handler_config: RiskHandlerConfig,
    signal_handler_config: SignalHandlerConfig,
) -> tuple[BarHandler, RiskHandler, SignalHandler]:
    """Build the three pipeline handlers: BarHandler, RiskHandler, SignalHandler.

    These run in order on every bar inside PaperEngine._on_bar:
        1. BarHandler  — equity update, deferred exit processing
        2. RiskHandler — SL/TP check, trailing stop, force-flat
        3. SignalHandler — strategy signal generation, order submission

    Note: the deferred exit callback on RiskHandler is wired separately in
    build_engine after the engine instance exists.

    Args:
        tracker: Shared account tracker.
        order_manager: Order submission manager.
        session_manager: Session manager for trading hours checks.
        risk_manager: Risk manager for daily loss and trailing stop logic.
        strategy: Instantiated trading strategy.
        position_sizer: Position sizer for calculating order quantities.
        risk_handler_config: Config for RiskHandler (force-flat, pre-close, etc.).
        signal_handler_config: Config for SignalHandler (entry cutoff, contract multiplier).
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


def build_engine(
    feed: FeedBase,
    tracker: Tracker,
    reconciler: Reconciler,
    order_manager: OrderManager,
    session_manager: SessionManager,
    strategy: ORBStrategy,
    bar_handler: BarHandler,
    risk_handler: RiskHandler,
    signal_handler: SignalHandler,
    symbol: str,
) -> PaperEngine:
    """Build and fully wire the PaperEngine.

    Reads engine lifecycle config from environment (close_on_shutdown,
    force_hard_exit, etc.), constructs SessionStats, creates the engine,
    then wires the deferred exit callback on RiskHandler — which requires
    the engine instance and so can't be done earlier.

    Args:
        feed: Market data feed (RedisFeed for live, SimFeed for sim).
        tracker: Account tracker.
        reconciler: Broker state reconciler.
        order_manager: Order submission manager.
        session_manager: Session manager.
        strategy: Instantiated trading strategy (used for warmup).
        bar_handler: First handler in the pipeline.
        risk_handler: Second handler in the pipeline.
        signal_handler: Third handler in the pipeline.
        symbol: Trading symbol.
    """
    from config.schemas.paper import get_paper_engine_config
    from src.paper.engine import PaperEngine
    from src.paper.stats import SessionStats

    engine_config = get_paper_engine_config()
    logger.info(
        "Engine config: close_on_shutdown=%s force_hard_exit=%s "
        "force_flat_on_session_close=%s force_flat_preclose=%ss force_flat_on_last_candle=%s",
        engine_config.close_on_shutdown,
        engine_config.force_hard_exit,
        engine_config.force_flat_on_session_close,
        engine_config.force_flat_preclose_seconds,
        engine_config.force_flat_on_last_candle,
    )

    stats = SessionStats(tracker=tracker, benchmark_equity=None)

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
        symbol=symbol,
        close_on_shutdown=engine_config.close_on_shutdown,
        force_hard_exit=engine_config.force_hard_exit,
        output_dir="results/paper",
    )

    # Deferred exit callback requires the engine instance — wire after construction
    risk_handler.set_deferred_exit_callback(
        lambda reason: setattr(engine, "_deferred_exit_reason", reason)
    )

    return engine


# ---------------------------------------------------------------------------
# Mode-level assemblers
# ---------------------------------------------------------------------------


def prepare_sim(
    args: Any,
    strategy_config: Any,
    strategy: ORBStrategy,
    tracker: Tracker,
) -> tuple[FeedBase, OrderManager, Reconciler, pd.DataFrame]:
    """Assemble all runtime objects needed for SIM mode.

    Loads historical data from DB (or a date range if specified), resamples
    to the strategy frequency, runs the full indicator pipeline upfront, then
    builds a SimFeed, a dry-run OrderManager, and a no-op Reconciler.

    Args:
        args: Parsed CLI arguments (symbol, sample, sim_start, sim_end).
        strategy_config: Validated ORBConfig instance.
        strategy: Instantiated strategy, used to build the indicator registry.
        tracker: Account tracker.

    Returns:
        Tuple of (feed, order_manager, reconciler, sim_df).
    """
    from src.data.preprocessor import DataPreprocessor
    from src.database.data_service import get_data_service
    from src.paper.bootstrap import build_clients
    from src.paper.warmup_cache import load_with_cache

    logger.info("SIM mode: loading historical data...")
    data_service = get_data_service()

    if args.sim_start or args.sim_end:
        from src.data.loader import DataLoader

        start = args.sim_start or "2020-01-01"
        end = args.sim_end or pd.Timestamp.now().strftime("%Y-%m-%d")
        logger.info("Loading data range: %s to %s", start, end)
        try:
            raw_df = DataLoader(data_service, cache_dir="data/cache").load(
                symbol=args.symbol, start=start, end=end, use_cache=True
            )
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            sys.exit(1)
    else:
        raw_df = load_with_cache(data_service=data_service, db_symbol=args.symbol, n_days=30)

    if raw_df.empty:
        logger.error("No historical data available for %s", args.symbol)
        sys.exit(1)

    logger.info(
        "Loaded %d raw bars (%s to %s)",
        len(raw_df),
        raw_df["datetime"].iloc[0],
        raw_df["datetime"].iloc[-1],
    )

    target_freq = strategy_config.strategy.resample_freq
    processed_df = DataPreprocessor().prepare(raw_df, freq=target_freq)
    logger.info("Resampled to %s: %d bars", target_freq, len(processed_df))

    pipeline = build_indicator_pipeline(strategy_config, use_cache=True)
    processed_df = pipeline.run(processed_df)
    logger.info("Indicators computed (warmup=%d bars)", pipeline.get_required_lookback())

    sim_df = processed_df.head(args.sample) if args.sample else processed_df
    if args.sample:
        logger.info("Sample limit: using first %d of %d bars", len(sim_df), len(processed_df))
    logger.info(
        "Sim period: %s to %s (%d bars)",
        sim_df["datetime"].iloc[0],
        sim_df["datetime"].iloc[-1],
        len(sim_df),
    )

    feed, _ = build_clients(
        dry_run=False,
        sim=True,
        sim_df=sim_df,
        atr_period=strategy_config.strategy.atr_period,
        sim_speed=0.0,
    )
    order_manager = build_order_manager(tracker, args.symbol, dry_run=True)
    reconciler = build_reconciler(tracker, order_manager, args.symbol)

    return feed, order_manager, reconciler, sim_df


def prepare_live(
    args: Any,
    mode: str,
    strategy_config: Any,
    tracker: Tracker,
    session_manager: SessionManager,
    freq_minutes: int,
) -> tuple[FeedBase, OrderManager, Reconciler, pd.DataFrame | None]:
    """Assemble all runtime objects needed for LIVE or DRY-RUN mode.

    Loads recent warmup bars from DB, builds the indicator pipeline, creates
    and preloads the BarAggregator (including seeding any incomplete current bar),
    then connects to Redis and optionally the FIX broker.

    Args:
        args: Parsed CLI arguments (symbol).
        mode: "LIVE" or "DRY-RUN". Controls whether FIX orders are sent.
        strategy_config: Validated ORBConfig instance.
        tracker: Account tracker.
        session_manager: Session manager, passed to BarAggregator and RedisFeed.
        freq_minutes: Bar frequency in minutes, used for warmup data resampling.

    Returns:
        Tuple of (feed, order_manager, reconciler, historical_df).
        historical_df is None if no warmup data was available.
    """
    from src.database.data_service import get_data_service
    from src.paper.bootstrap import build_clients
    from src.paper.warmup_cache import load_with_cache
    from src.paper.warmup_seed import extract_incomplete_bar
    from src.utils.frequency import format_minutes_to_frequency

    logger.info("%s mode: loading warmup data...", mode)
    raw_df = load_with_cache(
        data_service=get_data_service(),
        db_symbol=args.symbol,
        n_days=7,
        convert_to_ohlcv=True,
        ohlcv_freq=format_minutes_to_frequency(freq_minutes),
    )

    incomplete_bar = extract_incomplete_bar(raw_df)

    if not raw_df.empty:
        logger.info("Loaded %d bars for warmup", len(raw_df))
        if incomplete_bar:
            logger.info(
                "Incomplete bar: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f",
                incomplete_bar["datetime"],
                incomplete_bar["open"],
                incomplete_bar["high"],
                incomplete_bar["low"],
                incomplete_bar["close"],
                incomplete_bar["volume"],
            )
    else:
        logger.warning("No historical data available for warmup")

    pipeline = build_indicator_pipeline(strategy_config, use_cache=False)
    logger.info(
        "Live indicator pipeline built (atr_period=%d)", strategy_config.strategy.atr_period
    )

    bar_aggregator = build_bar_aggregator(
        freq_minutes=freq_minutes,
        atr_period=strategy_config.strategy.atr_period,
        session_manager=session_manager,
        symbol=args.symbol,
        pipeline=pipeline,
    )

    if not raw_df.empty:
        logger.info("Preloading %d bars into BarAggregator...", len(raw_df))
        bar_aggregator.preload_history(raw_df)
        if incomplete_bar is not None:
            logger.info("Seeding incomplete bar...")
            bar_aggregator.seed_current_live_bar(incomplete_bar, validate_bucket=True)

    feed, broker_client = build_clients(
        dry_run=(mode == "DRY-RUN"),
        sim=False,
        bar_aggregator=bar_aggregator,
        session_manager=session_manager,
    )
    order_manager = build_order_manager(
        tracker, args.symbol, dry_run=(mode == "DRY-RUN"), broker_client=broker_client
    )
    reconciler = build_reconciler(tracker, order_manager, args.symbol, broker_client=broker_client)

    return feed, order_manager, reconciler, raw_df if not raw_df.empty else None
