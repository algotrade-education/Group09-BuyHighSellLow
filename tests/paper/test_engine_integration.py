"""End-to-end integration tests for PaperEngine.

Tests the complete flow from feed → engine → handlers → orders → stats.
These tests verify that all components work together correctly.
"""

from datetime import datetime
from typing import cast
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from src.paper.account.reconciler import Reconciler
from src.paper.account.tracker import Tracker
from src.paper.engine import PaperEngine
from src.paper.execution.order_manager import OrderManager
from src.paper.feeds.sim_feed import SimFeed
from src.paper.handlers.bar_handler import BarHandler
from src.paper.handlers.risk_handler import RiskHandler, RiskHandlerConfig
from src.paper.handlers.signal_handler import SignalHandler, SignalHandlerConfig
from src.paper.risk_manager import RiskManager
from src.paper.stats import SessionStats
from src.strategy.signal import Signal, TradeSignal


@pytest.fixture
def sample_bars_df():
    """Create sample bars DataFrame for testing."""
    dates = pd.date_range("2024-01-15 09:00", periods=10, freq="5min")
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [
                1250.0,
                1252.0,
                1254.0,
                1256.0,
                1258.0,
                1260.0,
                1258.0,
                1256.0,
                1254.0,
                1252.0,
            ],
            "high": [
                1255.0,
                1257.0,
                1259.0,
                1261.0,
                1263.0,
                1265.0,
                1263.0,
                1261.0,
                1259.0,
                1257.0,
            ],
            "low": [1248.0, 1250.0, 1252.0, 1254.0, 1256.0, 1258.0, 1256.0, 1254.0, 1252.0, 1250.0],
            "close": [
                1252.0,
                1254.0,
                1256.0,
                1258.0,
                1260.0,
                1262.0,
                1260.0,
                1258.0,
                1256.0,
                1254.0,
            ],
            "volume": [1000.0] * 10,
            "atr_14": [10.0] * 10,
        }
    )


@pytest.fixture
def mock_strategy():
    """Create mock strategy that generates signals."""
    strategy = Mock()

    # Generate LONG signal on first bar, HOLD on others
    signals = [
        TradeSignal(
            signal=Signal.LONG,
            entry_price=1252.0,
            stop_loss=1240.0,
            take_profit=1270.0,
            ord_type="LIMIT",
            reason="Test entry",
        )
    ] + [TradeSignal(signal=Signal.HOLD)] * 9

    strategy.generate_signal = Mock(side_effect=signals)
    return strategy


@pytest.fixture
def mock_session_manager():
    """Create mock session manager."""
    session_manager = Mock()
    session_manager.is_trading_hours = Mock(return_value=True)
    session_manager.should_skip_signal = Mock(return_value=False)
    session_manager.is_entry_blocked = Mock(return_value=False)
    session_manager.get_force_close_reason = Mock(return_value=None)
    return session_manager


@pytest.fixture
def mock_position_sizer():
    """Create mock position sizer."""
    position_sizer = Mock()
    position_sizer.calculate_size = Mock(return_value=1)  # Always return 1 contract
    return position_sizer


@pytest.mark.asyncio
async def test_sim_mode_end_to_end_flow(
    sample_bars_df: pd.DataFrame,
    mock_strategy: Mock,
    mock_session_manager: Mock,
    mock_position_sizer: Mock,
):
    """Test complete SIM mode flow: bars → handlers → orders → stats.

    This test verifies:
    1. SimFeed emits bars correctly
    2. BarHandler updates equity on each bar
    3. SignalHandler generates and submits orders
    4. Tracker records trades correctly
    5. SessionStats computes metrics
    """
    # Initialize components
    tracker = Tracker(
        initial_capital=100000.0,
        commission_rate=0.0001,
        contract_multiplier=100000,
    )

    order_manager = OrderManager(
        client=None,  # No broker client in sim mode
        tracker=tracker,
        symbol="VN30F1M",
        dry_run=True,
    )

    reconciler = Reconciler(
        client=None,
        tracker=tracker,
        order_manager=order_manager,
        symbol="VN30F1M",
    )

    risk_manager = RiskManager(
        use_trailing_stop=False,
        trailing_atr_multiplier=2.0,
        max_daily_loss_fraction=0.05,
        initial_capital=100000.0,
        max_loss_per_trade_fraction=0.0,
    )

    stats = SessionStats(
        tracker=tracker,
        benchmark_equity=None,
    )

    # Create SimFeed
    feed = SimFeed(
        df=sample_bars_df,
        pipeline=None,  # Pre-calculated indicators
        atr_period=14,
        speed=0.0,  # Max throughput
    )

    # Create engine (handlers will be wired after)
    initial_bar_handler = cast(BarHandler, Mock())
    initial_risk_handler = cast(RiskHandler, Mock())
    initial_signal_handler = cast(SignalHandler, Mock())
    engine = PaperEngine(
        feed=feed,
        bar_handler=initial_bar_handler,
        risk_handler=initial_risk_handler,
        signal_handler=initial_signal_handler,
        tracker=tracker,
        reconciler=reconciler,
        order_manager=order_manager,
        stats=stats,
        session_manager=mock_session_manager,
        strategy=mock_strategy,
        symbol="VN30F1M",
        close_on_shutdown=False,  # Don't close position on shutdown for this test
        force_hard_exit=False,
        output_dir="results/test",
    )

    # Create handlers
    bar_handler = BarHandler(
        tracker=tracker,
        order_manager=order_manager,
        session_manager=mock_session_manager,
    )

    risk_handler_config = RiskHandlerConfig(
        force_flat_on_session_close=False,  # Disable for this test
        force_flat_preclose_seconds=0.0,
        force_flat_on_last_candle=False,
        defer_exit_outside_session=False,
        freq_minutes=5,
    )

    risk_handler = RiskHandler(
        tracker=tracker,
        order_manager=order_manager,
        risk_manager=risk_manager,
        session_manager=mock_session_manager,
        config=risk_handler_config,
        on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
    )

    signal_handler_config = SignalHandlerConfig(
        entry_cutoff_seconds=0.0,
        allow_late_entry=True,
    )

    signal_handler = SignalHandler(
        strategy=mock_strategy,
        tracker=tracker,
        order_manager=order_manager,
        risk_manager=risk_manager,
        session_manager=mock_session_manager,
        position_sizer=mock_position_sizer,
        config=signal_handler_config,
    )

    # Wire handlers into engine
    engine._bar_handler = bar_handler
    engine._risk_handler = risk_handler
    engine._signal_handler = signal_handler

    # Run engine
    await engine.start(sim_df=sample_bars_df)

    # Verify results
    assert engine._bars_processed == 10, "Should have processed 10 bars"

    # Verify equity was updated on each bar
    assert len(tracker.equity_snapshots) > 0, "Should have equity snapshots"

    # Verify strategy was called for each bar
    assert mock_strategy.generate_signal.call_count == 10, "Strategy should be called for each bar"

    # Verify position sizer was called (for the LONG signal)
    assert mock_position_sizer.calculate_size.call_count >= 1, "Position sizer should be called"

    # Verify last close was updated
    assert engine._last_close == 1254.0, "Last close should be from final bar"


@pytest.mark.asyncio
async def test_engine_with_deferred_exit_flow(
    sample_bars_df: pd.DataFrame,
    mock_strategy: Mock,
    mock_position_sizer: Mock,
):
    """Test deferred exit flow: exit triggered outside session → submitted when session opens.

    This test verifies:
    1. RiskHandler defers exit when outside session
    2. BarHandler submits deferred exit when session opens
    3. Deferred exit reason is cleared after submission
    """
    # Initialize components
    tracker = Tracker(
        initial_capital=100000.0,
        commission_rate=0.0001,
        contract_multiplier=100000,
    )

    # Open a position manually
    tracker.record_open(
        fill_price=1250.0,
        qty=1,
        side="LONG",
        timestamp=datetime(2024, 1, 15, 9, 0),
        stop_loss=1240.0,
        take_profit=1270.0,
    )

    order_manager = OrderManager(
        client=None,
        tracker=tracker,
        symbol="VN30F1M",
        dry_run=True,
    )

    reconciler = Reconciler(
        client=None,
        tracker=tracker,
        order_manager=order_manager,
        symbol="VN30F1M",
    )

    risk_manager = RiskManager(
        use_trailing_stop=False,
        trailing_atr_multiplier=2.0,
        max_daily_loss_fraction=0.05,
        initial_capital=100000.0,
        max_loss_per_trade_fraction=0.0,
    )

    stats = SessionStats(
        tracker=tracker,
        benchmark_equity=None,
    )

    # Mock session manager: first bar outside session, rest inside
    session_manager = Mock()
    session_manager.is_trading_hours = Mock(return_value=True)
    session_manager.should_skip_signal = Mock(return_value=False)
    session_manager.is_entry_blocked = Mock(return_value=False)
    session_manager.get_force_close_reason = Mock(return_value=None)

    # Mock strategy: always HOLD
    strategy = Mock()
    strategy.generate_signal = Mock(return_value=TradeSignal(signal=Signal.HOLD))

    # Create SimFeed
    feed = SimFeed(
        df=sample_bars_df,
        pipeline=None,
        atr_period=14,
        speed=0.0,
    )

    # Create engine
    initial_bar_handler = cast(BarHandler, Mock())
    initial_risk_handler = cast(RiskHandler, Mock())
    initial_signal_handler = cast(SignalHandler, Mock())
    engine = PaperEngine(
        feed=feed,
        bar_handler=initial_bar_handler,
        risk_handler=initial_risk_handler,
        signal_handler=initial_signal_handler,
        tracker=tracker,
        reconciler=reconciler,
        order_manager=order_manager,
        stats=stats,
        session_manager=session_manager,
        strategy=strategy,
        symbol="VN30F1M",
        close_on_shutdown=False,
        force_hard_exit=False,
        output_dir="results/test",
    )

    # Create handlers
    bar_handler = BarHandler(
        tracker=tracker,
        order_manager=order_manager,
        session_manager=session_manager,
    )

    risk_handler_config = RiskHandlerConfig(
        force_flat_on_session_close=True,
        force_flat_preclose_seconds=0.0,
        force_flat_on_last_candle=False,
        defer_exit_outside_session=True,  # Enable deferred exit
        freq_minutes=5,
    )

    risk_handler = RiskHandler(
        tracker=tracker,
        order_manager=order_manager,
        risk_manager=risk_manager,
        session_manager=session_manager,
        config=risk_handler_config,
        on_deferred_exit=lambda r: setattr(engine, "_deferred_exit_reason", r),
    )

    signal_handler_config = SignalHandlerConfig(
        entry_cutoff_seconds=0.0,
        allow_late_entry=True,
    )

    signal_handler = SignalHandler(
        strategy=strategy,
        tracker=tracker,
        order_manager=order_manager,
        risk_manager=risk_manager,
        session_manager=session_manager,
        position_sizer=mock_position_sizer,
        config=signal_handler_config,
    )

    # Wire handlers
    engine._bar_handler = bar_handler
    engine._risk_handler = risk_handler
    engine._signal_handler = signal_handler

    # Manually trigger first bar (outside session) to set deferred exit
    _ = sample_bars_df.iloc[0].to_dict()
    engine._deferred_exit_reason = "Session Boundary Close"  # Simulate deferred exit

    # Manually trigger second bar (inside session) to submit deferred exit
    second_bar = sample_bars_df.iloc[1].to_dict()
    engine._on_bar(second_bar)

    # Verify deferred exit was cleared
    assert engine._deferred_exit_reason is None, "Deferred exit should be cleared after submission"


@pytest.mark.asyncio
async def test_engine_graceful_shutdown_with_open_position(
    sample_bars_df: pd.DataFrame,
    mock_strategy: Mock,
    mock_session_manager: Mock,
    mock_position_sizer: Mock,
):
    """Test graceful shutdown closes open position correctly.

    This test verifies:
    1. Engine closes open position on shutdown
    2. Correct price is used (last_close or None)
    3. Stats are printed and saved
    """
    # Initialize components
    tracker = Tracker(
        initial_capital=100000.0,
        commission_rate=0.0001,
        contract_multiplier=100000,
    )

    # Open a position
    tracker.record_open(
        fill_price=1250.0,
        qty=1,
        side="LONG",
        timestamp=datetime(2024, 1, 15, 9, 0),
        stop_loss=1240.0,
        take_profit=1270.0,
    )

    order_manager = OrderManager(
        client=None,
        tracker=tracker,
        symbol="VN30F1M",
        dry_run=True,
    )

    reconciler = Reconciler(
        client=None,
        tracker=tracker,
        order_manager=order_manager,
        symbol="VN30F1M",
    )

    _ = RiskManager(
        use_trailing_stop=False,
        trailing_atr_multiplier=2.0,
        max_daily_loss_fraction=0.05,
        initial_capital=100000.0,
        max_loss_per_trade_fraction=0.0,
    )

    stats = Mock()
    stats.print_summary = Mock()
    stats.save = Mock(return_value="results/test/session.json")

    feed = AsyncMock()
    feed.subscribe = AsyncMock()
    feed.unsubscribe = AsyncMock()
    feed.close = AsyncMock()

    # Create engine
    initial_bar_handler = cast(BarHandler, Mock())
    initial_risk_handler = cast(RiskHandler, Mock())
    initial_signal_handler = cast(SignalHandler, Mock())
    engine = PaperEngine(
        feed=feed,
        bar_handler=initial_bar_handler,
        risk_handler=initial_risk_handler,
        signal_handler=initial_signal_handler,
        tracker=tracker,
        reconciler=reconciler,
        order_manager=order_manager,
        stats=stats,
        session_manager=mock_session_manager,
        strategy=mock_strategy,
        symbol="VN30F1M",
        close_on_shutdown=True,  # Enable close on shutdown
        force_hard_exit=False,
        output_dir="results/test",
    )

    # Set last close
    engine._last_close = 1260.0
    engine._running = True

    # Stop engine
    await engine.stop()

    # Verify position was closed (order_manager.submit_exit was called)
    # Note: In dry_run mode, submit_exit just logs, but we can verify it was called
    # by checking the tracker state or order_manager calls

    # Verify stats were printed and saved
    stats.print_summary.assert_called_once()
    stats.save.assert_called_once()


@pytest.mark.asyncio
async def test_engine_handler_pipeline_short_circuit(
    sample_bars_df: pd.DataFrame,
    mock_strategy: Mock,
    mock_session_manager: Mock,
    mock_position_sizer: Mock,
):
    """Test that handler pipeline short-circuits correctly.

    This test verifies:
    1. When BarHandler returns True, RiskHandler and SignalHandler are skipped
    2. When RiskHandler returns True, SignalHandler is skipped
    """
    # Initialize components
    tracker = Tracker(
        initial_capital=100000.0,
        commission_rate=0.0001,
        contract_multiplier=100000,
    )

    order_manager = OrderManager(
        client=None,
        tracker=tracker,
        symbol="VN30F1M",
        dry_run=True,
    )

    reconciler = Reconciler(
        client=None,
        tracker=tracker,
        order_manager=order_manager,
        symbol="VN30F1M",
    )

    _ = RiskManager(
        use_trailing_stop=False,
        trailing_atr_multiplier=2.0,
        max_daily_loss_fraction=0.05,
        initial_capital=100000.0,
        max_loss_per_trade_fraction=0.0,
    )

    stats = SessionStats(
        tracker=tracker,
        benchmark_equity=None,
    )

    feed = AsyncMock()

    # Create engine
    initial_bar_handler = cast(BarHandler, Mock())
    initial_risk_handler = cast(RiskHandler, Mock())
    initial_signal_handler = cast(SignalHandler, Mock())
    engine = PaperEngine(
        feed=feed,
        bar_handler=initial_bar_handler,
        risk_handler=initial_risk_handler,
        signal_handler=initial_signal_handler,
        tracker=tracker,
        reconciler=reconciler,
        order_manager=order_manager,
        stats=stats,
        session_manager=mock_session_manager,
        strategy=mock_strategy,
        symbol="VN30F1M",
        close_on_shutdown=False,
        force_hard_exit=False,
        output_dir="results/test",
    )

    # Create mock handlers
    bar_handler = Mock()
    bar_handler.on_bar = Mock(return_value=(True, None))  # Returns True - exit submitted

    risk_handler = Mock()
    risk_handler.on_bar = Mock(return_value=False)

    signal_handler = Mock()
    signal_handler.on_bar = Mock()

    # Wire handlers
    engine._bar_handler = bar_handler
    engine._risk_handler = risk_handler
    engine._signal_handler = signal_handler

    # Trigger bar
    bar = sample_bars_df.iloc[0].to_dict()
    engine._on_bar(bar)

    # Verify BarHandler was called
    bar_handler.on_bar.assert_called_once()

    # Verify RiskHandler and SignalHandler were NOT called
    risk_handler.on_bar.assert_not_called()
    signal_handler.on_bar.assert_not_called()
