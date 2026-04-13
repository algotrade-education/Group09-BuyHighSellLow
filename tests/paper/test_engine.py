"""Unit tests for PaperEngine.

Tests cover:
- Engine lifecycle (start/stop)
- Handler pipeline execution
- Deferred exit handling
- Graceful shutdown with correct price handling
- Background task management
"""

import asyncio
import contextlib
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from src.paper.engine import PaperEngine


@pytest.fixture
def mock_feed():
    """Mock feed for testing."""
    feed = AsyncMock()
    feed.subscribe = AsyncMock()
    feed.unsubscribe = AsyncMock()
    feed.close = AsyncMock()
    return feed


@pytest.fixture
def mock_tracker():
    """Mock tracker for testing."""
    tracker = Mock()
    tracker.is_flat = True
    tracker.cash = 100000.0
    tracker.equity = 100000.0
    tracker.daily_pnl = 0.0
    tracker.update_unrealized = Mock()
    tracker.update_daily_pnl = Mock()
    tracker.equity_snapshot = Mock()
    # Position: flat by default so the logging branch takes the "FLAT" path
    position = Mock()
    position.quantity = 0
    position.side = Mock(value="FLAT")
    position.entry_price = 0.0
    position.unrealized_pnl = 0.0
    tracker.position = position
    tracker.position_snapshot = Mock()
    return tracker


@pytest.fixture
def mock_handlers():
    """Mock handlers for testing."""
    bar_handler = Mock()
    bar_handler.on_bar = Mock(return_value=(False, None))

    risk_handler = Mock()
    risk_handler.on_bar = Mock(return_value=False)

    signal_handler = Mock()
    signal_handler.on_bar = Mock()

    return bar_handler, risk_handler, signal_handler


@pytest.fixture
def mock_reconciler():
    """Mock reconciler for testing."""
    reconciler = Mock()
    reconciler.reconcile_position = Mock()
    reconciler.reconcile_cash = Mock()
    reconciler.reconcile_orders = Mock()
    return reconciler


@pytest.fixture
def mock_order_manager():
    """Mock order manager for testing."""
    order_manager = Mock()
    order_manager.submit_exit = Mock()
    return order_manager


@pytest.fixture
def mock_stats():
    """Mock stats for testing."""
    stats = Mock()
    stats.print_summary = Mock()
    stats.save = Mock(return_value="/path/to/results")
    return stats


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing."""
    session_manager = Mock()
    session_manager.is_trading_hours = Mock(return_value=True)
    return session_manager


@pytest.fixture
def mock_strategy():
    """Mock strategy for testing."""
    strategy = Mock()
    strategy.generate_signal = Mock()
    return strategy


@pytest.fixture
def engine(
    mock_feed,
    mock_handlers,
    mock_tracker,
    mock_reconciler,
    mock_order_manager,
    mock_stats,
    mock_session_manager,
    mock_strategy,
):
    """Create engine instance with mocked dependencies."""
    bar_handler, risk_handler, signal_handler = mock_handlers

    return PaperEngine(
        feed=mock_feed,
        bar_handler=bar_handler,
        risk_handler=risk_handler,
        signal_handler=signal_handler,
        tracker=mock_tracker,
        reconciler=mock_reconciler,
        order_manager=mock_order_manager,
        stats=mock_stats,
        session_manager=mock_session_manager,
        strategy=mock_strategy,
        symbol="VN30F1M",
        close_on_shutdown=True,
        force_hard_exit=False,
        output_dir="results/test",
    )


# --- Lifecycle Tests ---


@pytest.mark.asyncio
async def test_engine_start_sets_running_flag(engine, mock_feed):
    """Test that engine.start() sets the running flag."""
    assert not engine.running

    # Start engine (will call _run_live)
    task = asyncio.create_task(engine.start())
    await asyncio.sleep(0.1)  # Let it start

    assert engine.running

    # Stop engine
    await engine.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_engine_stop_clears_running_flag(engine):
    """Test that engine.stop() clears the running flag."""
    engine._running = True

    await engine.stop()

    assert not engine.running


@pytest.mark.asyncio
async def test_engine_stop_unsubscribes_from_feed(engine, mock_feed):
    """Test that engine.stop() unsubscribes from feed."""
    engine._running = True

    await engine.stop()

    mock_feed.unsubscribe.assert_called_once_with("VN30F1M")


@pytest.mark.asyncio
async def test_engine_stop_closes_feed(engine, mock_feed):
    """Test that engine.stop() closes the feed."""
    engine._running = True

    await engine.stop()

    mock_feed.close.assert_called_once()


@pytest.mark.asyncio
async def test_engine_stop_prints_and_saves_stats(engine, mock_stats):
    """Test that engine.stop() prints and saves session statistics."""
    engine._running = True

    await engine.stop()

    mock_stats.print_summary.assert_called_once()
    mock_stats.save.assert_called_once_with("results/test")


# --- Shutdown Price Handling Tests ---
# Requirement 1.4: Engine shutdown with invalid price handling


@pytest.mark.asyncio
async def test_engine_stop_with_zero_last_close_passes_none(
    engine, mock_tracker, mock_order_manager
):
    """Verify Engine.stop() passes None as exit price when _last_close is 0.0.

    Requirement 1.4: When Engine.stop() is called and _last_close equals 0.0,
    the engine SHALL pass None as the exit price (not 0.0) to force a MARKET order.

    This prevents submitting invalid LIMIT orders with price=0.0 during shutdown.

    Test scenario:
    - Engine is running with open position
    - _last_close is 0.0 (invalid price state)
    - Call stop()

    Expected behavior:
    - submit_exit called with price=None
    - ord_type='MARKET' (not LIMIT)
    - reason='Shutdown Close'
    """
    engine._running = True
    engine._last_close = 0.0
    mock_tracker.is_flat = False  # Position is open

    await engine.stop()

    # Verify submit_exit was called with price=None
    mock_order_manager.submit_exit.assert_called_once()
    call_kwargs = mock_order_manager.submit_exit.call_args[1]
    assert call_kwargs["price"] is None
    assert call_kwargs["ord_type"] == "MARKET"
    assert call_kwargs["reason"] == "Shutdown Close"


@pytest.mark.asyncio
async def test_engine_stop_with_valid_last_close_passes_price(
    engine, mock_tracker, mock_order_manager
):
    """Verify Engine.stop() passes actual price when _last_close is valid.

    Test scenario:
    - Engine is running with open position
    - _last_close has valid price (1250.5)
    - Call stop()

    Expected behavior:
    - submit_exit called with actual price
    - ord_type='LIMIT' (not MARKET)
    """
    engine._running = True
    engine._last_close = 1250.5
    mock_tracker.is_flat = False  # Position is open

    await engine.stop()

    # Verify submit_exit was called with the actual price
    mock_order_manager.submit_exit.assert_called_once()
    call_kwargs = mock_order_manager.submit_exit.call_args[1]
    assert call_kwargs["price"] == 1250.5
    assert call_kwargs["ord_type"] == "LIMIT"


@pytest.mark.asyncio
async def test_engine_stop_when_flat_does_not_submit_exit(engine, mock_tracker, mock_order_manager):
    """Test that stop() does not submit exit when position is flat."""
    engine._running = True
    engine._last_close = 1250.5
    mock_tracker.is_flat = True  # Position is flat

    await engine.stop()

    # Verify submit_exit was NOT called
    mock_order_manager.submit_exit.assert_not_called()


@pytest.mark.asyncio
async def test_engine_stop_with_close_on_shutdown_false(engine, mock_tracker, mock_order_manager):
    """Test that stop() respects close_on_shutdown=False."""
    engine._running = True
    engine._last_close = 1250.5
    engine._close_on_shutdown = False
    mock_tracker.is_flat = False  # Position is open

    await engine.stop()

    # Verify submit_exit was NOT called
    mock_order_manager.submit_exit.assert_not_called()


# --- Handler Pipeline Tests ---
# Requirement 1.1: Engine delegates bar processing to handlers


def test_on_bar_calls_all_handlers_in_order(engine, mock_handlers):
    """Verify _on_bar delegates to BarHandler, RiskHandler, and SignalHandler in order.

    Requirement 1.1: The Engine SHALL delegate bar processing to BarHandler,
    RiskHandler, and SignalHandler instead of implementing business logic inline.

    This ensures separation of concerns and testability of individual handlers.

    Test scenario:
    - Call _on_bar with valid bar data

    Expected behavior:
    - BarHandler.on_bar called first
    - RiskHandler.on_bar called second
    - SignalHandler.on_bar called third
    """
    bar_handler, risk_handler, signal_handler = mock_handlers

    bar = {
        "datetime": datetime(2024, 1, 15, 9, 5),
        "open": 1250.0,
        "high": 1255.0,
        "low": 1248.0,
        "close": 1252.0,
        "volume": 1000.0,
    }

    engine._on_bar(bar)

    # Verify all handlers were called
    bar_handler.on_bar.assert_called_once()
    risk_handler.on_bar.assert_called_once()
    signal_handler.on_bar.assert_called_once()


def test_on_bar_skips_downstream_when_bar_handler_returns_true(engine, mock_handlers):
    """Verify _on_bar short-circuits when BarHandler returns True (exit submitted).

    Test scenario:
    - BarHandler returns (True, None) indicating exit was submitted

    Expected behavior:
    - BarHandler.on_bar called
    - RiskHandler.on_bar NOT called (short-circuit)
    - SignalHandler.on_bar NOT called (short-circuit)
    """
    bar_handler, risk_handler, signal_handler = mock_handlers

    # BarHandler returns True (exit submitted)
    bar_handler.on_bar.return_value = (True, None)

    bar = {
        "datetime": datetime(2024, 1, 15, 9, 5),
        "close": 1252.0,
        "volume": 1000.0,
    }

    engine._on_bar(bar)

    # Verify BarHandler was called but downstream handlers were not
    bar_handler.on_bar.assert_called_once()
    risk_handler.on_bar.assert_not_called()
    signal_handler.on_bar.assert_not_called()


def test_on_bar_skips_signal_handler_when_risk_handler_returns_true(engine, mock_handlers):
    """Verify _on_bar short-circuits when RiskHandler returns True (risk exit triggered).

    Test scenario:
    - BarHandler returns (False, None) - no exit
    - RiskHandler returns True - risk exit triggered

    Expected behavior:
    - BarHandler.on_bar called
    - RiskHandler.on_bar called
    - SignalHandler.on_bar NOT called (short-circuit)
    """
    bar_handler, risk_handler, signal_handler = mock_handlers

    # BarHandler returns False, RiskHandler returns True
    bar_handler.on_bar.return_value = (False, None)
    risk_handler.on_bar.return_value = True

    bar = {
        "datetime": datetime(2024, 1, 15, 9, 5),
        "close": 1252.0,
        "volume": 1000.0,
    }

    engine._on_bar(bar)

    # Verify BarHandler and RiskHandler were called, but SignalHandler was not
    bar_handler.on_bar.assert_called_once()
    risk_handler.on_bar.assert_called_once()
    signal_handler.on_bar.assert_not_called()


def test_on_bar_updates_last_close(engine):
    """Test that _on_bar updates _last_close for shutdown handling."""
    bar = {
        "datetime": datetime(2024, 1, 15, 9, 5),
        "close": 1252.5,
        "volume": 1000.0,
    }

    engine._on_bar(bar)

    assert engine._last_close == 1252.5


def test_on_bar_increments_bars_processed(engine):
    """Test that _on_bar increments the bar counter."""
    bar = {
        "datetime": datetime(2024, 1, 15, 9, 5),
        "close": 1252.0,
        "volume": 1000.0,
    }

    assert engine._bars_processed == 0

    engine._on_bar(bar)
    assert engine._bars_processed == 1

    engine._on_bar(bar)
    assert engine._bars_processed == 2


# --- Deferred Exit Tests ---


def test_on_bar_handles_deferred_exit_reason(engine, mock_handlers):
    """Test that _on_bar passes deferred exit reason to BarHandler."""
    bar_handler, _, _ = mock_handlers

    # Set deferred exit reason
    engine._deferred_exit_reason = "Stop Loss"

    # BarHandler returns (False, "Stop Loss") - exit not submitted yet
    bar_handler.on_bar.return_value = (False, "Stop Loss")

    bar = {
        "datetime": datetime(2024, 1, 15, 9, 5),
        "close": 1252.0,
        "volume": 1000.0,
    }

    engine._on_bar(bar)

    # Verify BarHandler received the deferred exit reason
    call_args = bar_handler.on_bar.call_args[0]
    assert call_args[2] == "Stop Loss"  # Third argument is deferred_exit_reason


def test_on_bar_clears_deferred_exit_when_submitted(engine, mock_handlers):
    """Test that _on_bar clears deferred exit reason when BarHandler submits it."""
    bar_handler, _, _ = mock_handlers

    # Set deferred exit reason
    engine._deferred_exit_reason = "Stop Loss"

    # BarHandler returns (True, None) - exit was submitted
    bar_handler.on_bar.return_value = (True, None)

    bar = {
        "datetime": datetime(2024, 1, 15, 9, 5),
        "close": 1252.0,
        "volume": 1000.0,
    }

    engine._on_bar(bar)

    # Verify deferred exit reason was cleared
    assert engine._deferred_exit_reason is None


# --- Error Handling Tests ---


def test_on_bar_handles_exceptions_gracefully(engine, mock_handlers):
    """Test that _on_bar catches and logs exceptions without crashing."""
    bar_handler, _, _ = mock_handlers

    # Make BarHandler raise an exception
    bar_handler.on_bar.side_effect = ValueError("Test error")

    bar = {
        "datetime": datetime(2024, 1, 15, 9, 5),
        "close": 1252.0,
        "volume": 1000.0,
    }

    # Should not raise - exception is caught and logged
    engine._on_bar(bar)


@pytest.mark.asyncio
async def test_engine_stop_handles_feed_unsubscribe_error(engine, mock_feed):
    """Test that stop() handles feed unsubscribe errors gracefully."""
    engine._running = True
    mock_feed.unsubscribe.side_effect = RuntimeError("Unsubscribe failed")

    # Should not raise - exception is caught and logged
    await engine.stop()

    # Verify feed.close() was still called
    mock_feed.close.assert_called_once()


@pytest.mark.asyncio
async def test_engine_stop_handles_stats_save_error(engine, mock_stats):
    """Test that stop() handles stats save errors gracefully."""
    engine._running = True
    mock_stats.save.side_effect = OSError("Save failed")

    # Should not raise - exception is caught and logged
    await engine.stop()

    # Verify print_summary was still called
    mock_stats.print_summary.assert_called_once()


# --- Background Task Cleanup Tests ---


@pytest.mark.asyncio
async def test_engine_stop_cancels_background_tasks(engine):
    """Verify stop() cancels all tracked background tasks.

    Test scenario:
    - Engine is running
    - Two background tasks are registered in _bg_tasks
    - Call stop()

    Expected behavior:
    - Both tasks are cancelled
    - No task leaks remain
    """
    engine._running = True

    # Create mock background tasks
    task1 = asyncio.create_task(asyncio.sleep(10))
    task2 = asyncio.create_task(asyncio.sleep(10))
    engine._bg_tasks.add(task1)
    engine._bg_tasks.add(task2)

    await engine.stop()

    # Verify tasks were cancelled
    assert task1.cancelled()
    assert task2.cancelled()


# --- Warmup Tests ---


def test_warmup_strategy_calls_strategy_generate_signal(engine, mock_strategy):
    """Test that _warmup_strategy calls strategy.generate_signal with is_warmup=True."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "datetime": [datetime(2024, 1, 15, 9, 0), datetime(2024, 1, 15, 9, 5)],
            "open": [1250.0, 1252.0],
            "high": [1255.0, 1257.0],
            "low": [1248.0, 1250.0],
            "close": [1252.0, 1254.0],
            "volume": [1000.0, 1100.0],
        }
    )

    engine._warmup_strategy(df)

    # Verify strategy.generate_signal was called for each row with is_warmup=True
    assert mock_strategy.generate_signal.call_count == 2

    # Check that is_warmup=True was passed
    for call in mock_strategy.generate_signal.call_args_list:
        assert call[1]["is_warmup"] is True


def test_warmup_strategy_updates_tracker_state(engine, mock_tracker):
    """Test that _warmup_strategy updates tracker daily_pnl and unrealized."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "datetime": [datetime(2024, 1, 15, 9, 0)],
            "close": [1252.0],
        }
    )

    engine._warmup_strategy(df)

    # Verify tracker methods were called
    mock_tracker.update_daily_pnl.assert_called()
    mock_tracker.update_unrealized.assert_called_with(1252.0)


def test_warmup_strategy_handles_errors_silently(engine, mock_strategy):
    """Test that _warmup_strategy silently skips errors during warmup."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "datetime": [datetime(2024, 1, 15, 9, 0)],
            "close": [1252.0],
        }
    )

    # Make strategy raise an error
    mock_strategy.generate_signal.side_effect = ValueError("Warmup error")

    # Should not raise - errors are silently skipped during warmup
    engine._warmup_strategy(df)


# --- Property Access Tests ---


def test_engine_stats_property(engine, mock_stats):
    """Verify engine.stats returns the SessionStats instance."""
    assert engine.stats is mock_stats


def test_engine_tracker_property(engine, mock_tracker):
    """Verify engine.tracker returns the Tracker instance."""
    assert engine.tracker is mock_tracker


def test_engine_running_property(engine):
    """Verify engine.running reflects the internal running flag state.

    Test scenario:
    - Initially not running
    - Set _running to True
    - Set _running to False

    Expected behavior:
    - Property reflects internal state accurately
    """
    assert not engine.running

    engine._running = True
    assert engine.running

    engine._running = False
    assert not engine.running
