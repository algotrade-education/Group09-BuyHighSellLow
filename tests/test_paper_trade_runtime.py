"""
Scenario tests for PaperTrader runtime behavior.

These tests validate real paper-trade control-flow situations requested by users:
- chạy đầy đủ (sim run end-to-end)
- chạy xong bỏ / stop giữa chừng (cancellation)
- chạy tới cuối phiên: có vị thế và không vị thế
- defer exit khi ngoài phiên rồi đóng lại khi vào phiên
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Avoid import-time DB config errors when src package is imported in tests
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_NAME", "test")


class HoldStrategy:
    """Simple strategy stub that always returns HOLD."""

    def generate_signal(self, bar, current_position=None, is_warmup=False):
        from src.strategy.base import Signal, TradeSignal

        return TradeSignal(signal=Signal.HOLD, reason="test hold")


class SignalStrategy:
    def __init__(self, signal_name="HOLD", ord_type="LIMIT", should_raise=False):
        self.signal_name = signal_name
        self.ord_type = ord_type
        self.should_raise = should_raise
        self.calls = []

    def generate_signal(self, bar, current_position=None, is_warmup=False):
        self.calls.append({"bar": bar, "is_warmup": is_warmup})
        if self.should_raise:
            raise RuntimeError("strategy boom")

        from src.strategy.base import Signal, TradeSignal

        return TradeSignal(
            signal=Signal[self.signal_name],
            ord_type=self.ord_type,
            reason=f"test {self.signal_name.lower()}",
        )


def _make_bar(dt=None, close=1302.0):
    return {
        "datetime": dt or datetime(2025, 1, 2, 10, 0, 0),
        "open": close - 1,
        "high": close + 4,
        "low": close - 4,
        "close": close,
        "volume": 100,
        "atr_14": 5.0,
    }


class FakeDataFrame:
    def __init__(self, records):
        self._records = list(records)

    @property
    def empty(self):
        return len(self._records) == 0

    def __len__(self):
        return len(self._records)

    def to_dict(self, orient="records"):
        if orient != "records":
            raise ValueError("Unsupported orient")
        return list(self._records)


def _make_config(**risk_overrides):
    risk = {
        "min_position_size": 1,
        "max_position_size": 1,
        "risk_per_trade_pct": 0.0,
        "max_daily_loss": 0.0,
    }
    risk.update(risk_overrides)
    return {
        "strategy": {"atr_period": 14},
        "risk": risk,
    }


def _make_trader(**risk_overrides):
    config = _make_config(**risk_overrides)
    from src.paper.engine import PaperTrader

    return PaperTrader(
        strategy=HoldStrategy(),
        symbol="HNXDS:VN30F2601",
        config=config,
        client=None,
        redis_client=None,
        bar_freq="5min",
        dry_run=True,
    )


class TestPaperTraderRuntimeScenarios:
    def test_start_sim_runs_replay_and_stop(self):
        """Chạy đầy đủ sim: start(sim_df) phải replay xong và stop()."""
        trader = _make_trader()

        replay_mock = AsyncMock()
        stop_mock = AsyncMock()
        trader._bar_provider.replay = replay_mock
        trader.stop = stop_mock

        class SimBars(list):
            pass

        sim_df = SimBars([1, 2, 3])

        asyncio.run(trader.start(sim_df=sim_df))

        replay_mock.assert_awaited_once_with(sim_df, speed=0.0)
        stop_mock.assert_awaited_once()

    def test_run_live_cancelled_calls_stop(self):
        """Chạy tới giữa chừng rồi bỏ: CancelledError trong live loop vẫn gọi stop()."""
        trader = _make_trader()

        trader._client = MagicMock()
        trader._redis_client = MagicMock()
        trader._redis_client.subscribe = AsyncMock()

        async def raise_cancel(_seconds):
            raise asyncio.CancelledError

        stop_mock = AsyncMock()
        trader.stop = stop_mock

        original_sleep = asyncio.sleep
        try:
            asyncio.sleep = raise_cancel
            asyncio.run(trader._run_live())
        finally:
            asyncio.sleep = original_sleep

        stop_mock.assert_awaited_once()

    def test_stop_with_open_position_and_close_on_shutdown_submits_exit(self):
        """Cuối phiên/stop có vị thế + close_on_shutdown=True -> phải gửi lệnh thoát."""
        trader = _make_trader(close_on_shutdown=True)
        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 10, 0, 0),
        )
        trader._last_close = 1312.0

        submit_exit = MagicMock()
        trader._order_mgr.submit_exit = submit_exit

        asyncio.run(trader.stop())

        submit_exit.assert_called_once_with(reason="Shutdown Close", price=1312.0)

    def test_stop_with_open_position_and_close_on_shutdown_disabled_keeps_position(self):
        """Stop có vị thế + close_on_shutdown=False -> không tự đóng lệnh."""
        trader = _make_trader(close_on_shutdown=False)
        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 10, 0, 0),
        )

        submit_exit = MagicMock()
        trader._order_mgr.submit_exit = submit_exit

        asyncio.run(trader.stop())

        submit_exit.assert_not_called()
        assert not trader._tracker.is_flat

    def test_stop_when_flat_does_not_submit_exit(self):
        """Stop ở trạng thái không vị thế -> không gọi submit_exit."""
        trader = _make_trader(close_on_shutdown=True)

        submit_exit = MagicMock()
        trader._order_mgr.submit_exit = submit_exit

        asyncio.run(trader.stop())

        submit_exit.assert_not_called()

    def test_on_new_bar_outside_session_defers_exit_then_reopen_submits_deferred_exit(self):
        """Ngoài phiên có vị thế -> defer exit; vào lại phiên -> submit deferred exit."""
        trader = _make_trader(
            force_flat_on_session_close=True,
            defer_exit_outside_session=True,
        )

        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 10, 0, 0),
        )

        submit_exit = MagicMock()
        trader._order_mgr.submit_exit = submit_exit

        class ToggleSession:
            def __init__(self):
                self.trading = False

            def is_trading_hours(self, _dt):
                return self.trading

            def get_force_close_reason(self, dt, preclose_seconds):
                return None

            def is_entry_blocked(self, dt, cutoff_seconds, allow_late=False):
                return False

        trader._session_mgr = ToggleSession()

        bar = {
            "datetime": datetime(2025, 1, 2, 11, 31, 0),
            "open": 1301.0,
            "high": 1305.0,
            "low": 1298.0,
            "close": 1302.0,
            "volume": 100,
            "atr_14": 5.0,
        }

        # outside session -> defer
        trader._on_new_bar(bar)
        assert trader._deferred_exit_reason == "Session Boundary Close"
        submit_exit.assert_not_called()

        # reopen session -> execute deferred exit
        trader._session_mgr.trading = True
        trader._on_new_bar(bar)

        assert trader._deferred_exit_reason is None
        submit_exit.assert_called_once()
        called_kwargs = submit_exit.call_args.kwargs
        assert called_kwargs["reason"] == "Session Boundary Close"
        assert called_kwargs["price"] == 1302.0

    def test_on_new_bar_preclose_force_close_when_holding_position(self):
        """Tới cuối phiên và đang có vị thế -> force close theo tín hiệu preclose."""
        trader = _make_trader(force_flat_preclose_seconds=600)

        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 14, 20, 0),
        )

        class LastCandleSession:
            def is_trading_hours(self, dt):
                return dt.hour < 15

            def get_force_close_reason(self, dt, preclose_seconds):
                return "Session Preclose (test)"

            def is_entry_blocked(self, dt, cutoff_seconds, allow_late=False):
                return False

        trader._session_mgr = LastCandleSession()

        submit_exit = MagicMock()
        trader._order_mgr.submit_exit = submit_exit

        bar = {
            "datetime": datetime(2025, 1, 2, 14, 25, 0),
            "open": 1301.0,
            "high": 1306.0,
            "low": 1299.0,
            "close": 1304.0,
            "volume": 100,
            "atr_14": 5.0,
        }

        trader._on_new_bar(bar)

        submit_exit.assert_called_once()
        kwargs = submit_exit.call_args.kwargs
        assert kwargs["reason"] == "Session Preclose (test)"
        assert kwargs["price"] == 1304.0

    def test_on_new_bar_last_candle_when_flat_does_not_force_close(self):
        """Tới nến cuối phiên nhưng không có vị thế -> không gửi lệnh thoát."""
        trader = _make_trader(force_flat_on_last_candle=True)

        class LastCandleSession:
            def is_trading_hours(self, dt):
                return dt.hour < 14 or (dt.hour == 14 and dt.minute < 30)

            def get_force_close_reason(self, dt, preclose_seconds):
                return None

            def is_entry_blocked(self, dt, cutoff_seconds, allow_late=False):
                return False

        trader._session_mgr = LastCandleSession()
        trader._order_mgr.submit_exit = MagicMock()

        bar = {
            "datetime": datetime(2025, 1, 2, 14, 25, 0),
            "open": 1301.0,
            "high": 1306.0,
            "low": 1299.0,
            "close": 1304.0,
            "volume": 100,
            "atr_14": 5.0,
        }

        trader._on_new_bar(bar)

        trader._order_mgr.submit_exit.assert_not_called()

    def test_start_without_sim_routes_to_live(self):
        trader = _make_trader()
        run_live = AsyncMock()
        trader._run_live = run_live

        asyncio.run(trader.start(sim_df=None))

        run_live.assert_awaited_once_with(None, None)

    def test_run_live_warmup_sync_and_subscription_flow(self):
        strategy = SignalStrategy("HOLD")
        from src.paper.engine import PaperTrader

        trader = PaperTrader(
            strategy=strategy,
            symbol="HNXDS:VN30F2601",
            config=_make_config(),
            client=MagicMock(),
            redis_client=MagicMock(),
            bar_freq="5min",
            dry_run=True,
        )

        preload = MagicMock()
        seed = MagicMock()
        check_time = MagicMock()
        trader._bar_provider.preload_history = preload
        trader._bar_provider.seed_current_live_bar = seed
        trader._bar_provider.check_time = check_time

        sync_state = MagicMock()
        trader._sync_broker_state = sync_state

        trader._redis_client.subscribe = AsyncMock()
        stop_mock = AsyncMock()
        trader.stop = stop_mock
        trader._running = True

        historical_df = FakeDataFrame([_make_bar(close=1300.0), _make_bar(close=1301.0)])
        incomplete_bar = _make_bar(close=1302.0)

        async def raise_cancel(_seconds):
            raise asyncio.CancelledError

        original_sleep = asyncio.sleep
        try:
            asyncio.sleep = raise_cancel
            asyncio.run(
                trader._run_live(
                    historical_df=historical_df,
                    incomplete_bar=incomplete_bar,
                )
            )
        finally:
            asyncio.sleep = original_sleep

        preload.assert_called_once_with(historical_df)
        seed.assert_called_once_with(incomplete_bar)
        sync_state.assert_called_once()
        trader._redis_client.subscribe.assert_awaited_once()
        assert any(call["is_warmup"] for call in strategy.calls)
        check_time.assert_called_once()
        stop_mock.assert_awaited_once()

    def test_run_live_redis_subscribe_failure_returns_without_stop(self):
        trader = _make_trader()
        trader._client = MagicMock()
        trader._redis_client = MagicMock()
        trader._redis_client.subscribe = AsyncMock(side_effect=RuntimeError("redis down"))
        stop_mock = AsyncMock()
        trader.stop = stop_mock

        asyncio.run(trader._run_live())

        stop_mock.assert_not_awaited()

    def test_redis_quote_callback_schedules_coroutine(self):
        trader = _make_trader()
        scheduled = object()
        trader._bar_provider.on_quote = MagicMock(return_value=scheduled)

        with (
            patch("src.paper.engine.asyncio.iscoroutine", return_value=True),
            patch("src.paper.engine.asyncio.create_task") as create_task_mock,
        ):
            snapshot = MagicMock()
            snapshot.instrument = "HNXDS:VN30F2601"
            trader._redis_quote_callback(snapshot)

        trader._bar_provider.on_quote.assert_called_once()
        create_task_mock.assert_called_once_with(scheduled)

    def test_on_new_bar_risk_exit_has_priority_over_strategy_entry(self):
        strategy = SignalStrategy("LONG")
        from src.paper.engine import PaperTrader

        trader = PaperTrader(
            strategy=strategy,
            symbol="HNXDS:VN30F2601",
            config=_make_config(),
            client=None,
            redis_client=None,
            bar_freq="5min",
            dry_run=True,
        )

        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 10, 0, 0),
        )

        trader._risk_mgr.get_exit_trigger = MagicMock(return_value="Stop Loss")
        trader._order_mgr.submit_exit = MagicMock()
        trader._order_mgr.submit_entry = MagicMock()

        class InSession:
            def is_trading_hours(self, _dt):
                return True

            def get_force_close_reason(self, dt, preclose_seconds):
                return None

            def is_entry_blocked(self, dt, cutoff_seconds, allow_late=False):
                return False

        trader._session_mgr = InSession()

        trader._on_new_bar(_make_bar(close=1290.0))

        trader._order_mgr.submit_exit.assert_called_once()
        trader._order_mgr.submit_entry.assert_not_called()

    def test_on_new_bar_daily_loss_hit_skips_strategy_generation(self):
        strategy = SignalStrategy("LONG")
        from src.paper.engine import PaperTrader

        trader = PaperTrader(
            strategy=strategy,
            symbol="HNXDS:VN30F2601",
            config=_make_config(),
            client=None,
            redis_client=None,
            bar_freq="5min",
            dry_run=True,
        )

        trader._risk_mgr.is_daily_loss_hit = MagicMock(return_value=True)
        trader._order_mgr.submit_entry = MagicMock()

        trader._on_new_bar(_make_bar(close=1300.0))

        assert strategy.calls == []
        trader._order_mgr.submit_entry.assert_not_called()

    def test_on_new_bar_strategy_exception_is_handled_without_orders(self):
        strategy = SignalStrategy("HOLD", should_raise=True)
        from src.paper.engine import PaperTrader

        trader = PaperTrader(
            strategy=strategy,
            symbol="HNXDS:VN30F2601",
            config=_make_config(),
            client=None,
            redis_client=None,
            bar_freq="5min",
            dry_run=True,
        )
        trader._order_mgr.submit_entry = MagicMock()
        trader._order_mgr.submit_exit = MagicMock()

        trader._on_new_bar(_make_bar(close=1305.0))

        trader._order_mgr.submit_entry.assert_not_called()
        trader._order_mgr.submit_exit.assert_not_called()

    def test_on_new_bar_entry_blocked_by_cutoff(self):
        strategy = SignalStrategy("LONG")
        from src.paper.engine import PaperTrader

        trader = PaperTrader(
            strategy=strategy,
            symbol="HNXDS:VN30F2601",
            config=_make_config(entry_cutoff_seconds=3600),
            client=None,
            redis_client=None,
            bar_freq="5min",
            dry_run=True,
        )

        class BlockedSession:
            def is_trading_hours(self, _dt):
                return True

            def get_force_close_reason(self, dt, preclose_seconds):
                return None

            def is_entry_blocked(self, dt, cutoff_seconds, allow_late=False):
                return True

        trader._session_mgr = BlockedSession()
        trader._order_mgr.submit_entry = MagicMock()

        trader._on_new_bar(_make_bar(close=1306.0))

        trader._order_mgr.submit_entry.assert_not_called()

    def test_on_new_bar_long_signal_submits_entry(self):
        strategy = SignalStrategy("LONG")
        from src.paper.engine import PaperTrader

        trader = PaperTrader(
            strategy=strategy,
            symbol="HNXDS:VN30F2601",
            config=_make_config(min_position_size=2, max_position_size=2),
            client=None,
            redis_client=None,
            bar_freq="5min",
            dry_run=True,
        )

        class OpenSession:
            def is_trading_hours(self, _dt):
                return True

            def get_force_close_reason(self, dt, preclose_seconds):
                return None

            def is_entry_blocked(self, dt, cutoff_seconds, allow_late=False):
                return False

        trader._session_mgr = OpenSession()
        trader._order_mgr.submit_entry = MagicMock()

        bar = _make_bar(close=1304.0)
        trader._on_new_bar(bar)

        trader._order_mgr.submit_entry.assert_called_once()
        _, kwargs = trader._order_mgr.submit_entry.call_args
        assert kwargs["qty"] == 2
        assert kwargs["bar"] == bar

    def test_on_new_bar_close_signal_submits_exit(self):
        strategy = SignalStrategy("CLOSE", ord_type="MARKET")
        from src.paper.engine import PaperTrader

        trader = PaperTrader(
            strategy=strategy,
            symbol="HNXDS:VN30F2601",
            config=_make_config(),
            client=None,
            redis_client=None,
            bar_freq="5min",
            dry_run=True,
        )

        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 10, 0, 0),
        )
        trader._order_mgr.submit_exit = MagicMock()

        trader._on_new_bar(_make_bar(close=1303.0))

        trader._order_mgr.submit_exit.assert_called_once()
        kwargs = trader._order_mgr.submit_exit.call_args.kwargs
        assert kwargs["reason"] == "Strategy Close"
        assert kwargs["ord_type"] == "MARKET"

    def test_submit_entry_skips_when_position_sizer_returns_zero(self):
        from src.strategy.base import Signal, TradeSignal

        trader = _make_trader()
        trader._position_sizer.calculate_size = MagicMock(return_value=0)
        trader._order_mgr.submit_entry = MagicMock()

        signal = TradeSignal(signal=Signal.LONG, entry_price=1300.0)
        trader._submit_entry(
            signal,
            _make_bar(close=1300.0),
            datetime(2025, 1, 2, 10, 0, 0),
        )

        trader._order_mgr.submit_entry.assert_not_called()

    def test_submit_exit_or_defer_in_session_submits_immediately(self):
        trader = _make_trader()
        trader._order_mgr.submit_exit = MagicMock()
        trader._deferred_exit_reason = "old"

        class InSession:
            def is_trading_hours(self, _dt):
                return True

        trader._session_mgr = InSession()

        trader._submit_exit_or_defer(
            reason="Stop Loss",
            price=1290.0,
            process_time=datetime(2025, 1, 2, 10, 0, 0),
            ord_type="MARKET",
        )

        trader._order_mgr.submit_exit.assert_called_once()
        kwargs = trader._order_mgr.submit_exit.call_args.kwargs
        assert kwargs["reason"] == "Stop Loss"
        assert kwargs["price"] == 1290.0
        assert kwargs["ord_type"] == "MARKET"
        assert trader._deferred_exit_reason is None

    def test_submit_exit_or_defer_outside_session_sets_deferred_reason(self):
        trader = _make_trader(defer_exit_outside_session=True)
        trader._order_mgr.submit_exit = MagicMock()

        class OutSession:
            def is_trading_hours(self, _dt):
                return False

        trader._session_mgr = OutSession()

        trader._submit_exit_or_defer(
            reason="Session Boundary Close",
            price=1301.0,
            process_time=datetime(2025, 1, 2, 11, 45, 0),
            ord_type="MARKET",
        )

        trader._order_mgr.submit_exit.assert_not_called()
        assert trader._deferred_exit_reason == "Session Boundary Close"

    def test_submit_exit_or_defer_outside_session_without_defer_skips(self):
        trader = _make_trader(defer_exit_outside_session=False)
        trader._order_mgr.submit_exit = MagicMock()

        class OutSession:
            def is_trading_hours(self, _dt):
                return False

        trader._session_mgr = OutSession()

        trader._submit_exit_or_defer(
            reason="Stop Loss",
            price=1299.0,
            process_time=datetime(2025, 1, 2, 11, 45, 0),
            ord_type="MARKET",
        )

        trader._order_mgr.submit_exit.assert_not_called()
        assert trader._deferred_exit_reason is None

    def test_maybe_force_flat_by_clock_skips_when_last_close_unavailable(self):
        trader = _make_trader(force_flat_preclose_seconds=60)
        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 14, 20, 0),
        )
        trader._last_close = 0.0

        class Session:
            def get_force_close_reason(self, dt, preclose_seconds):
                return "Session Preclose"

        trader._session_mgr = Session()
        trader._order_mgr.submit_exit = MagicMock()

        trader._maybe_force_flat_by_clock(datetime(2025, 1, 2, 14, 29, 0))

        trader._order_mgr.submit_exit.assert_not_called()

    def test_maybe_force_flat_by_clock_submits_exit_when_triggered(self):
        trader = _make_trader(force_flat_preclose_seconds=60)
        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 14, 20, 0),
        )
        trader._last_close = 1307.0

        class Session:
            def get_force_close_reason(self, dt, preclose_seconds):
                return "Session Preclose"

            def is_trading_hours(self, _dt):
                return True

        trader._session_mgr = Session()
        trader._order_mgr.submit_exit = MagicMock()

        trader._maybe_force_flat_by_clock(datetime(2025, 1, 2, 14, 29, 0))

        trader._order_mgr.submit_exit.assert_called_once()
        kwargs = trader._order_mgr.submit_exit.call_args.kwargs
        assert kwargs["reason"] == "Session Preclose"
        assert kwargs["price"] == 1307.0
        assert kwargs["ord_type"] == "MARKET"

    def test_maybe_force_flat_by_clock_triggers_atc_safety_close(self):
        from src.engine.session_manager import VN30Session

        trader = _make_trader(
            force_flat_on_session_close=True,
            force_flat_preclose_seconds=15,
        )
        trader._session_mgr = VN30Session()
        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="SHORT",
            timestamp=datetime(2025, 1, 2, 14, 20, 0),
        )
        trader._last_close = 1297.0
        trader._order_mgr.submit_exit = MagicMock()

        trader._maybe_force_flat_by_clock(datetime(2025, 1, 2, 14, 31, 0))

        trader._order_mgr.submit_exit.assert_called_once()
        kwargs = trader._order_mgr.submit_exit.call_args.kwargs
        assert kwargs["reason"] == "ATC Safety Close"
        assert kwargs["price"] == 1297.0
        assert kwargs["ord_type"] == "MARKET"

    def test_on_new_bar_in_atc_triggers_safety_close(self):
        from src.engine.session_manager import VN30Session

        trader = _make_trader(
            force_flat_on_session_close=True,
            force_flat_preclose_seconds=15,
        )
        trader._session_mgr = VN30Session()
        trader._tracker.record_open(
            fill_price=1300.0,
            qty=1,
            side="LONG",
            timestamp=datetime(2025, 1, 2, 14, 20, 0),
        )
        trader._order_mgr.submit_exit = MagicMock()

        trader._on_new_bar(_make_bar(dt=datetime(2025, 1, 2, 14, 31, 0), close=1302.0))

        trader._order_mgr.submit_exit.assert_called_once()
        kwargs = trader._order_mgr.submit_exit.call_args.kwargs
        assert kwargs["reason"] == "ATC Safety Close"
        assert kwargs["ord_type"] == "MARKET"
