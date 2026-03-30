"""Unit tests for Account Reconciler.

Tests cover:
- Error isolation: any exception from the broker API is caught and sync() returns normally (hypothesis)
- reconcile_position: syncs qty/price from broker portfolio response
- reconcile_cash: syncs cash balance from broker
- reconcile_orders: classifies open orders as entries or exits
- sync(): calls all three reconcile methods; failures in one do not block the others
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from src.paper.account.reconciler import Reconciler
from src.paper.account.tracker import Tracker
from src.paper.execution.order_manager import OrderManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYMBOL = "VN30F2501"


def make_tracker(commission_rate: float = 0.0003) -> Tracker:
    return Tracker(
        initial_capital=100_000.0,
        commission_rate=commission_rate,
        contract_multiplier=100_000.0,
    )


def make_order_manager(tracker: Tracker) -> OrderManager:
    return OrderManager(client=None, tracker=tracker, symbol=SYMBOL, dry_run=True)


def make_reconciler(
    client: MagicMock | None = None,
    tracker: Tracker | None = None,
    order_manager: OrderManager | None = None,
    symbol: str = SYMBOL,
) -> tuple[Reconciler, Tracker, OrderManager]:
    if tracker is None:
        tracker = make_tracker()
    if order_manager is None:
        order_manager = make_order_manager(tracker)
    if client is None:
        client = MagicMock()
    rec = Reconciler(client=client, tracker=tracker, order_manager=order_manager, symbol=symbol)
    return rec, tracker, order_manager


def portfolio_response(symbol: str, qty: float, avg_price: float) -> dict:
    return {
        "success": True,
        "items": [{"instrument": symbol, "quantity": qty, "avgPrice": avg_price}],
    }


def cash_response(remain_cash: float) -> dict:
    return {"remainCash": remain_cash}


def orders_response(orders: list) -> dict:
    return {"success": True, "items": orders}


# ---------------------------------------------------------------------------
# Property: Reconciler Error Isolation
# ---------------------------------------------------------------------------

exception_strategy = st.sampled_from(
    [
        ConnectionError("Connection refused"),
        TimeoutError("Request timed out"),
        ValueError("Invalid response"),
        RuntimeError("Unexpected error"),
        KeyError("missing_key"),
        AttributeError("NoneType has no attribute"),
        Exception("Generic error"),
    ]
)


@given(exc=exception_strategy)
@settings(max_examples=100)
def test_reconciler_error_isolation_portfolio(exc: Exception) -> None:
    """Any exception during portfolio fetch must be caught; sync() must return normally."""
    client = MagicMock()
    client.get_portfolio_by_sub.side_effect = exc
    rec, _, _ = make_reconciler(client=client)
    rec.sync()  # must not raise


@given(exc=exception_strategy)
@settings(max_examples=100)
def test_reconciler_error_isolation_cash(exc: Exception) -> None:
    """Exception during cash balance fetch must be caught and not re-raised."""
    client = MagicMock()
    client.get_portfolio_by_sub.return_value = {"success": False}
    client.get_cash_balance.side_effect = exc
    rec, _, _ = make_reconciler(client=client)
    rec.sync()  # must not raise


@given(exc=exception_strategy)
@settings(max_examples=100)
def test_reconciler_error_isolation_orders(exc: Exception) -> None:
    """Exception during open orders fetch must be caught and not re-raised."""
    client = MagicMock()
    client.get_portfolio_by_sub.return_value = {"success": False}
    client.get_cash_balance.return_value = {}
    client.get_orders.side_effect = exc
    rec, _, _ = make_reconciler(client=client)
    rec.sync()  # must not raise


# ---------------------------------------------------------------------------
# reconcile_position
# ---------------------------------------------------------------------------


class TestReconcilePosition:
    def test_syncs_position_when_symbol_found(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = portfolio_response(
            SYMBOL, qty=2.0, avg_price=1300.0
        )
        rec, tracker, _ = make_reconciler(client=client)

        rec.reconcile_position()

        assert not tracker.is_flat
        assert tracker.position.quantity == 2
        assert tracker.synced_position

    def test_no_sync_when_symbol_not_in_portfolio(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {
            "success": True,
            "items": [{"instrument": "OTHER", "quantity": 1.0, "avgPrice": 1300.0}],
        }
        rec, tracker, _ = make_reconciler(client=client)

        rec.reconcile_position()

        assert tracker.is_flat

    def test_no_sync_when_response_unsuccessful(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": False, "error": "Unauthorized"}
        rec, tracker, _ = make_reconciler(client=client)

        rec.reconcile_position()

        assert tracker.is_flat

    def test_logs_mismatch_when_broker_qty_differs(self, caplog):
        """Should log a warning when broker qty != tracker qty."""
        import logging

        client = MagicMock()
        client.get_portfolio_by_sub.return_value = portfolio_response(
            SYMBOL, qty=3.0, avg_price=1300.0
        )
        rec, tracker, _ = make_reconciler(client=client)
        # tracker is flat (qty=0), broker says qty=3 → mismatch

        with caplog.at_level(logging.WARNING, logger="src.paper.account.reconciler"):
            rec.reconcile_position()

        assert any("mismatch" in r.message.lower() for r in caplog.records)

    def test_exception_is_caught_and_not_reraised(self):
        client = MagicMock()
        client.get_portfolio_by_sub.side_effect = ConnectionError("broker down")
        rec, _, _ = make_reconciler(client=client)

        rec.reconcile_position()  # must not raise

    def test_empty_portfolio_syncs_zero_position(self):
        """Empty portfolio items → no position synced, tracker stays flat."""
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": True, "items": []}
        rec, tracker, _ = make_reconciler(client=client)

        rec.reconcile_position()

        assert tracker.is_flat


# ---------------------------------------------------------------------------
# reconcile_cash
# ---------------------------------------------------------------------------


class TestReconcileCash:
    def test_syncs_cash_from_remain_cash(self):
        client = MagicMock()
        client.get_cash_balance.return_value = cash_response(75_000.0)
        rec, tracker, _ = make_reconciler(client=client)

        rec.reconcile_cash()

        assert tracker.cash == 75_000.0

    def test_no_sync_when_remain_cash_missing(self):
        client = MagicMock()
        client.get_cash_balance.return_value = {}
        rec, tracker, _ = make_reconciler(client=client)
        initial_cash = tracker.cash

        rec.reconcile_cash()

        assert tracker.cash == initial_cash

    def test_no_sync_when_response_is_none(self):
        client = MagicMock()
        client.get_cash_balance.return_value = None
        rec, tracker, _ = make_reconciler(client=client)
        initial_cash = tracker.cash

        rec.reconcile_cash()

        assert tracker.cash == initial_cash

    def test_exception_is_caught_and_not_reraised(self):
        client = MagicMock()
        client.get_cash_balance.side_effect = TimeoutError("timeout")
        rec, _, _ = make_reconciler(client=client)

        rec.reconcile_cash()  # must not raise

    def test_multiple_calls_use_latest_value(self):
        client = MagicMock()
        client.get_cash_balance.side_effect = [
            cash_response(80_000.0),
            cash_response(70_000.0),
        ]
        rec, tracker, _ = make_reconciler(client=client)

        rec.reconcile_cash()
        rec.reconcile_cash()

        assert tracker.cash == 70_000.0


# ---------------------------------------------------------------------------
# reconcile_orders
# ---------------------------------------------------------------------------


class TestReconcileOrders:
    def test_syncs_open_orders(self):
        client = MagicMock()
        client.get_orders.return_value = orders_response(
            [
                {"clOrdId": "E-001", "side": "BUY", "orderQty": 2, "cumQty": 0, "ordStatus": "0"},
            ]
        )
        rec, _, order_manager = make_reconciler(client=client)

        rec.reconcile_orders()

        assert "E-001" in order_manager._cum_fills

    def test_no_sync_when_response_unsuccessful(self):
        client = MagicMock()
        client.get_orders.return_value = {"success": False, "error": "Not found"}
        rec, _, order_manager = make_reconciler(client=client)

        rec.reconcile_orders()

        assert len(order_manager._cum_fills) == 0

    def test_exception_is_caught_and_not_reraised(self):
        client = MagicMock()
        client.get_orders.side_effect = RuntimeError("API error")
        rec, _, _ = make_reconciler(client=client)

        rec.reconcile_orders()  # must not raise

    def test_classifies_exit_orders_for_long_position(self):
        """SELL orders should be exits when tracker has a LONG position."""
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = portfolio_response(
            SYMBOL, qty=1.0, avg_price=1300.0
        )
        client.get_orders.return_value = orders_response(
            [
                {"clOrdId": "X-001", "side": "SELL", "orderQty": 1, "cumQty": 0, "ordStatus": "0"},
            ]
        )
        rec, tracker, order_manager = make_reconciler(client=client)

        # First sync position so tracker knows it's LONG
        rec.reconcile_position()
        rec.reconcile_orders()

        assert "X-001" in order_manager._pending_exits


# ---------------------------------------------------------------------------
# sync() - full pipeline
# ---------------------------------------------------------------------------


class TestSyncCallsAllThree:
    def test_sync_calls_all_three_reconcile_methods(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": True, "items": []}
        client.get_cash_balance.return_value = cash_response(50_000.0)
        client.get_orders.return_value = orders_response([])

        rec, _, _ = make_reconciler(client=client)
        rec.sync()

        client.get_portfolio_by_sub.assert_called_once()
        client.get_cash_balance.assert_called_once()
        client.get_orders.assert_called_once()

    def test_sync_continues_when_position_fails(self):
        """If reconcile_position raises, cash and orders should still be synced."""
        client = MagicMock()
        client.get_portfolio_by_sub.side_effect = ConnectionError("down")
        client.get_cash_balance.return_value = cash_response(60_000.0)
        client.get_orders.return_value = orders_response([])

        rec, tracker, _ = make_reconciler(client=client)
        rec.sync()

        assert tracker.cash == 60_000.0

    def test_sync_continues_when_cash_fails(self):
        """If reconcile_cash raises, orders should still be synced."""
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": True, "items": []}
        client.get_cash_balance.side_effect = TimeoutError("timeout")
        client.get_orders.return_value = orders_response([])

        rec, _, _ = make_reconciler(client=client)
        rec.sync()

        client.get_orders.assert_called_once()

    def test_sync_continues_when_orders_fails(self):
        """If reconcile_orders raises, sync() still returns normally."""
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": True, "items": []}
        client.get_cash_balance.return_value = cash_response(50_000.0)
        client.get_orders.side_effect = RuntimeError("orders API down")

        rec, tracker, _ = make_reconciler(client=client)
        rec.sync()  # must not raise

        assert tracker.cash == 50_000.0


# ---------------------------------------------------------------------------
# sync() - unit tests mirroring old test_reconciler_5_2_5_3 sync() tests
# ---------------------------------------------------------------------------


class TestSyncPosition:
    def test_calls_sync_position_with_correct_args(self):
        """sync must call tracker.sync_position with qty and avg_price."""
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = portfolio_response(
            SYMBOL, qty=2.0, avg_price=1300.0
        )
        client.get_cash_balance.return_value = cash_response(50_000.0)
        client.get_orders.return_value = orders_response([])

        tracker = make_tracker()
        order_manager = make_order_manager(tracker)
        rec, _, _ = make_reconciler(client=client, tracker=tracker, order_manager=order_manager)

        rec.sync()

        assert not tracker.is_flat
        assert tracker.position.quantity == 2
        assert tracker.synced_position

    def test_skips_sync_position_when_symbol_not_in_portfolio(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {
            "success": True,
            "items": [{"instrument": "OTHER_SYMBOL", "quantity": 1.0, "avgPrice": 1300.0}],
        }
        client.get_cash_balance.return_value = cash_response(50_000.0)
        client.get_orders.return_value = orders_response([])

        tracker = make_tracker()
        rec, _, _ = make_reconciler(client=client, tracker=tracker)

        rec.sync()

        assert tracker.is_flat

    def test_skips_sync_position_when_portfolio_unsuccessful(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": False, "error": "Unauthorized"}
        client.get_cash_balance.return_value = cash_response(50_000.0)
        client.get_orders.return_value = orders_response([])

        tracker = make_tracker()
        rec, _, _ = make_reconciler(client=client, tracker=tracker)

        rec.sync()

        assert tracker.is_flat


class TestSyncCash:
    def test_calls_sync_cash_with_remain_cash(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": True, "items": []}
        client.get_cash_balance.return_value = cash_response(75_000.0)
        client.get_orders.return_value = orders_response([])

        tracker = make_tracker()
        rec, _, _ = make_reconciler(client=client, tracker=tracker)

        rec.sync()

        assert tracker.cash == 75_000.0

    def test_skips_sync_cash_when_remain_cash_missing(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": True, "items": []}
        client.get_cash_balance.return_value = {}  # no remainCash
        client.get_orders.return_value = orders_response([])

        tracker = make_tracker()
        initial_cash = tracker.cash
        rec, _, _ = make_reconciler(client=client, tracker=tracker)

        rec.sync()

        assert tracker.cash == initial_cash


class TestSyncOpenOrders:
    def test_calls_sync_open_orders_with_order_list(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": True, "items": []}
        client.get_cash_balance.return_value = cash_response(50_000.0)
        client.get_orders.return_value = orders_response(
            [
                {"clOrdId": "E-001", "side": "BUY", "orderQty": 2, "cumQty": 0, "ordStatus": "0"},
            ]
        )

        tracker = make_tracker()
        order_manager = make_order_manager(tracker)
        rec, _, _ = make_reconciler(client=client, tracker=tracker, order_manager=order_manager)

        rec.sync()

        assert "E-001" in order_manager._cum_fills

    def test_skips_sync_open_orders_when_unsuccessful(self):
        client = MagicMock()
        client.get_portfolio_by_sub.return_value = {"success": True, "items": []}
        client.get_cash_balance.return_value = cash_response(50_000.0)
        client.get_orders.return_value = {"success": False, "error": "Not found"}

        tracker = make_tracker()
        order_manager = make_order_manager(tracker)
        rec, _, _ = make_reconciler(client=client, tracker=tracker, order_manager=order_manager)

        rec.sync()

        assert len(order_manager._cum_fills) == 0


class TestSyncErrorIsolation:
    def test_exception_does_not_propagate(self):
        """Any exception from broker API must be caught and not re-raised."""
        client = MagicMock()
        client.get_portfolio_by_sub.side_effect = RuntimeError("Broker down")

        rec, _, _ = make_reconciler(client=client)

        rec.sync()  # must not raise

    def test_engine_continues_after_sync_error(self):
        """After a sync error, the tracker should remain in its initial state."""
        client = MagicMock()
        client.get_portfolio_by_sub.side_effect = ConnectionError("Network error")

        tracker = make_tracker()
        initial_cash = tracker.cash
        order_manager = make_order_manager(tracker)
        rec, _, _ = make_reconciler(client=client, tracker=tracker, order_manager=order_manager)

        rec.sync()

        assert tracker.is_flat
        assert tracker.cash == initial_cash
        assert not tracker.synced_position
