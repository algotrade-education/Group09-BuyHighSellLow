"""Unit tests for OrderManager (order submission and lifecycle management).

Tests verify order management functionality and property-based invariants:
- submit_exit() idempotence (Property 14)
- Execution report cleanup after order completion (Property 15)
- Order submission and state tracking
- Duplicate order prevention
- Order lifecycle management

Property-based tests ensure critical invariants:
- Property 14: Multiple exit submissions don't create duplicate orders
- Property 15: Execution reports are cleaned up after order completion

Test organization:
- Property tests: Invariant verification using Hypothesis
- Unit tests: Individual method behavior
- Integration tests: Order lifecycle workflows
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from src.paper.account.tracker import Tracker
from src.paper.execution.order_manager import OrderManager
from src.strategy.signal import Signal, TradeSignal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tracker(initial_capital: float = 100_000.0) -> Tracker:
    return Tracker(
        initial_capital=initial_capital,
        commission_rate=0.0,  # zero commission for simpler tests
        contract_multiplier=100_000.0,
    )


def make_order_manager(
    tracker: Tracker | None = None,
    dry_run: bool = True,
    client=None,
) -> OrderManager:
    if tracker is None:
        tracker = make_tracker()
    return OrderManager(
        client=client,
        tracker=tracker,
        symbol="VN30F2501",
        dry_run=dry_run,
    )


def open_position(
    tracker: Tracker, price: float = 1300.0, qty: int = 1, side: str = "LONG"
) -> None:
    """Helper to open a position on the tracker."""
    tracker.record_open(
        fill_price=price,
        qty=qty,
        side=side,
        timestamp=datetime(2024, 1, 15, 9, 0, 0),
        stop_loss=None,
        take_profit=None,
    )


def dt(hour: int = 9, minute: int = 0) -> datetime:
    return datetime(2024, 1, 15, hour, minute, 0)


# ---------------------------------------------------------------------------
# Property 14: OrderManager Exit Idempotence
# Feature: paper-trade-v2, Property 14: OrderManager Exit Idempotence
# ---------------------------------------------------------------------------


@given(
    reason1=st.text(
        min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
    ),
    reason2=st.text(
        min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))
    ),
    price=st.one_of(
        st.none(),
        st.floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=100)
def test_order_manager_exit_idempotence(reason1: str, reason2: str, price: float | None) -> None:
    """
    # Feature: paper-trade-v2, Property 14: OrderManager Exit Idempotence
    For any sequence of two consecutive submit_exit calls while the same position is open,
    _pending_exits must contain at most one entry after both calls complete.
    The second call must be a no-op.

    Note: In dry_run mode, submit_exit immediately fills (no pending_exits entry).
    We test with a real FIX client mock to verify the idempotence guard.
    """
    tracker = make_tracker()
    # Use a mock client (not dry_run) so pending_exits is populated
    client = MagicMock()
    client.place_order.return_value = "X-MOCK001"
    om = make_order_manager(tracker=tracker, dry_run=False, client=client)

    # Open a position
    open_position(tracker)

    # First submit_exit
    om.submit_exit(reason=reason1, price=price, ord_type="MARKET", timestamp=dt())
    count_after_first = len(om._pending_exits)

    # Second submit_exit - must be a no-op
    om.submit_exit(reason=reason2, price=price, ord_type="MARKET", timestamp=dt())
    count_after_second = len(om._pending_exits)

    assert count_after_second <= 1, (
        f"Expected at most 1 pending exit, got {count_after_second} after two submit_exit calls"
    )
    # Count should not increase on second call
    assert count_after_second == count_after_first, (
        f"Second submit_exit added entries: before={count_after_first}, after={count_after_second}"
    )


@given(
    n_calls=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=100)
def test_order_manager_exit_idempotence_many_calls(n_calls: int) -> None:
    """
    # Feature: paper-trade-v2, Property 14: OrderManager Exit Idempotence
    No matter how many submit_exit calls are made, _pending_exits has at most 1 entry.
    """
    tracker = make_tracker()
    client = MagicMock()
    client.place_order.return_value = "X-MOCK001"
    om = make_order_manager(tracker=tracker, dry_run=False, client=client)

    open_position(tracker)

    for i in range(n_calls):
        om.submit_exit(reason=f"reason_{i}", price=1300.0, ord_type="MARKET", timestamp=dt())

    assert len(om._pending_exits) <= 1


# ---------------------------------------------------------------------------
# Property 15: OrderManager Execution Report Cleanup
# Feature: paper-trade-v2, Property 15: OrderManager Execution Report Cleanup
# ---------------------------------------------------------------------------


@given(
    cl_ord_id=st.text(
        min_size=5,
        max_size=20,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-"),
    ),
    avg_px=st.floats(min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    cum_qty=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_order_manager_execution_report_filled_removes_cum_fills(
    cl_ord_id: str, avg_px: float, cum_qty: int
) -> None:
    """
    # Feature: paper-trade-v2, Property 15: OrderManager Execution Report Cleanup
    For any execution report with ord_status == "2" (FILLED),
    the corresponding cl_ord_id must be removed from _cum_fills.
    """
    tracker = make_tracker()
    om = make_order_manager(tracker=tracker)

    # Manually inject a pending entry
    om._cum_fills[cl_ord_id] = 0
    om._pending_entries[cl_ord_id] = {
        "side": "LONG",
        "qty": cum_qty,
        "stop_loss": None,
        "take_profit": None,
    }

    # Process FILLED execution report
    om.on_execution_report(
        cl_ord_id=cl_ord_id,
        ord_status="2",
        avg_px=avg_px,
        cum_qty=cum_qty,
    )

    assert cl_ord_id not in om._cum_fills, (
        f"cl_ord_id={cl_ord_id} should be removed from _cum_fills after FILLED status"
    )


@given(
    cl_ord_id=st.text(
        min_size=5,
        max_size=20,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-"),
    ),
    ord_status=st.sampled_from(["4", "8"]),
)
@settings(max_examples=100)
def test_order_manager_execution_report_canceled_rejected_removes_pending_exits(
    cl_ord_id: str, ord_status: str
) -> None:
    """
    # Feature: paper-trade-v2, Property 15: OrderManager Execution Report Cleanup
    For ord_status == "4" (CANCELED) or "8" (REJECTED) on an exit order,
    the entry must be removed from _pending_exits.
    """
    tracker = make_tracker()
    om = make_order_manager(tracker=tracker)

    # Manually inject a pending exit
    om._pending_exits[cl_ord_id] = "Stop Loss"

    # Process CANCELED or REJECTED execution report
    om.on_execution_report(
        cl_ord_id=cl_ord_id,
        ord_status=ord_status,
        avg_px=0.0,
        cum_qty=0,
    )

    assert cl_ord_id not in om._pending_exits, (
        f"cl_ord_id={cl_ord_id} should be removed from _pending_exits after status={ord_status}"
    )


# ---------------------------------------------------------------------------
# Unit tests for OrderManager (task 6.6)
# ---------------------------------------------------------------------------


class TestSubmitExitMarketOrder:
    def test_submit_exit_no_price_no_sl_tp_uses_market(self):
        """submit_exit with no price and no SL/TP must submit MARKET order (Req 8.3)."""
        tracker = make_tracker()
        client = MagicMock()
        client.place_order.return_value = "X-MOCK001"
        om = make_order_manager(tracker=tracker, dry_run=False, client=client)

        open_position(tracker)

        om.submit_exit(reason="Force Flat", price=None, ord_type="LIMIT", timestamp=dt())

        # Verify place_order was called with MARKET ord_type
        call_kwargs = client.place_order.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("ord_type") == "MARKET" or (
            len(call_kwargs.args) >= 6 and call_kwargs.args[5] == "MARKET"
        )

    def test_submit_exit_with_price_uses_provided_price(self):
        """submit_exit with a valid price should use that price."""
        tracker = make_tracker()
        client = MagicMock()
        client.place_order.return_value = "X-MOCK001"
        om = make_order_manager(tracker=tracker, dry_run=False, client=client)

        open_position(tracker)

        om.submit_exit(reason="Take Profit", price=1350.0, ord_type="LIMIT", timestamp=dt())

        call_kwargs = client.place_order.call_args
        assert call_kwargs is not None
        # Price should be 1350.0
        price_arg = call_kwargs.kwargs.get("price") or call_kwargs.args[3]
        assert price_arg == 1350.0

    def test_submit_exit_idempotence_guard_dry_run(self):
        """In dry_run mode, submit_exit immediately fills - second call on flat position is no-op."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker, dry_run=True)

        open_position(tracker)

        # First call fills immediately in dry_run
        om.submit_exit(reason="Stop Loss", price=1290.0, ord_type="LIMIT", timestamp=dt())

        # Position is now flat - second call should be a no-op
        result = om.submit_exit(
            reason="Take Profit", price=1350.0, ord_type="LIMIT", timestamp=dt()
        )
        assert result is None

    def test_submit_exit_no_position_is_noop(self):
        """submit_exit when flat should be a no-op."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker, dry_run=True)

        result = om.submit_exit(
            reason="Force Flat", price=1300.0, ord_type="MARKET", timestamp=dt()
        )
        assert result is None


class TestSyncOpenOrders:
    def test_sync_open_orders_classifies_exit_for_long_position(self):
        """SELL orders should be classified as exits when position is LONG (Req 7.3)."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        # Open a LONG position
        open_position(tracker, side="LONG")

        orders = [
            {"clOrdId": "X-001", "side": "SELL", "orderQty": 1, "cumQty": 0, "ordStatus": "0"},
        ]
        om.sync_open_orders(orders)

        assert "X-001" in om._pending_exits

    def test_sync_open_orders_classifies_exit_for_short_position(self):
        """BUY orders should be classified as exits when position is SHORT (Req 7.3)."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        # Open a SHORT position
        open_position(tracker, side="SHORT")

        orders = [
            {"clOrdId": "X-002", "side": "BUY", "orderQty": 1, "cumQty": 0, "ordStatus": "0"},
        ]
        om.sync_open_orders(orders)

        assert "X-002" in om._pending_exits

    def test_sync_open_orders_classifies_entry_when_flat(self):
        """When no position is open, all orders should be classified as entries."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        orders = [
            {"clOrdId": "E-001", "side": "BUY", "orderQty": 2, "cumQty": 0, "ordStatus": "0"},
        ]
        om.sync_open_orders(orders)

        assert "E-001" in om._cum_fills
        assert "E-001" not in om._pending_exits

    def test_sync_open_orders_skips_filled_orders(self):
        """Orders with ordStatus "2" (FILLED) should be skipped."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        orders = [
            {"clOrdId": "E-001", "side": "BUY", "orderQty": 1, "cumQty": 1, "ordStatus": "2"},
        ]
        om.sync_open_orders(orders)

        assert "E-001" not in om._cum_fills
        assert "E-001" not in om._pending_exits

    def test_sync_open_orders_clears_previous_state(self):
        """sync_open_orders must clear previous state before populating."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        # Pre-populate state
        om._pending_exits["OLD-EXIT"] = "old reason"
        om._cum_fills["OLD-ENTRY"] = 0

        om.sync_open_orders([])

        assert "OLD-EXIT" not in om._pending_exits
        assert "OLD-ENTRY" not in om._cum_fills

    def test_sync_open_orders_numeric_side_codes(self):
        """FIX numeric side codes '1' (BUY) and '2' (SELL) should be normalized."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        # Open a LONG position
        open_position(tracker, side="LONG")

        orders = [
            {"clOrdId": "X-003", "side": "2", "orderQty": 1, "cumQty": 0, "ordStatus": "0"},
        ]
        om.sync_open_orders(orders)

        # "2" = SELL = exit for LONG position
        assert "X-003" in om._pending_exits


class TestOnExecutionReport:
    def test_filled_entry_removes_from_cum_fills(self):
        """FILLED entry removes cl_ord_id from _cum_fills (Req 8.4)."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        cl_ord_id = "E-FILL001"
        om._cum_fills[cl_ord_id] = 0
        om._pending_entries[cl_ord_id] = {
            "side": "LONG",
            "qty": 1,
            "stop_loss": None,
            "take_profit": None,
        }

        om.on_execution_report(cl_ord_id=cl_ord_id, ord_status="2", avg_px=1300.0, cum_qty=1)

        assert cl_ord_id not in om._cum_fills

    def test_canceled_exit_removes_from_pending_exits(self):
        """CANCELED exit removes cl_ord_id from _pending_exits (Req 8.5)."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        cl_ord_id = "X-CANCEL001"
        om._pending_exits[cl_ord_id] = "Stop Loss"

        om.on_execution_report(cl_ord_id=cl_ord_id, ord_status="4", avg_px=0.0, cum_qty=0)

        assert cl_ord_id not in om._pending_exits

    def test_rejected_exit_removes_from_pending_exits(self):
        """REJECTED exit removes cl_ord_id from _pending_exits (Req 8.5)."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        cl_ord_id = "X-REJECT001"
        om._pending_exits[cl_ord_id] = "Take Profit"

        om.on_execution_report(cl_ord_id=cl_ord_id, ord_status="8", avg_px=0.0, cum_qty=0)

        assert cl_ord_id not in om._pending_exits

    def test_has_pending_exit_property(self):
        """has_pending_exit returns True when _pending_exits is non-empty."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        assert not om.has_pending_exit

        om._pending_exits["X-001"] = "Stop Loss"
        assert om.has_pending_exit

        del om._pending_exits["X-001"]
        assert not om.has_pending_exit

    def test_partial_exit_does_not_flatten_position_until_filled(self):
        """PARTIAL exit report reduces qty but must not flatten before final fill."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        open_position(tracker, price=1300.0, qty=2, side="LONG")

        cl_ord_id = "X-PART001"
        om._pending_exits[cl_ord_id] = "Stop Loss"
        om._cum_fills[cl_ord_id] = 0

        om.on_execution_report(
            cl_ord_id=cl_ord_id,
            ord_status="1",
            avg_px=1298.0,
            cum_qty=1,
        )

        assert not tracker.is_flat
        assert tracker.position.quantity == 1
        assert cl_ord_id in om._pending_exits

        om.on_execution_report(
            cl_ord_id=cl_ord_id,
            ord_status="2",
            avg_px=1297.0,
            cum_qty=2,
        )

        assert tracker.is_flat
        assert cl_ord_id not in om._pending_exits

    def test_execution_report_prefers_last_price_over_avg_price(self):
        """When last_px exists, it should be used as fill price instead of avg_px."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        cl_ord_id = "E-LASTPX1"
        om._cum_fills[cl_ord_id] = 0
        om._pending_entries[cl_ord_id] = {
            "side": "LONG",
            "qty": 1,
            "stop_loss": None,
            "take_profit": None,
        }

        om.on_execution_report(
            cl_ord_id=cl_ord_id,
            ord_status="2",
            avg_px=1310.0,
            last_px=1302.0,
            cum_qty=1,
        )

        assert not tracker.is_flat
        assert tracker.position.entry_price == 1302.0

    def test_execution_report_accepts_camel_case_fields(self):
        """Broker camelCase payload should be normalized and processed."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        cl_ord_id = "E-CAMEL01"
        om._cum_fills[cl_ord_id] = 0
        om._pending_entries[cl_ord_id] = {
            "side": "LONG",
            "qty": 1,
            "stop_loss": None,
            "take_profit": None,
        }

        om.on_execution_report(
            clOrdId=cl_ord_id,
            ordStatus="2",
            avgPx=1310.0,
            cumQty=1,
            lastPx=1304.0,
        )

        assert not tracker.is_flat
        assert tracker.position.entry_price == 1304.0

    def test_execution_report_accepts_textual_filled_status(self):
        """Text status values (e.g., FILLED) should map to FIX status codes."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker)

        cl_ord_id = "E-TEXT01"
        om._cum_fills[cl_ord_id] = 0
        om._pending_entries[cl_ord_id] = {
            "side": "LONG",
            "qty": 1,
            "stop_loss": None,
            "take_profit": None,
        }

        om.on_execution_report(
            cl_ord_id=cl_ord_id,
            status="FILLED",
            avg_px=1312.0,
            cum_qty=1,
            last_px=1308.0,
        )

        assert not tracker.is_flat
        assert tracker.position.entry_price == 1308.0


# ---------------------------------------------------------------------------
# submit_entry - dry-run fills immediately
# ---------------------------------------------------------------------------


class TestSubmitEntryDryRun:
    def _make_long_signal(self, entry_price: float = 1300.0) -> TradeSignal:
        return TradeSignal(
            signal=Signal.LONG,
            entry_price=entry_price,
            stop_loss=1280.0,
            take_profit=1350.0,
            ord_type="MARKET",
        )

    def _make_short_signal(self, entry_price: float = 1300.0) -> TradeSignal:
        return TradeSignal(
            signal=Signal.SHORT,
            entry_price=entry_price,
            stop_loss=1320.0,
            take_profit=1250.0,
            ord_type="MARKET",
        )

    def test_dry_run_long_creates_position(self):
        """In dry-run, submit_entry immediately populates a LONG position."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker, dry_run=True)

        om.submit_entry(
            signal=self._make_long_signal(1300.0),
            qty=2,
            bar={"close": 1300.0},
            timestamp=dt(),
        )

        assert not tracker.is_flat
        assert tracker.position.is_long
        assert tracker.position.quantity == 2

    def test_dry_run_short_creates_short_position(self):
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker, dry_run=True)

        om.submit_entry(
            signal=self._make_short_signal(1300.0),
            qty=1,
            bar={"close": 1300.0},
            timestamp=dt(),
        )

        assert not tracker.is_flat
        assert tracker.position.is_short

    def test_dry_run_returns_none(self):
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker, dry_run=True)
        result = om.submit_entry(
            signal=self._make_long_signal(),
            qty=1,
            bar={"close": 1300.0},
            timestamp=dt(),
        )
        assert result is None

    def test_non_entry_signal_is_ignored(self):
        """submit_entry with non-entry signal must be a no-op."""
        tracker = make_tracker()
        om = make_order_manager(tracker=tracker, dry_run=True)
        # EXIT signal
        non_entry = TradeSignal(
            signal=Signal.EXIT,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            ord_type="MARKET",
        )
        result = om.submit_entry(signal=non_entry, qty=1, bar=None, timestamp=dt())
        assert result is None
        assert tracker.is_flat


# ---------------------------------------------------------------------------
# Entry fill end-to-end: submit_entry (FIX) → on_execution_report FILLED
# ---------------------------------------------------------------------------


class TestEntryFillEndToEnd:
    def test_filled_execution_report_opens_position(self):
        """submit_entry (FIX mode) followed by on_execution_report FILLED opens a position."""
        tracker = make_tracker()
        client = MagicMock()
        client.place_order.return_value = "E-MOCK001"
        om = make_order_manager(tracker=tracker, dry_run=False, client=client)

        signal = TradeSignal(
            signal=Signal.LONG,
            entry_price=1300.0,
            stop_loss=1280.0,
            take_profit=1350.0,
            ord_type="MARKET",
        )
        cl_ord_id = om.submit_entry(signal=signal, qty=2, bar=None, timestamp=dt())

        assert cl_ord_id == "E-MOCK001"
        assert cl_ord_id in om._cum_fills
        assert cl_ord_id in om._pending_entries

        # Simulate FILLED execution report
        om.on_execution_report(
            cl_ord_id=cl_ord_id,
            ord_status="2",
            avg_px=1302.0,
            cum_qty=2,
        )

        # Position should now be open
        assert not tracker.is_flat
        assert tracker.position.quantity == 2
        # cl_ord_id removed from tracking state after full fill
        assert cl_ord_id not in om._cum_fills
        assert cl_ord_id not in om._pending_entries

    def test_partial_fill_accumulates_qty(self):
        """Partial fills (ord_status=1) should add incremental quantity."""
        tracker = make_tracker()
        client = MagicMock()
        client.place_order.return_value = "E-MOCK002"
        om = make_order_manager(tracker=tracker, dry_run=False, client=client)

        signal = TradeSignal(
            signal=Signal.LONG,
            entry_price=1300.0,
            stop_loss=0.0,
            take_profit=0.0,
            ord_type="MARKET",
        )
        om.submit_entry(signal=signal, qty=3, bar=None, timestamp=dt())

        # First partial fill: 1 of 3
        om.on_execution_report(cl_ord_id="E-MOCK002", ord_status="1", avg_px=1300.0, cum_qty=1)
        assert not tracker.is_flat
        assert tracker.position.quantity == 1

        # Second partial fill: 1 more (cum_qty=2)
        om.on_execution_report(cl_ord_id="E-MOCK002", ord_status="1", avg_px=1301.0, cum_qty=2)
        assert tracker.position.quantity == 2

        # Still pending
        assert "E-MOCK002" in om._cum_fills

    def test_exit_fill_end_to_end_closes_position(self):
        """submit_exit (FIX) followed by on_execution_report FILLED closes the position."""
        tracker = make_tracker()
        # place_order is only called for submit_exit (entry is injected manually)
        client = MagicMock()
        client.place_order.return_value = "X-MOCK003"
        om = make_order_manager(tracker=tracker, dry_run=False, client=client)

        # Manually inject an already-filled entry position
        om._cum_fills["E-MOCK003"] = 0
        om._pending_entries["E-MOCK003"] = {
            "side": "LONG",
            "qty": 1,
            "stop_loss": None,
            "take_profit": None,
        }
        om.on_execution_report(cl_ord_id="E-MOCK003", ord_status="2", avg_px=1300.0, cum_qty=1)
        assert not tracker.is_flat

        # Submit exit - place_order returns "X-MOCK003"
        om.submit_exit(reason="Take Profit", price=1350.0, ord_type="LIMIT", timestamp=dt())
        assert "X-MOCK003" in om._pending_exits

        # Fill the exit
        om.on_execution_report(cl_ord_id="X-MOCK003", ord_status="2", avg_px=1350.0, cum_qty=1)

        assert tracker.is_flat
        assert "X-MOCK003" not in om._pending_exits
