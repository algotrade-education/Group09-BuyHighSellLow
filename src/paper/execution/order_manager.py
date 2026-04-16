"""
Order lifecycle management for paper trading.

Manages FIX order submission, execution report processing, and order state tracking.
Handles both entry and exit orders with proper idempotence guarantees and state cleanup.

Key responsibilities:
- Submit entry orders (LONG/SHORT) with optional stop-loss and take-profit levels
- Submit exit orders with idempotence protection (prevents duplicate exit submissions)
- Process FIX execution reports for fills, cancellations, and rejections
- Maintain order state tracking (_pending_exits, _cum_fills, _pending_entries)
- Sync open orders from broker during reconciliation
- Generate unique client order IDs for all submissions

Order state tracking:
- _pending_exits: Maps cl_ord_id → exit reason for pending exit orders
- _cum_fills: Maps cl_ord_id → cumulative filled quantity for tracking partial fills
- _pending_entries: Maps cl_ord_id → entry metadata (side, qty, SL, TP)

Exit order semantics:
- Idempotent: Only one exit order can be pending at a time
- MARKET orders: Submitted when no price and no SL/TP are available
- LIMIT orders: Use provided price or fall back to position entry price
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from config.constants import MARKET_ORDER_PRICE_BUFFER
from src.strategy.signal import TradeSignal

if TYPE_CHECKING:
    from paperbroker.client import PaperBrokerClient

    from src.paper.account.tracker import Tracker

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages FIX order submission and state tracking for paper trading.

    Tracks three types of order state:
    - _pending_exits: Exit orders awaiting fill/cancel/reject
    - _cum_fills: Cumulative filled quantity for tracking partial fills
    - _pending_entries: Entry order metadata (side, qty, SL, TP)

    Provides idempotent exit submission (only one exit can be pending at a time)
    and proper state cleanup on order completion or cancellation.
    """

    def __init__(
        self,
        client: PaperBrokerClient | None,
        tracker: Tracker,
        symbol: str,
        dry_run: bool,
    ) -> None:
        """Initialize the order manager.

        Args:
            client: Paper broker client for FIX order submission (None in dry-run mode).
            tracker: Account tracker for recording fills and position updates.
            symbol: Trading symbol (e.g. 'HNXDS:VN30F2601').
            dry_run: If True, orders are logged but not submitted to broker.
        """
        self._client = client
        self._tracker = tracker
        self._symbol = symbol
        self._dry_run = dry_run

        self._pending_exits: dict[str, str] = {}
        self._cum_fills: dict[str, int] = {}
        self._pending_entries: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_pending_exit(self) -> bool:
        """Return True when at least one exit order is pending."""
        return bool(self._pending_exits)

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_entry(
        self,
        signal: TradeSignal,
        qty: int,
        bar: dict | None,
        timestamp: datetime | None,
    ) -> str | None:
        """Submit an entry order (LONG or SHORT).

        Supports both MARKET and LIMIT orders based on signal.ord_type.
        In dry-run mode, simulates an immediate fill without broker submission.
        Handles scale-ins by tracking cumulative fills per order.

        Args:
            signal: Trade signal containing entry direction, type, price, SL, TP.
            qty: Order quantity (number of contracts).
            bar: Current bar dict for fallback pricing (used when no limit price).
            timestamp: Fill timestamp for dry-run simulation.

        Returns:
            Client order ID on success, or None on failure/dry-run.
        """
        if not signal.is_entry:
            logger.warning("submit_entry called with non-entry signal %s - ignoring", signal.signal)
            return None

        side = "BUY" if signal.is_long else "SELL"
        ord_type = signal.ord_type  # "MARKET" or "LIMIT"
        price = signal.entry_price
        stop_loss = signal.stop_loss if signal.stop_loss > 0 else None
        take_profit = signal.take_profit if signal.take_profit > 0 else None

        cl_ord_id = self._gen_cl_ord_id("E")

        logger.info(
            "submit_entry: symbol=%s side=%s qty=%d ord_type=%s price=%s sl=%s tp=%s cl_ord_id=%s",
            self._symbol,
            side,
            qty,
            ord_type,
            price,
            stop_loss,
            take_profit,
            cl_ord_id,
        )

        if self._dry_run or self._client is None:
            logger.info("submit_entry [DRY-RUN]: order logged only, not sent")
            effective_price = (
                price if price and price > 0 else float(bar.get("close", 1800.0) if bar else 1800.0)
            )
            self._tracker.record_open(
                fill_price=effective_price,
                qty=qty,
                side="LONG" if signal.is_long else "SHORT",
                timestamp=timestamp or datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            return None

        try:
            cl_ord_id = self._client.place_order(
                full_symbol=self._symbol,
                side=side,
                qty=qty,
                price=price
                if price and price > 0
                else float(bar.get("close", 1800.0) if bar else 1800.0),
                ord_type=ord_type,
                tif="GTC",
            )
            self._cum_fills[cl_ord_id] = 0
            self._pending_entries[cl_ord_id] = {
                "side": "LONG" if signal.is_long else "SHORT",
                "qty": qty,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
            return str(cl_ord_id)
        except Exception:
            logger.error("submit_entry: FIX order submission failed", exc_info=True)
            return None

    def submit_exit(
        self,
        reason: str,
        price: float | None,
        ord_type: str,
        timestamp: datetime | None,
    ) -> str | None:
        """Submit an exit order to close the current position.

        Idempotence: Only one exit order can be pending at a time. If an exit
        is already pending, this method logs a warning and returns None.

        Order type logic:
        - If price is provided: Uses the specified ord_type with that price
        - If no price and position has SL/TP: Uses ord_type as-is (broker handles levels)
        - If no price and no SL/TP: Submits MARKET order (no price parameter sent)

        Args:
            reason: Exit reason string (e.g. "Stop Loss", "Take Profit", "Signal Exit").
            price: Optional limit price for the exit order.
            ord_type: Order type ("MARKET" or "LIMIT").
            timestamp: Fill timestamp for dry-run simulation.

        Returns:
            Client order ID on success, or None on failure/dry-run/no-op.
        """
        if self._pending_exits:
            logger.warning(
                "submit_exit: pending exit already exists (%s) - skipping duplicate submission (reason=%s)",
                list(self._pending_exits.values()),
                reason,
            )
            return None

        position = self._tracker.position
        if position.is_flat:
            logger.warning(
                "submit_exit: no open position - ignoring exit request (reason=%s)", reason
            )
            return None

        side = "SELL" if position.is_long else "BUY"
        effective_ord_type = ord_type
        effective_price = price if price and price > 0 else None

        if effective_price is None:
            has_sl_tp = position.stop_loss is not None or position.take_profit is not None
            if not has_sl_tp:
                effective_ord_type = "MARKET"
                logger.info(
                    "submit_exit: no price and no SL/TP - submitting MARKET order (reason=%s)",
                    reason,
                )

        cl_ord_id = self._gen_cl_ord_id("X")

        logger.info(
            "submit_exit: symbol=%s side=%s qty=%d ord_type=%s price=%s reason=%s cl_ord_id=%s",
            self._symbol,
            side,
            position.quantity,
            effective_ord_type,
            effective_price,
            reason,
            cl_ord_id,
        )

        if self._dry_run or self._client is None:
            logger.info("submit_exit [DRY-RUN]: order logged only, not sent")
            fill_price = effective_price or position.entry_price
            self._tracker.record_close(
                fill_price=fill_price,
                qty=position.quantity,
                timestamp=timestamp or datetime.now(),
                exit_reason=reason,
            )
            return None

        try:
            if effective_ord_type == "MARKET":
                # Use price buffer for MARKET orders to ensure execution
                # (Paperbroker API requires price parameter even for MARKET orders)
                price = effective_price or position.entry_price
                if position.is_long:
                    price += MARKET_ORDER_PRICE_BUFFER
                else:
                    price -= MARKET_ORDER_PRICE_BUFFER

                cl_ord_id = self._client.place_order(
                    full_symbol=self._symbol,
                    side=side,
                    qty=position.quantity,
                    price=price,
                    ord_type=effective_ord_type,
                    tif="GTC",
                )
            else:
                cl_ord_id = self._client.place_order(
                    full_symbol=self._symbol,
                    side=side,
                    qty=position.quantity,
                    price=effective_price or position.entry_price,
                    ord_type=effective_ord_type,
                    tif="GTC",
                )
            self._pending_exits[cl_ord_id] = reason
            return str(cl_ord_id)
        except Exception:
            logger.error("submit_exit: FIX order submission failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Execution report handler
    # ------------------------------------------------------------------

    def on_execution_report(self, **kwargs: Any) -> None:
        """Process a FIX execution report from the broker.

        Handles order status updates and triggers position updates in the tracker.
        Supports partial fills by tracking cumulative filled quantity per order.

        FIX status codes:
        - "0" (NEW): Order accepted, no fills yet
        - "1" (PARTIALLY_FILLED): Partial fill, update cumulative quantity
        - "2" (FILLED): Order fully filled, clean up tracking state
        - "4" (CANCELED): Order canceled, release locks
        - "8" (REJECTED): Order rejected, release locks

        Args:
            **kwargs: Execution report fields including cl_ord_id, ord_status,
                     avg_px, cum_qty, and optional transact_time/timestamp.
        """
        report = self._normalize_execution_report_fields(kwargs)

        cl_ord_id: str = str(report.get("cl_ord_id", ""))
        ord_status: str = str(report.get("ord_status", ""))
        avg_px: float = float(report.get("avg_px") or 0.0)
        fill_px: float = self._extract_fill_price(report, fallback=avg_px)
        cum_qty: int = int(report.get("cum_qty") or 0)

        logger.debug(
            "on_execution_report: cl_ord_id=%s status=%s cum_qty=%d avg_px=%.2f",
            cl_ord_id,
            ord_status,
            cum_qty,
            avg_px,
        )

        is_exit = cl_ord_id in self._pending_exits
        is_entry = cl_ord_id in self._pending_entries

        if ord_status in ("4", "8"):
            status_name = "CANCELED" if ord_status == "4" else "REJECTED"
            self._cum_fills.pop(cl_ord_id, None)
            if is_entry:
                self._pending_entries.pop(cl_ord_id, None)
                logger.warning("on_execution_report: entry %s cl_ord_id=%s", status_name, cl_ord_id)
            if is_exit:
                self._pending_exits.pop(cl_ord_id, None)
                logger.warning(
                    "on_execution_report: exit %s cl_ord_id=%s - cleared from pending_exits",
                    status_name,
                    cl_ord_id,
                )
            return

        if ord_status not in ("1", "2"):
            logger.debug(
                "on_execution_report: unhandled status=%s cl_ord_id=%s", ord_status, cl_ord_id
            )
            return

        # Calculate incremental fill quantity for partial fill support
        prev_fill = self._cum_fills.get(cl_ord_id, 0)
        incremental_qty = cum_qty - prev_fill
        if incremental_qty <= 0:
            return
        self._cum_fills[cl_ord_id] = cum_qty

        if is_entry:
            meta = self._pending_entries[cl_ord_id]
            logger.info(
                "on_execution_report: entry %s cl_ord_id=%s fill_px=%.2f qty=%d",
                "FILLED" if ord_status == "2" else "PARTIAL",
                cl_ord_id,
                fill_px,
                incremental_qty,
            )
            fill_timestamp = self._extract_broker_timestamp(report)
            self._tracker.record_open(
                fill_price=fill_px,
                qty=incremental_qty,
                side=meta["side"],
                timestamp=fill_timestamp,
                stop_loss=meta.get("stop_loss"),
                take_profit=meta.get("take_profit"),
            )
            if ord_status == "2":
                self._pending_entries.pop(cl_ord_id)
                self._cum_fills.pop(cl_ord_id, None)
            return

        if is_exit:
            reason = self._pending_exits[cl_ord_id]
            logger.info(
                "on_execution_report: exit %s cl_ord_id=%s reason=%s fill_px=%.2f qty=%d",
                "FILLED" if ord_status == "2" else "PARTIAL",
                cl_ord_id,
                reason,
                fill_px,
                incremental_qty,
            )

            # Do not mutate tracker position on partial exit fills.
            # Tracker.record_close currently closes the full position; applying it on
            # partial fills would flatten the position too early and corrupt PnL.
            if ord_status == "1":
                return

            # On full fill, close the remaining local position quantity.
            close_qty = self._tracker.position.quantity if not self._tracker.is_flat else cum_qty
            fill_timestamp = self._extract_broker_timestamp(report)
            self._tracker.record_close(
                fill_price=fill_px,
                qty=close_qty,
                timestamp=fill_timestamp,
                exit_reason=reason,
            )
            if ord_status == "2":
                self._pending_exits.pop(cl_ord_id)
                self._cum_fills.pop(cl_ord_id, None)
            return

        logger.debug("on_execution_report: unknown cl_ord_id=%s (status=%s)", cl_ord_id, ord_status)

    # ------------------------------------------------------------------
    # Open order reconciliation
    # ------------------------------------------------------------------

    def sync_open_orders(self, orders: list) -> None:
        """Populate order state from broker's open orders during reconciliation.

        Classifies each open order as either an entry or exit based on the current
        position side. Exit orders have side opposite to position; entry orders
        have side matching the position direction.

        Called by Reconciler during startup and periodic reconciliation.

        Args:
            orders: List of open order dicts from broker API.
        """
        position = self._tracker.position
        exit_side: str | None = None
        if position.is_long:
            exit_side = "SELL"
        elif position.is_short:
            exit_side = "BUY"

        self._pending_exits.clear()
        self._cum_fills.clear()
        self._pending_entries.clear()

        for order in orders:
            cl_ord_id: str = str(order.get("clOrdId") or order.get("cl_ord_id", ""))
            if not cl_ord_id:
                logger.warning("sync_open_orders: order missing cl_ord_id - skipping: %s", order)
                continue

            order_side: str = str(order.get("side", "")).upper()
            if order_side == "1":
                order_side = "BUY"
            elif order_side == "2":
                order_side = "SELL"

            ord_status: str = str(order.get("ordStatus") or order.get("ord_status", ""))
            if ord_status in ("2", "4", "8"):
                continue

            if exit_side and order_side == exit_side:
                reason = order.get("text") or order.get("reason") or "reconciled_exit"
                self._pending_exits[cl_ord_id] = str(reason)
                logger.info(
                    "sync_open_orders: classified as EXIT cl_ord_id=%s side=%s reason=%s",
                    cl_ord_id,
                    order_side,
                    reason,
                )
            else:
                qty = int(float(order.get("orderQty") or order.get("order_qty") or 0))
                cum_qty = int(order.get("cumQty") or order.get("cum_qty") or 0)
                self._cum_fills[cl_ord_id] = cum_qty
                self._pending_entries[cl_ord_id] = {
                    "side": "LONG" if order_side == "BUY" else "SHORT",
                    "qty": qty,
                    "stop_loss": None,
                    "take_profit": None,
                }
                logger.info(
                    "sync_open_orders: classified as ENTRY cl_ord_id=%s side=%s qty=%d cum_qty=%d",
                    cl_ord_id,
                    order_side,
                    qty,
                    cum_qty,
                )

        logger.info(
            "sync_open_orders: %d pending exits, %d entry orders tracked",
            len(self._pending_exits),
            len(self._cum_fills),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gen_cl_ord_id(prefix: str) -> str:
        """Generate a unique client order ID with prefix (E=entry, X=exit)."""
        return f"{prefix}-{uuid.uuid4().hex[:12].upper()}"

    @staticmethod
    def _extract_broker_timestamp(kwargs: dict[str, Any]) -> datetime:
        """Extract fill timestamp from broker execution report.

        Prefers broker-provided timestamps (transact_time or timestamp fields)
        over datetime.now() for deterministic audit trails and replay capability.

        Args:
            kwargs: Execution report fields from broker.

        Returns:
            Broker timestamp if available, otherwise current time.
        """
        fill_timestamp = kwargs.get("transact_time") or kwargs.get("timestamp") or datetime.now()
        if isinstance(fill_timestamp, str):
            try:
                fill_timestamp = datetime.fromisoformat(fill_timestamp)
            except (ValueError, TypeError):
                fill_timestamp = datetime.now()
        return fill_timestamp

    @staticmethod
    def _extract_fill_price(kwargs: dict[str, Any], fallback: float) -> float:
        """Extract execution price for the current fill.

        Prefers last-fill price fields when present; falls back to avg_px.
        """
        candidates = (
            kwargs.get("last_px"),
            kwargs.get("lastPx"),
            kwargs.get("last_price"),
            kwargs.get("lastPrice"),
            kwargs.get("exec_px"),
            kwargs.get("execPx"),
            kwargs.get("price"),
            kwargs.get("px"),
        )

        for value in candidates:
            if value is None:
                continue

            try:
                px = float(value)
                if px > 0:
                    return px
            except (TypeError, ValueError):
                continue

        return fallback

    @staticmethod
    def _normalize_execution_report_fields(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize execution report payload to snake_case keys.

        Supports common broker payload variants such as camelCase and snake_case.
        """

        def _first(*keys: str) -> Any:
            for key in keys:
                if key in kwargs and kwargs[key] is not None:
                    return kwargs[key]
            return None

        normalized = dict(kwargs)

        normalized["cl_ord_id"] = _first("cl_ord_id", "clOrdId", "clientOrderId") or ""
        status_raw = _first("ord_status", "ordStatus", "status") or ""
        status_text = str(status_raw).strip().upper().replace("-", "_")
        status_map = {
            "NEW": "0",
            "PARTIALLY_FILLED": "1",
            "PARTIAL_FILL": "1",
            "FILLED": "2",
            "CANCELED": "4",
            "CANCELLED": "4",
            "REJECTED": "8",
            "PENDING_NEW": "A",
            "PENDING_CANCEL": "6",
        }
        normalized["ord_status"] = status_map.get(status_text, status_raw)
        normalized["avg_px"] = _first("avg_px", "avgPx", "averagePrice") or 0.0
        normalized["cum_qty"] = _first("cum_qty", "cumQty", "filledQuantity") or 0

        # Mirror common fill-price aliases so _extract_fill_price can pick them up.
        if "last_px" not in normalized:
            normalized["last_px"] = _first("last_px", "lastPx", "lastPrice", "execPrice")

        return normalized
