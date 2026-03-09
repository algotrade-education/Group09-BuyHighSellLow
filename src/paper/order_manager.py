"""
OrderManager - FIX Order Submission and Execution Handling.

The `OrderManager` serves as the bridge between abstract strategy signals and
the low-level FIX protocol via `PaperBrokerClient`. It tracks the lifecycle of
unfilled orders to ensure accurate state transition.

Key Responsibilities:
1.  **Entry Submission**: Translates `LONG`/`SHORT` signals into `LIMIT` orders.
2.  **Exit Submission**: Dispatches liquidation orders for SL, TP, or EOD exits.
3.  **Execution Processing**: Listens for `fix:execution_report` events and
    extracts fill price and quantity for the `PositionTracker`.
4.  **State Recovery**: Reconstructs the list of pending orders on engine
    restart to avoid duplicate submissions.
5.  **Partial Fill Support**: Increments positions based on `PARTIALLY_FILLED`
    status updates rather than waiting for complete fills.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from src.strategy.base import Signal, TradeSignal

if TYPE_CHECKING:
    from paperbroker.client import PaperBrokerClient
    from src.paper.position_tracker import PositionTracker

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Bridges strategy signals to FIX order execution.

    Usage:
        mgr = OrderManager(client, tracker, symbol, dry_run=False)
        # Wire fill callback:
        client.on("fix:execution_report", mgr.on_execution_report)
        # Submit from engine:
        mgr.submit_entry(signal, qty=1)
        mgr.submit_exit("Stop Loss")
    """

    def __init__(
        self,
        client: "PaperBrokerClient",
        tracker: "PositionTracker",
        symbol: str,
        dry_run: bool = False,
    ):
        """
        Args:
            client:   Connected PaperBrokerClient instance.
            tracker:  PositionTracker to update on fill confirmations.
            symbol:   Full symbol, e.g. 'HNXDS:VN30F2601'.
            dry_run:  If True, log orders but don't send via FIX.
        """
        self._client = client
        self._tracker = tracker
        self._symbol = symbol
        self._dry_run = dry_run

        # Map cl_ord_id → pending signal metadata so we can call tracker on fill
        # Map cl_ord_id → pending signal metadata so we can call tracker on fill
        self._pending_entries: Dict[str, Dict[str, Any]] = {}

        # Map cl_ord_id → exit reason for pending exits
        self._pending_exits: Dict[str, str] = {}

        # Track cumulative filled quantity per cl_ord_id to handle broker updates
        self._cum_fills: Dict[str, int] = {}

        # Callback to notify engine of a confirmed fill (optional)
        self.on_fill: Optional[Callable[[str, float, int], None]] = None

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def submit_entry(
        self,
        signal: TradeSignal,
        qty: int,
        bar: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Submit an entry order derived from a TradeSignal.

        Args:
            signal: TradeSignal with signal=LONG or SHORT.
            qty:    Number of contracts to trade.
            bar:    Current bar dict (used for mid-price fallback if entry_price=0).

        Returns:
            clOrdId string if submitted, None on failure or dry-run.
        """
        if signal.signal not in (Signal.LONG, Signal.SHORT):
            return None

        side = "BUY" if signal.is_long else "SELL"
        # Use MARKET order: price is still required by the broker for LIMIT.
        # entry_price=0 in ORB means market order; use current close as limit reference.
        price = signal.entry_price
        if price <= 0 and bar is not None:
            price = float(bar.get("close", 0))

        logger.info(
            "%sSubmitting ENTRY: %s %d %s @ %.2f | SL=%.2f | TP=%.2f",
            "[DRY-RUN] " if self._dry_run else "",
            side,
            qty,
            self._symbol,
            price,
            signal.stop_loss or 0,
            signal.take_profit or 0,
        )

        if self._dry_run:
            # Simulate an immediate fill for dry-run mode
            self._tracker.record_open(
                fill_price=price or 1300.0,
                qty=qty,
                side="LONG" if signal.is_long else "SHORT",
                timestamp=datetime.now(),
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )
            return None

        try:
            cl_ord_id = self._client.place_order(
                full_symbol=self._symbol,
                side=side,
                qty=qty,
                price=price,
                ord_type="LIMIT",
                tif="GTC",
            )
            # Store metadata so on_execution_report can update the tracker
            self._pending_entries[cl_ord_id] = {
                "side": "LONG" if signal.is_long else "SHORT",
                "qty": qty,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
            }
            logger.info("Entry order submitted: clOrdId=%s", cl_ord_id)
            return cl_ord_id
        except Exception as exc:
            logger.error("Failed to submit entry order: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def submit_exit(
        self,
        reason: str = "Manual",
        price: Optional[float] = None,
    ) -> Optional[str]:
        """
        Submit an exit order to close the current position.

        Args:
            reason: Human-readable exit reason (e.g. 'Stop Loss', 'EOD').
            price:  Limit price. If None, uses SL/TP from position or entry_price fallback.

        Returns:
            clOrdId string if submitted, None on failure or dry-run.
        """
        if self._tracker.is_flat:
            logger.debug("submit_exit called but position is already flat.")
            return None

        pos = self._tracker.position
        qty = pos.quantity
        # To close: LONG → SELL, SHORT → BUY
        side = "SELL" if pos.is_long else "BUY"

        # Use SL/TP price if relevant, else the provided price, else entry_price as fallback
        if reason == "Stop Loss" and pos.stop_loss:
            price = pos.stop_loss
        elif reason == "Take Profit" and pos.take_profit:
            price = pos.take_profit

        if price is None:
            # Dangerous fallback: if we don't have a market price, paper traders
            # often use entry_price which might be far away.
            price = pos.entry_price

        logger.info(
            "%sSubmitting EXIT (%s): %s %d %s @ %.2f",
            "[DRY-RUN] " if self._dry_run else "",
            reason,
            side,
            qty,
            self._symbol,
            price,
        )

        if self._dry_run:
            self._tracker.record_close(
                fill_price=price,
                qty=qty,
                timestamp=datetime.now(),
                exit_reason=reason,
            )
            return None

        try:
            cl_ord_id = self._client.place_order(
                full_symbol=self._symbol,
                side=side,
                qty=qty,
                price=price,
                ord_type="LIMIT",
                tif="GTC",
            )
            self._pending_exits[cl_ord_id] = reason
            logger.info("Exit order submitted: clOrdId=%s (%s)", cl_ord_id, reason)
            return cl_ord_id
        except Exception as exc:
            logger.error("Failed to submit exit order: %s", exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # FIX execution report handler
    # ------------------------------------------------------------------

    def on_execution_report(self, **kwargs: Any) -> None:
        """
        Main Event Handler for Broker Execution Reports.

        This method is triggered by the `fix:execution_report` event. It
        interprets FIX `OrdStatus` codes to update the internal trading state.

        Status Handling:
        - **'1' (PARTIALLY_FILLED)**: Records the incremental fill in the
          tracker but keeps the order ID in `_pending_entries`.
        - **'2' (FILLED)**: Records the final fill and clears the order lock.
        - **'4' (CANCELED) / '8' (REJECTED)**: Clears the order lock and
          logs a warning, allowing the strategy to re-evaluate.

        Args:
            **kwargs: Dictionary containing `cl_ord_id`, `ord_status`,
                      `avg_px`, `cum_qty`, and `side`.
        """
        cl_ord_id = kwargs.get("cl_ord_id", "")
        ord_status = str(kwargs.get("ord_status", ""))
        avg_px = float(kwargs.get("avg_px") or 0.0)
        cum_qty = int(kwargs.get("cum_qty") or 0)

        if ord_status in ("4", "8"):  # Canceled or Rejected
            # Release locks
            self._cum_fills.pop(cl_ord_id, None)
            # Release entry locks if canceled/rejected
            if cl_ord_id in self._pending_entries:
                self._pending_entries.pop(cl_ord_id)
                logger.warning(
                    "Entry order %s was %s.",
                    cl_ord_id,
                    "Canceled" if ord_status == "4" else "Rejected",
                )

            # Release exit lock so strategy can try again
            if cl_ord_id in self._pending_exits:
                reason = self._pending_exits.pop(cl_ord_id)
                logger.warning(
                    "Exit order %s (%s) was %s. Lock released.",
                    cl_ord_id,
                    reason,
                    "Canceled" if ord_status == "4" else "Rejected",
                )
            return

        if ord_status not in ("1", "2"):  # Only process PARTIALLY_FILLED or FILLED
            return

        # Calculate the incremental fill amount correctly
        # We track how much we've already recorded for this cl_ord_id
        prev_fill = self._cum_fills.get(cl_ord_id, 0)
        incremental_qty = cum_qty - prev_fill

        if incremental_qty <= 0:
            return

        # Update cumulative tracker
        self._cum_fills[cl_ord_id] = cum_qty

        # --- Entry fill ---
        if cl_ord_id in self._pending_entries:
            meta = self._pending_entries[cl_ord_id]
            logger.info(
                "Entry %s: %s @ %.2f x %d (total %d/%d)",
                "filled" if ord_status == "2" else "partially filled",
                cl_ord_id,
                avg_px,
                incremental_qty,
                cum_qty,
                meta["qty"],
            )
            self._tracker.record_open(
                fill_price=avg_px,
                qty=incremental_qty,
                side=meta["side"],
                timestamp=datetime.now(),
                stop_loss=meta.get("stop_loss"),
                take_profit=meta.get("take_profit"),
            )
            if self.on_fill:
                self.on_fill("entry", avg_px, incremental_qty)

            if ord_status == "2":
                self._pending_entries.pop(cl_ord_id)
                self._cum_fills.pop(cl_ord_id, None)
            return

        # --- Exit fill ---
        if cl_ord_id in self._pending_exits:
            reason = self._pending_exits[cl_ord_id]

            logger.info(
                "Exit %s (%s): %s @ %.2f x %d (total %d)",
                "filled" if ord_status == "2" else "partially filled",
                reason,
                cl_ord_id,
                avg_px,
                incremental_qty,
                cum_qty,
            )
            self._tracker.record_close(
                fill_price=avg_px,
                qty=incremental_qty,  # Pass incremental quantity
                timestamp=datetime.now(),
                exit_reason=reason,
            )
            if self.on_fill:
                self.on_fill("exit", avg_px, incremental_qty)

            if ord_status == "2":
                self._pending_exits.pop(cl_ord_id)
            return

        logger.debug(
            "Unknown execution report for clOrdId=%s (status=%s)", cl_ord_id, ord_status
        )

    def sync_open_orders(self, orders: list) -> None:
        """
        Synchronize known open orders from the broker on startup.
        Filters for New (status '0') and Partially Filled (status '1') orders
        for the tracked symbol.

        Note: Currently assumes open orders are entry orders (for simplicity),
        or exits depending on position state.
        """
        for o in orders:
            sym = o.get("symbol")
            if sym != self._symbol:
                continue

            status = str(o.get("ordStatus"))
            if status not in {"0", "1"}:  # Only New or Partially Filled
                continue

            cl_ord_id = o.get("clOrdId")
            if not cl_ord_id:
                continue

            side_code = o.get("side", "")
            side_str = (
                "BUY" if side_code == "1" else "SELL" if side_code == "2" else "N/A"
            )
            qty = int(float(o.get("orderQty", 0)))

            is_buy = side_code == "1"
            pos = self._tracker.position

            # Determine if this is an entry (opening/scaling-in) or exit
            is_entry = False
            if pos.is_flat:
                is_entry = True
            elif pos.is_long:
                is_entry = is_buy
            elif pos.is_short:
                is_entry = not is_buy

            if is_entry:
                self._pending_entries[cl_ord_id] = {
                    "side": "LONG" if is_buy else "SHORT",
                    "qty": qty,
                    "stop_loss": None,
                    "take_profit": None,
                }
                logger.info(
                    "Resynced pending Entry order: clOrdId=%s (%s %d)",
                    cl_ord_id,
                    side_str,
                    qty,
                )
            else:
                self._pending_exits[cl_ord_id] = "Resynced Exit"
                logger.info(
                    "Resynced pending Exit order: clOrdId=%s (%s %d)",
                    cl_ord_id,
                    side_str,
                    qty,
                )
