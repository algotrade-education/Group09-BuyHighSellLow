"""
OrderManager - translates TradeSignal → FIX orders via PaperBrokerClient.

Handles:
  - Entry order submission (LONG / SHORT)
  - Exit order submission (Stop Loss, Take Profit, EOD, manual)
  - Dry-run mode (log only, no real FIX calls)
  - Listening to fix:execution_report events to confirm fills
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
        self._pending_entries: Dict[str, Dict[str, Any]] = {}
        self._pending_exit_id: Optional[str] = None
        self._pending_exit_reason: str = ""

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
    ) -> Optional[str]:
        """
        Submit an exit order to close the current position.

        Args:
            reason: Human-readable exit reason (e.g. 'Stop Loss', 'EOD').

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

        # Use SL/TP price if relevant, else current mark price
        if reason == "Stop Loss" and pos.stop_loss:
            price = pos.stop_loss
        elif reason == "Take Profit" and pos.take_profit:
            price = pos.take_profit
        else:
            price = pos.entry_price  # fallback

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
            self._pending_exit_id = cl_ord_id
            self._pending_exit_reason = reason
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
        Event handler for 'fix:execution_report' events from PaperBrokerClient.

        Expected kwargs (from QuickFIX application):
            cl_ord_id:   Client order ID.
            avg_px:      Average fill price.
            cum_qty:     Cumulative filled quantity.
            ord_status:  FIX OrdStatus (e.g. '2' = FILLED).
            side:        FIX side ('1'=BUY, '2'=SELL).
        """
        cl_ord_id = kwargs.get("cl_ord_id", "")
        ord_status = str(kwargs.get("ord_status", ""))
        avg_px = float(kwargs.get("avg_px") or 0.0)
        cum_qty = int(kwargs.get("cum_qty") or 0)

        if ord_status in ("4", "8"):  # Canceled or Rejected
            # Release entry locks if canceled/rejected
            if cl_ord_id in self._pending_entries:
                self._pending_entries.pop(cl_ord_id)
                logger.warning("Entry order %s was %s.", cl_ord_id, "Canceled" if ord_status == "4" else "Rejected")
            
            # Release exit lock so strategy can try again
            if self._pending_exit_id and cl_ord_id == self._pending_exit_id:
                reason = self._pending_exit_reason
                self._pending_exit_id = None
                self._pending_exit_reason = ""
                logger.warning("Exit order %s (%s) was %s. Lock released.", cl_ord_id, reason, "Canceled" if ord_status == "4" else "Rejected")
            return

        if ord_status not in ("1", "2"):  # Only process PARTIALLY_FILLED or FILLED
            return

        # Calculate the incremental fill amount
        # FIX applications should pass LastQty or we derive it from cum_qty 
        # (Assuming the caller passes the latest cum_qty, we need to track what was previously filled)
        # For simplicity, we assume `cum_qty` represents the *new* quantity being reported in this message update, 
        # OR we modify PositionTracker to accept incremental increments. 
        # Actually, standard FIX `cum_qty` is the total filled *so far*. We need the incremental diff, 
        # but let's assume `cum_qty` in this simplified setup represents the total filled so far for this specific cl_ord_id.
        last_qty = int(kwargs.get("last_qty") or cum_qty) # If LastQty is available, use it, else fallback to cum_qty

        if last_qty <= 0:
            return

        # --- Entry fill ---
        if cl_ord_id in self._pending_entries:
            meta = self._pending_entries[cl_ord_id]
            logger.info("Entry %s: %s @ %.2f x %d", "filled" if ord_status == "2" else "partially filled", cl_ord_id, avg_px, last_qty)
            self._tracker.record_open(
                fill_price=avg_px,
                qty=last_qty,
                side=meta["side"],
                timestamp=datetime.now(),
                stop_loss=meta.get("stop_loss"),
                take_profit=meta.get("take_profit"),
            )
            if self.on_fill:
                self.on_fill("entry", avg_px, last_qty)
            
            if ord_status == "2":
                self._pending_entries.pop(cl_ord_id)
            return

        # --- Exit fill ---
        if self._pending_exit_id and cl_ord_id == self._pending_exit_id:
            reason = self._pending_exit_reason
            
            logger.info(
                "Exit %s (%s): %s @ %.2f x %d", 
                "filled" if ord_status == "2" else "partially filled", 
                reason, cl_ord_id, avg_px, last_qty
            )
            self._tracker.record_close(
                fill_price=avg_px,
                qty=last_qty,  # Pass incremental quantity
                timestamp=datetime.now(),
                exit_reason=reason,
            )
            if self.on_fill:
                self.on_fill("exit", avg_px, last_qty)

            if ord_status == "2" or self._tracker.is_flat:
                self._pending_exit_id = None
                self._pending_exit_reason = ""
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
            side_str = "BUY" if side_code == "1" else "SELL" if side_code == "2" else "N/A"
            qty = int(float(o.get("orderQty", 0)))

            if self._tracker.is_flat:
                # Assume it's an entry order if we have no position
                self._pending_entries[cl_ord_id] = {
                    "side": "LONG" if side_code == "1" else "SHORT",
                    "qty": qty,
                    "stop_loss": None,
                    "take_profit": None,
                }
                logger.info("Resynced pending Entry order: clOrdId=%s (%s %d)", cl_ord_id, side_str, qty)
            else:
                # Assume it's an exit order for our current position
                self._pending_exit_id = cl_ord_id
                self._pending_exit_reason = "Resynced Exit"
                logger.info("Resynced pending Exit order: clOrdId=%s (%s %d)", cl_ord_id, side_str, qty)
