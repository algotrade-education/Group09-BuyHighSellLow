"""
Broker state reconciliation for paper trading.

Compares and syncs state between the local Tracker/OrderManager and the broker
REST API. Handles three reconciliation areas:

1. Position reconciliation: Syncs broker portfolio holdings with local position state
2. Cash reconciliation: Syncs broker cash balance with local cash tracking
3. Order reconciliation: Syncs broker open orders with local order state

Each reconciliation method is independent and isolated - failures in one area
don't affect the others. All exceptions are caught and logged to ensure the
trading engine continues operating even when reconciliation encounters issues.

Typically called at startup and periodically during live trading to detect and
correct any drift between local state and broker state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paperbroker.client import PaperBrokerClient

    from src.paper.account.tracker import Tracker
    from src.paper.execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


class Reconciler:
    """Syncs broker state into local Tracker and OrderManager.

    Provides three independent reconciliation methods that can be called
    individually or together via sync(). All methods are fault-tolerant -
    exceptions are caught and logged without propagating to callers.
    """

    def __init__(
        self,
        client: PaperBrokerClient | None,
        tracker: Tracker,
        order_manager: OrderManager,
        symbol: str,
    ) -> None:
        """Initialize the reconciler.

        Args:
            client: Paper broker client for REST API calls.
            tracker: Account tracker to sync position and cash into.
            order_manager: Order manager to sync open orders into.
            symbol: Trading symbol (e.g. 'HNXDS:VN30F2601').
        """
        self._client = client
        self._tracker = tracker
        self._order_manager = order_manager
        self._symbol = symbol

    # ------------------------------------------------------------------
    # Individual reconcile methods (can be called independently)
    # ------------------------------------------------------------------

    def reconcile_position(self) -> None:
        """Compare broker portfolio vs local tracker position and sync if needed.

        Fetches the broker's current portfolio holdings and compares with local
        position state. Always calls tracker.sync_position() to ensure alignment,
        logging a warning when mismatches are detected.

        Handles broker flat positions (qty=0) by flattening stale local positions.
        """
        if not self._client:
            logger.warning("reconcile_position: no broker client available")
            return

        try:
            response = self._client.get_portfolio_by_sub()
            if not response.get("success"):
                logger.warning(
                    "reconcile_position: portfolio fetch unsuccessful: %s",
                    response.get("error"),
                )
                return

            qty: float = 0.0
            avg_price: float = 0.0
            for item in response.get("items", []):
                if item.get("instrument") == self._symbol:
                    qty = float(item.get("quantity", 0))
                    avg_price = float(item.get("avgPrice") or item.get("totalCost", 0) / (qty or 1))
                    break

            broker_qty = qty
            tracker_qty = self._tracker.position.quantity if not self._tracker.is_flat else 0.0

            if broker_qty != tracker_qty:
                logger.warning(
                    "reconcile_position: mismatch detected - broker=%.0f tracker=%.0f; syncing",
                    broker_qty,
                    tracker_qty,
                )

            self._tracker.sync_position(qty, avg_price)
            logger.info(
                "reconcile_position: synced qty=%.0f avg_price=%.2f",
                qty,
                avg_price,
            )

        except Exception:
            logger.error("reconcile_position: broker API error", exc_info=True)

    def reconcile_cash(self) -> None:
        """Sync cash balance from broker into the local tracker."""
        if not self._client:
            logger.warning("reconcile_cash: no broker client available")
            return

        try:
            response = self._client.get_cash_balance()
            if not response or "remainCash" not in response:
                logger.warning("reconcile_cash: response missing 'remainCash': %s", response)
                return

            cash = float(response["remainCash"])
            self._tracker.sync_cash(cash)
            logger.info("reconcile_cash: synced cash=%.2f", cash)

        except Exception:
            logger.error("reconcile_cash: broker API error", exc_info=True)

    def reconcile_orders(self) -> None:
        """Sync today's open orders from broker into the local OrderManager."""
        if not self._client:
            logger.warning("reconcile_orders: no broker client available")
            return

        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            response = self._client.get_orders(today_str, today_str)
            if not response.get("success"):
                logger.warning(
                    "reconcile_orders: orders fetch unsuccessful: %s",
                    response.get("error"),
                )
                return

            orders = response.get("items", [])
            self._order_manager.sync_open_orders(orders)
            logger.info("reconcile_orders: synced %d open orders", len(orders))

        except Exception:
            logger.error("reconcile_orders: broker API error", exc_info=True)

    # ------------------------------------------------------------------
    # Convenience: run all three in sequence (startup / periodic use)
    # ------------------------------------------------------------------

    def sync(self) -> None:
        """Run all three reconciliation steps in sequence.

        Each reconciliation method is fault-tolerant, so failures in one area
        don't prevent the others from running. Safe to call at startup,
        periodically during trading, or when anomalies are detected.
        """
        logger.info("Reconciler.sync: starting for %s", self._symbol)
        self.reconcile_position()
        self.reconcile_cash()
        self.reconcile_orders()
        logger.info("Reconciler.sync: complete")
