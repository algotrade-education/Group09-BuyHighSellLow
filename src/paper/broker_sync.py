"""Broker state synchronization helpers for the paper engine."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paperbroker.client import PaperBrokerClient

    from src.paper.order_manager import OrderManager
    from src.paper.position_tracker import PositionTracker


def sync_broker_state(
    *,
    client: "PaperBrokerClient",
    symbol: str,
    tracker: "PositionTracker",
    order_manager: "OrderManager",
    logger: logging.Logger,
) -> None:
    """
    Fetch current portfolio and orders from PaperBroker to initialize
    PositionTracker and OrderManager on startup.
    """
    logger.info("Synchronizing broker state for %s...", symbol)

    try:
        portfolio_response = client.get_portfolio_by_sub()
        if portfolio_response.get("success"):
            items = portfolio_response.get("items", [])
            for item in items:
                if item.get("instrument") == symbol:
                    qty = float(item.get("quantity", 0))
                    avg_price = float(
                        item.get("avgPrice") or item.get("totalCost", 0) / (qty or 1)
                    )
                    tracker.sync_position(qty, avg_price)
                    logger.info("Synced Position: %s @ %.2f", qty, avg_price)
                    break
        else:
            logger.warning(
                "Failed to sync portfolio: %s", portfolio_response.get("error")
            )
    except Exception as exc:
        logger.error("Error syncing portfolio: %s", exc)

    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        orders_response = client.get_orders(today_str, today_str)
        if orders_response.get("success"):
            orders = orders_response.get("items", [])
            order_manager.sync_open_orders(orders)
        else:
            logger.warning("Failed to sync orders: %s", orders_response.get("error"))
    except Exception as exc:
        logger.error("Error syncing orders: %s", exc)
