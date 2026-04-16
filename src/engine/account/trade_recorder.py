"""
Trade recording and history management.

Separated from AccountState to follow Single Responsibility Principle.

Why separate P&L calculation from Position?
- Position tracks UNREALIZED P&L for open positions (live, changes every bar)
- TradeRecorder calculates REALIZED P&L for closed trades (final, immutable)

This separation ensures:
1. Position focuses on live position state and risk management
2. TradeRecorder focuses on historical trade records and performance tracking
3. No confusion between unrealized (floating) and realized (locked-in) P&L
4. Clean separation of concerns for testing and maintenance

The P&L calculation logic is intentionally duplicated because they serve
different purposes and operate on different data at different times.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src.engine.account.position import Position, PositionSide
from src.engine.execution.order import Order
from src.metrics.trade_metrics import Trade, TradeSide

logger = logging.getLogger(__name__)


class TradeRecorder:
    """
    Records and manages trade history.

    Responsibilities:
    - Create trade records from orders
    - Track current open trade
    - Finalize trades on close
    - Provide thread-safe access for async

    Usage:
        recorder = TradeRecorder()
        trade = recorder.open_trade(order, timestamp)
        recorder.close_trade(trade, exit_price, timestamp, reason)
        all_trades = recorder.get_trades()
    """

    def __init__(
        self,
        contract_multiplier: float,
        enable_async_safety: bool = False,
    ) -> None:
        """
        Args:
            contract_multiplier: Contract multiplier for P&L calculation
            enable_async_safety: Enable asyncio.Lock for paper trading
        """
        self.contract_multiplier = contract_multiplier
        self._trades: list[Trade] = []
        self._trade_counter: int = 0
        self._current_trade: Trade | None = None
        self._enable_async_safety = enable_async_safety
        self._lock: asyncio.Lock | None = None

    def reset(self) -> None:
        """Reset all trade history."""
        self._trades = []
        self._trade_counter = 0
        self._current_trade = None

    def open_trade(
        self,
        order: Order,
        timestamp: datetime,
    ) -> Trade:
        """
        Create new trade record from order.

        Args:
            order: Filled order
            timestamp: Entry timestamp

        Returns:
            New Trade record
        """
        self._trade_counter += 1
        side = TradeSide.LONG if order.is_buy else TradeSide.SHORT

        trade = Trade(
            trade_id=str(self._trade_counter),
            side=side,
            entry_time=timestamp,
            entry_price=order.filled_price or 0.0,
            quantity=order.quantity,
            commission=order.commission,
            stop_loss=order.stop_loss or 0.0,
            take_profit=order.take_profit or 0.0,
        )

        self._trades.append(trade)
        self._current_trade = trade

        logger.info("Trade opened: %s @ %.2f", side, order.filled_price)
        return trade

    def close_trade(
        self,
        position: Position,
        exit_price: float,
        exit_commission: float,
        timestamp: datetime,
        exit_reason: str = "",
    ) -> Trade | None:
        """
        Finalize current trade with exit details.

        Calculates realized P&L for the closed trade. This is separate from
        position.unrealized_pnl because:
        - Position P&L is live and changes every bar (mark-to-market)
        - Trade P&L is final and immutable (actual execution prices)
        - Trade P&L includes both entry and exit commissions
        - Trade P&L is used for performance metrics and reporting

        Args:
            position: Position being closed (for MAE/MFE)
            exit_price: Exit price
            exit_commission: Commission for exit
            timestamp: Exit timestamp
            exit_reason: Reason for exit

        Returns:
            Closed Trade record, or None if no trade open
        """
        if self._current_trade is None:
            logger.warning("No current trade to close")
            return None

        trade = self._current_trade
        side = position.side
        qty = position.quantity
        entry_px = position.entry_price

        # Calculate realized P&L (final, immutable)
        # This differs from position.unrealized_pnl which is live and changes every bar
        gross_pnl = (
            (exit_price - entry_px) * qty * self.contract_multiplier
            if side == PositionSide.LONG
            else (entry_px - exit_price) * qty * self.contract_multiplier
        )
        total_commission = trade.commission + exit_commission
        net_pnl = gross_pnl - total_commission

        # Update trade record
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.commission = total_commission
        trade.gross_pnl = gross_pnl
        trade.pnl = net_pnl
        trade.exit_reason = exit_reason
        trade.mae = position.mae
        trade.mfe = position.mfe

        logger.info("Trade closed: pnl=%.0f, reason=%s", net_pnl, exit_reason)

        self._current_trade = None
        return trade

    def get_trades(self) -> list[Trade]:
        """Get copy of all trades (sync-safe)."""
        return list(self._trades)

    async def get_trades_async(self) -> list[Trade]:
        """Get copy of all trades (async-safe)."""
        if not self._enable_async_safety:
            return list(self._trades)

        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            return list(self._trades)

    @property
    def current_trade(self) -> Trade | None:
        """Get current open trade."""
        return self._current_trade

    @property
    def trade_count(self) -> int:
        """Get total number of trades."""
        return len(self._trades)
