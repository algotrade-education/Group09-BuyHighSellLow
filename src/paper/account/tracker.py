"""
Account state tracking for paper trading.

Wraps the core AccountState engine with live trading extensions:
- FIX execution report integration (record_open/record_close)
- Unrealized P&L updates on each bar
- Daily P&L tracking with automatic reset on new trading days
- Broker reconciliation (sync_position/sync_cash)
- Equity snapshot tracking with timestamp deduplication

The tracker maintains position state, cash balance, trade history, and equity
snapshots. It supports scale-in entries and handles broker synchronization
without polluting the trade history with reconciliation artifacts.
"""

from __future__ import annotations

import logging
from datetime import datetime

from src.engine.account.account import AccountState
from src.engine.account.position import Position, PositionSide
from src.engine.execution.order import Order, OrderSide, OrderType
from src.metrics.trade_metrics import Trade, TradeSide
from src.strategy.base import PositionSnapshot

logger = logging.getLogger(__name__)


def _make_synced_trade(
    side: PositionSide,
    entry_time: datetime,
    entry_price: float,
    quantity: int,
    commission: float,
) -> Trade:
    """Create a sentinel Trade for broker-synced positions.

    Returns a Trade marked with is_synced=True metadata, which is excluded
    from the tracker.trades property to avoid polluting trade history with
    reconciliation artifacts.

    Args:
        side: Position side (LONG or SHORT).
        entry_time: Position entry timestamp.
        entry_price: Average entry price from broker.
        quantity: Position quantity.
        commission: Entry commission.

    Returns:
        Sentinel Trade object for internal tracking only.
    """
    trade_side = TradeSide.LONG if side == PositionSide.LONG else TradeSide.SHORT

    return Trade(
        trade_id="__synced__",
        side=trade_side,
        entry_time=entry_time,
        entry_price=entry_price,
        quantity=quantity,
        commission=commission,
        metadata={"is_synced": True},
    )


class Tracker:
    """Live account tracker wrapping the core AccountState engine.

    Provides live trading extensions on top of the base AccountState:
    - FIX execution report handling
    - Bar-by-bar unrealized P&L updates
    - Daily P&L tracking with automatic resets
    - Broker state synchronization
    - Equity snapshot management with deduplication

    The tracker ensures that broker-synced positions don't pollute the trade
    history by using sentinel Trade objects marked with is_synced metadata.
    """

    def __init__(
        self,
        initial_capital: float,
        commission_rate: float,
        contract_multiplier: float,
    ) -> None:
        """Initialize the tracker with account parameters.

        Args:
            initial_capital: Starting cash balance.
            commission_rate: Commission rate per contract (e.g. 0.0001 for 0.01%).
            contract_multiplier: Contract size multiplier for P&L calculation.
        """
        self._state = AccountState(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            contract_multiplier=contract_multiplier,
            enable_async_safety=False,
        )
        self._synced_position: bool = False
        self._equity_snapshots: dict[datetime, float] = {}
        self._snapshot_order: list[datetime] = []

    # ------------------------------------------------------------------
    # Core wiring - record_open / record_close
    # ------------------------------------------------------------------

    def record_open(
        self,
        fill_price: float,
        qty: int,
        side: str,
        timestamp: datetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        """Record an entry fill from a FIX execution report.

        Supports scale-ins: if a position is already open on the same side,
        updates the weighted average entry price instead of opening a new position.

        Args:
            fill_price: Execution price from the fill report.
            qty: Filled quantity (number of contracts).
            side: Position side ("BUY"/"LONG" or "SELL"/"SHORT").
            timestamp: Fill timestamp from broker.
            stop_loss: Optional stop loss price.
            take_profit: Optional take profit price.
        """
        pos = self._state.position
        order_side = OrderSide.BUY if side.upper() in ("BUY", "LONG") else OrderSide.SELL

        if pos.is_flat:
            order = Order(
                order_type=OrderType.MARKET,
                side=order_side,
                quantity=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_at=timestamp,
            )
            commission = self._state._calc_commission(fill_price, qty)
            order.fill(
                price=fill_price,
                timestamp=timestamp,
                commission=commission,
                slippage=0.0,
            )
            self._state.portfolio.deduct_cash(commission)
            self._state._open_position(order, timestamp)
        else:
            # Normalize side input to LONG/SHORT for comparison
            # Accept both BUY/LONG and SELL/SHORT as documented
            normalized_side = "LONG" if side.upper() in ("BUY", "LONG") else "SHORT"
            if pos.side.value.upper() != normalized_side:
                logger.warning(
                    "record_open: reversing position is not supported - ignoring. "
                    "open_side=%s fill_side=%s",
                    pos.side,
                    side,
                )
                return

            old_qty = pos.quantity
            new_qty = old_qty + qty
            pos.entry_price = (pos.entry_price * old_qty + fill_price * qty) / new_qty
            pos.quantity = new_qty
            if stop_loss is not None:
                pos.stop_loss = stop_loss
            if take_profit is not None:
                pos.take_profit = take_profit

            commission = self._state._calc_commission(fill_price, qty)
            self._state.portfolio.deduct_cash(commission)

            if self._state.trade_recorder._current_trade:
                t = self._state.trade_recorder._current_trade
                t.entry_price = pos.entry_price
                t.quantity = new_qty
                t.commission += commission

        self._synced_position = False
        logger.info(
            "record_open: %s %d @ %.2f | SL=%s | TP=%s | cash=%.0f",
            side,
            qty,
            fill_price,
            stop_loss,
            take_profit,
            self._state.portfolio.cash,
        )

    def record_close(
        self,
        fill_price: float,
        qty: int,
        timestamp: datetime,
        exit_reason: str = "",
    ) -> Trade | None:
        """Record a position close via FIX execution report.

        Delegates to AccountState.close_position with exact fill price (no slippage).

        Args:
            fill_price: Execution price from the fill report.
            qty: Filled quantity (number of contracts).
            timestamp: Fill timestamp from broker.
            exit_reason: Reason for exit (e.g. "Stop Loss", "Take Profit").

        Returns:
            Closed Trade object, or None if no position was open.
        """
        if self._state.position.is_flat or self._state.trade_recorder._current_trade is None:
            logger.warning(
                "record_close called but no open position - ignoring. fill_price=%.2f reason=%s",
                fill_price,
                exit_reason,
            )
            return None

        if qty <= 0:
            logger.warning(
                "record_close called with non-positive qty=%s - ignoring. reason=%s",
                qty,
                exit_reason,
            )
            return None

        position = self._state.position
        close_qty = min(int(qty), int(position.quantity))
        current_trade = self._state.trade_recorder._current_trade

        # Partial close path: realize P&L/cash incrementally, keep trade open.
        if close_qty < position.quantity:
            exit_commission = self._state._calc_commission(fill_price, close_qty)
            if position.is_long:
                gross_pnl = (
                    (fill_price - position.entry_price)
                    * close_qty
                    * self._state.contract_multiplier
                )
            else:
                gross_pnl = (
                    (position.entry_price - fill_price)
                    * close_qty
                    * self._state.contract_multiplier
                )

            self._state.portfolio.deduct_cash(exit_commission)
            self._state.portfolio.add_cash(gross_pnl)

            # Accumulate realized partial components so final trade stats remain correct.
            current_trade.metadata["_partial_realized_gross"] = (
                float(current_trade.metadata.get("_partial_realized_gross", 0.0)) + gross_pnl
            )
            current_trade.metadata["_partial_exit_commission"] = (
                float(current_trade.metadata.get("_partial_exit_commission", 0.0)) + exit_commission
            )
            current_trade.metadata["_partial_closed_qty"] = (
                int(current_trade.metadata.get("_partial_closed_qty", 0)) + close_qty
            )

            position.quantity -= close_qty
            # Note: current_trade.quantity keeps original qty; close_position will use remaining position.quantity
            self._state.portfolio.update_equity(position)
            self._synced_position = False

            logger.info(
                "record_close partial: qty=%d/%d @ %.2f reason=%s | remaining=%d",
                close_qty,
                close_qty + position.quantity,
                fill_price,
                exit_reason,
                position.quantity,
            )
            return None

        # Full close path.
        pnl_before_adjust = 0.0
        trade = self._state.close_position(
            exit_price=fill_price,
            timestamp=timestamp,
            exit_reason=exit_reason,
            apply_slippage=False,
        )

        if trade is not None:
            pnl_before_adjust = trade.pnl

            partial_gross = float(trade.metadata.pop("_partial_realized_gross", 0.0))
            partial_exit_commission = float(trade.metadata.pop("_partial_exit_commission", 0.0))
            partial_closed_qty = int(trade.metadata.pop("_partial_closed_qty", 0))

            if partial_gross != 0.0 or partial_exit_commission != 0.0:
                trade.gross_pnl += partial_gross
                trade.commission += partial_exit_commission
                trade.pnl = trade.gross_pnl - trade.commission

                # Restore original total quantity (close_position used remaining qty)
                trade.quantity += partial_closed_qty

                # close_position already recorded remaining-leg PnL.
                # Add only the delta from partial realizations.
                self._state.risk_manager.record_trade_pnl(
                    trade.pnl - pnl_before_adjust,
                    self._state.portfolio.equity,
                )

        self._synced_position = False
        return trade

    # ------------------------------------------------------------------
    # Bar-by-bar updates
    # ------------------------------------------------------------------

    def update_unrealized(self, current_price: float) -> None:
        """Update unrealized P&L and equity based on current market price."""
        self._state.update_equity(current_price)

    def update_daily_pnl(self, bar_time: datetime) -> None:
        """Reset daily P&L tracking when a new trading day starts."""
        self._state.update_daily(bar_time)

    # ------------------------------------------------------------------
    # Broker sync stubs - fully implemented in tasks 4.2 / 4.3
    # ------------------------------------------------------------------

    def sync_position(self, qty: float, avg_price: float) -> None:
        """Sync position from broker without polluting trade history.

        When broker reports a non-zero position, sets position fields directly
        and creates a sentinel Trade marked with is_synced=True (excluded from
        trade history). When broker reports qty=0, flattens any stale local position.

        This ensures broker reconciliation doesn't create artificial trade records
        while keeping position state accurate.

        Args:
            qty: Broker position quantity (positive=LONG, negative=SHORT, 0=flat).
            avg_price: Broker average entry price.
        """
        qty = int(qty)
        if qty == 0:
            # Broker is flat - ensure local position is also flat
            if not self._state.position.is_flat:
                logger.warning(
                    "sync_position: broker flat but local has position - flattening local"
                )
                self._state.position.close()
                self._state.trade_recorder._current_trade = None
                self._synced_position = False
                logger.info("sync_position: broker flat, local synced to flat")
            else:
                logger.info("sync_position: broker flat, local already flat - no-op")
            return

        side = PositionSide.LONG if qty > 0 else PositionSide.SHORT
        abs_qty = abs(qty)

        # Check if position has changed since last sync (idempotence check)
        pos = self._state.position
        position_changed = (
            pos.side != side
            or pos.quantity != abs_qty
            or abs(pos.entry_price - avg_price) > 0.01  # Allow small price differences
        )

        # Broker-synced positions are historical state, so we must not re-charge
        # entry commission locally. Broker cash already reflects past fees.
        commission = 0.0

        if not position_changed and self._synced_position:
            logger.debug("sync_position: position unchanged, keeping synced state")
            return

        # Update position state
        pos.side = side
        pos.entry_price = avg_price
        pos.quantity = abs_qty
        pos.entry_time = datetime.now()
        pos.stop_loss = None
        pos.take_profit = None
        pos.unrealized_pnl = 0.0
        pos._best_price = avg_price
        pos._worst_price = avg_price

        # Create sentinel trade with zero entry commission to avoid double-counting
        # historical fees when this synced position is later closed.
        self._state.trade_recorder._current_trade = _make_synced_trade(
            side=side,
            entry_time=pos.entry_time,
            entry_price=avg_price,
            quantity=abs_qty,
            commission=commission,
        )

        self._synced_position = True
        logger.info(
            "sync_position: side=%s qty=%d avg_price=%.2f commission=%.2f",
            side,
            abs_qty,
            avg_price,
            commission,
        )

    def sync_cash(self, cash: float) -> None:
        """Sync cash balance from broker and recompute equity.

        Args:
            cash: Current cash balance from broker.
        """
        self._state.portfolio.add_cash(cash - self._state.portfolio.cash)

        # Update equity based on current position
        # Note: For positions, we can't calculate accurate unrealized P&L during
        # reconciliation because we don't have the current market price yet.
        # The equity will be properly updated when the first bar arrives.
        # For now, just update the portfolio equity with current position state.
        self._state.portfolio.update_equity(self._state.position)

        logger.info("sync_cash: cash=%.2f, equity=%.2f", cash, self._state.portfolio.equity)

    def set_initial_capital(self, capital: float) -> None:
        """Set the initial capital baseline after reconciliation.

        This should be called after reconciling with the broker to set the
        actual starting equity as the baseline for P&L calculations.

        Args:
            capital: Initial capital/equity to use as baseline.
        """
        self._state.portfolio.initial_capital = capital
        logger.info("set_initial_capital: baseline set to %.2f", capital)

    def equity_snapshot(self, ts: datetime) -> None:
        """Store equity snapshot with timestamp deduplication.

        Only the last equity value per timestamp is kept. Maintains insertion
        order for chronological replay.

        Args:
            ts: Snapshot timestamp (typically bar datetime).
        """
        equity = self._state.equity
        if ts not in self._equity_snapshots:
            self._snapshot_order.append(ts)
        self._equity_snapshots[ts] = equity

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def synced_position(self) -> bool:
        """Return True if the current position was set via broker sync."""
        return self._synced_position

    @property
    def initial_capital(self) -> float:
        """Return the initial capital the tracker was created with."""
        return self._state.portfolio.initial_capital

    @property
    def equity(self) -> float:
        """Return current equity (cash + unrealized P&L)."""
        return self._state.equity

    @property
    def cash(self) -> float:
        """Return current cash balance."""
        return self._state.portfolio.cash

    @property
    def daily_pnl(self) -> float:
        """Return cumulative P&L for the current trading day."""
        return self._state.risk_manager._daily_pnl

    @property
    def is_flat(self) -> bool:
        """Return True when no position is open."""
        return self._state.position.is_flat

    @property
    def position(self) -> Position:
        """Return current open position (or FLAT position if none)."""
        return self._state.position

    @property
    def position_snapshot(self) -> PositionSnapshot:
        """Return immutable snapshot of current position for strategy signal generation.

        Delegates to AccountState.position_snapshot, which is the canonical
        implementation shared with the backtesting engine.
        """
        return self._state.position_snapshot

    @property
    def trades(self) -> list[Trade]:
        """Return all recorded trades, excluding broker-synced sentinels."""
        return [t for t in self._state.trades if not t.metadata.get("is_synced")]

    @property
    def equity_snapshots(self) -> list[tuple[datetime, float]]:
        """Return ordered list of (timestamp, equity) snapshots."""
        return [(ts, self._equity_snapshots[ts]) for ts in self._snapshot_order]
